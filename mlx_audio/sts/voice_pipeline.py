import argparse
import asyncio
import logging

import mlx.core as mx
import numpy as np
import sounddevice as sd
from mlx_lm.generate import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.stt import load as load_stt
from mlx_audio.tts.audio_player import AudioPlayer
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.vad import load as load_vad

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        silence_threshold=0.03,
        silence_duration=1.5,
        input_sample_rate=16_000,
        output_sample_rate=24_000,
        streaming_interval=3,
        frame_duration_ms=32,
        stt_model="mlx-community/whisper-large-v3-turbo-asr-fp16",
        llm_model="Qwen/Qwen2.5-0.5B-Instruct-4bit",
        tts_model="mlx-community/csm-1b-fp16",
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.streaming_interval = streaming_interval
        self.frame_duration_ms = frame_duration_ms

        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model

        self.vad = load_vad("mlx-community/silero-vad")
        self.vad_threshold = 0.5

        # Rolling buffer of current turn audio (float32 arrays), capped at 8s
        self.turn_audio_buffer = []
        self.max_buffer_samples = 16000 * 8  # 8 seconds at 16kHz

        # Smart Turn endpoint detector
        self.smart_turn = load_vad("mlx-community/smart-turn-v3", strict=True)
        self.smart_turn_threshold = 0.5

        self.input_audio_queue = asyncio.Queue(maxsize=50)
        self.transcription_queue = asyncio.Queue()
        self.output_audio_queue = asyncio.Queue(maxsize=50)

        self.llm_loaded = asyncio.Event()
        self.tts_loaded = asyncio.Event()

    async def start(self):
        self.player = AudioPlayer(sample_rate=self.output_sample_rate)
        self.loop = asyncio.get_running_loop()

        tasks = [
            asyncio.create_task(self._listener()),
            asyncio.create_task(self._response_processor()),
            asyncio.create_task(self._audio_output_processor()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # speech detection and transcription

    def _voice_activity_detection(self, frame_np: np.ndarray):
        """Frame must be float32, 512 samples at 16kHz."""
        prob, self.vad_state = self.vad.feed(
            mx.array(frame_np), self.vad_state, sample_rate=16000
        )
        return float(prob.item()) > self.vad_threshold

    async def _listener(self):
        frame_size = 512  # Silero expects 512 samples at 16kHz
        self.vad_state = self.vad.initial_state(sample_rate=16000)

        logger.info(f"Loading speech-to-text model: {self.stt_model}")
        self.stt = load_stt(self.stt_model)
        await self.tts_loaded.wait()
        await self.llm_loaded.wait()

        stream = sd.InputStream(
            samplerate=self.input_sample_rate,
            blocksize=frame_size,
            channels=1,
            dtype="float32",
            callback=self._sd_callback,
        )
        stream.start()

        logger.info("Listening for voice input...")

        frames = []
        silent_frames = 0
        frames_until_silence = int(
            self.silence_duration * 1000 / self.frame_duration_ms
        )
        speaking_detected = False

        try:
            while True:
                frame = await self.input_audio_queue.get()
                is_speech = self._voice_activity_detection(frame)

                if is_speech:
                    speaking_detected = True
                    silent_frames = 0
                    frames.append(frame)

                    # Buffer audio for Smart Turn analysis (Step 2)
                    self.turn_audio_buffer.append(frame)
                    self._trim_turn_buffer()

                    # Cancel the current TTS task
                    if hasattr(self, "current_tts_task") and self.current_tts_task:
                        # Signal the generator loop to stop
                        self.current_tts_cancel.set()

                    # Clear the output audio queue
                    self.loop.call_soon_threadsafe(self.player.flush)
                elif speaking_detected:
                    silent_frames += 1
                    frames.append(frame)

                    if silent_frames > frames_until_silence:
                        # Process the voice input
                        if frames:

                            logger.info("Processing voice input...")
                            text = self._process_audio(frames)
                            if text:
                                logger.info(f"Transcribed: {text}")
                                await self.transcription_queue.put(text)

                        frames = []
                        self.turn_audio_buffer = []
                        speaking_detected = False
                        silent_frames = 0
        except (asyncio.CancelledError, KeyboardInterrupt):
            stream.stop()
            stream.close()
            raise
        finally:
            stream.stop()
            stream.close()

    def _sd_callback(self, indata, frames, _time, status):
        data = indata.reshape(-1).astype(np.float32)

        def _enqueue():
            try:
                self.input_audio_queue.put_nowait(data)
            except asyncio.QueueFull:
                return

        self.loop.call_soon_threadsafe(_enqueue)

    def _trim_turn_buffer(self):
        """Trim turn_audio_buffer to max_buffer_samples (oldest frames drop off)."""
        total_samples = sum(f.size for f in self.turn_audio_buffer)
        while total_samples > self.max_buffer_samples and len(self.turn_audio_buffer) > 1:
            total_samples -= self.turn_audio_buffer[0].size
            self.turn_audio_buffer.pop(0)

    def _process_audio(self, frames):
        audio = np.concatenate(frames)

        result = self.stt.generate(mx.array(audio))
        return result.text.strip()

    # response generation

    async def _response_processor(self):
        logger.info(f"Loading text generation model: {self.llm_model}")
        self.llm, self.tokenizer = load_llm(self.llm_model)
        self.llm_loaded.set()
        while True:
            text = await self.transcription_queue.get()
            await self._generate_response(text)
            self.transcription_queue.task_done()

    async def _generate_response(self, text):
        try:
            logger.info("Generating response...")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. You always respond with short sentences and never use punctuation like parentheses or colons that wouldn't appear in conversational speech.",
                },
                {"role": "user", "content": text},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            response_text = generate_text(
                self.llm, self.tokenizer, prompt, verbose=False
            ).strip()

            logger.info(f"Generated response: {response_text}")

            if response_text:
                self.current_tts_cancel = asyncio.Event()
                self.current_tts_task = asyncio.create_task(
                    self._speak_response(response_text, self.current_tts_cancel)
                )
        except Exception as e:
            logger.error(f"Generation error: {e}")

    # speech generation

    async def _speak_response(self, text: str, cancel_event: asyncio.Event):
        """
        Speak `text`, yielding PCM chunks into `self.output_audio_queue`.
        Playback can be interrupted at any moment by setting `cancel_event`.
        """
        loop = self.loop

        try:
            for chunk in self.tts.generate(
                text,
                sample_rate=self.output_sample_rate,
                stream=True,
                streaming_interval=self.streaming_interval,
                verbose=False,
            ):
                if cancel_event.is_set():  # <-- stop immediately
                    break
                loop.call_soon_threadsafe(
                    self.output_audio_queue.put_nowait, chunk.audio
                )

        except asyncio.CancelledError:
            # The coroutine itself was cancelled from outside → just exit cleanly.
            pass
        except Exception as exc:
            logger.error("Speech synthesis error: %s", exc)

    async def _audio_output_processor(self):
        logger.info(f"Loading text-to-speech model: {self.tts_model}")
        self.tts = load_tts(self.tts_model)  # pyright: ignore[reportArgumentType]
        self.tts_loaded.set()
        try:
            while True:
                audio = await self.output_audio_queue.get()
                self.player.queue_audio(audio)
                self.output_audio_queue.task_done()
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.player.stop()
            raise


async def main():
    parser = argparse.ArgumentParser(description="Voice Pipeline")
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo-asr-fp16",
        help="STT model",
    )
    parser.add_argument(
        "--tts_model", type=str, default="mlx-community/csm-1b-fp16", help="TTS model"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="LLM model",
    )
    parser.add_argument(
        "--silence_duration", type=float, default=1.5, help="Silence duration"
    )
    parser.add_argument(
        "--silence_threshold", type=float, default=0.03, help="Silence threshold"
    )
    parser.add_argument(
        "--streaming_interval", type=int, default=3, help="Streaming interval"
    )
    args = parser.parse_args()

    pipeline = VoicePipeline(
        stt_model=args.stt_model,
        tts_model=args.tts_model,
        llm_model=args.llm_model,
        silence_duration=args.silence_duration,
        silence_threshold=args.silence_threshold,
        streaming_interval=args.streaming_interval,
    )
    await pipeline.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
