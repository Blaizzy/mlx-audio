import argparse
import asyncio
import logging

import mlx.core as mx
import numpy as np
import sounddevice as sd
import webrtcvad
from mlx_lm.generate import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.stt.models.whisper import Model as Whisper
from mlx_audio.tts.audio_player import AudioPlayer
from mlx_audio.tts.utils import load_model as load_tts

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
        frame_duration_ms=30,
        vad_mode=3,
        stt_model="mlx-community/whisper-large-v3-turbo",
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

        self.vad = webrtcvad.Vad(vad_mode)

        self.input_audio_queue = asyncio.Queue(maxsize=50)
        self.transcription_queue = asyncio.Queue()
        self.output_audio_queue = asyncio.Queue(maxsize=50)

        self.mlx_lock = asyncio.Lock()

    async def init_models(self):
        logger.info(f"Loading text generation model: {self.llm_model}")
        self.llm, self.tokenizer = await asyncio.to_thread(
            lambda: load_llm(self.llm_model)
        )

        logger.info(f"Loading text-to-speech model: {self.tts_model}")
        self.tts = await asyncio.to_thread(lambda: load_tts(self.tts_model))

        logger.info(f"Loading speech-to-text model: {self.stt_model}")
        self.stt = Whisper.from_pretrained(self.stt_model)

    async def start(self):
        self.loop = asyncio.get_running_loop()

        await self.init_models()

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

    def _is_silent(self, audio_data):
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = (
                audio_np.astype(np.float32) / 32768.0
            )  # Normalize if input is bytes
        else:
            audio_np = audio_data

        # Ensure audio_np is float32 for energy calculation.
        audio_np = audio_np.astype(np.float32)

        energy = np.linalg.norm(audio_np) / np.sqrt(audio_np.size)
        return energy < self.silence_threshold

    def _voice_activity_detection(self, frame):
        try:
            return self.vad.is_speech(frame, self.input_sample_rate)
        except ValueError:
            # fall back to energy-based detection
            return not self._is_silent(frame)

    async def _listener(self):
        frame_size = int(self.input_sample_rate * (self.frame_duration_ms / 1000.0))
        stream = sd.InputStream(
            samplerate=self.input_sample_rate,
            blocksize=frame_size,
            channels=1,
            dtype="int16",
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
                            await self._process_audio(frames)

                        frames = []
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
        data = indata.reshape(-1).tobytes()

        def _enqueue():
            try:
                self.input_audio_queue.put_nowait(data)
            except asyncio.QueueFull:
                return

        self.loop.call_soon_threadsafe(_enqueue)

    async def _process_audio(self, frames):
        audio = (
            np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        )

        async with self.mlx_lock:
            result = await asyncio.to_thread(self.stt.generate, mx.array(audio))
        text = result.text.strip()

        if text:
            logger.info(f"Transcribed: {text}")
            await self.transcription_queue.put(text)

    # response generation

    async def _response_processor(self):
        while True:
            text = await self.transcription_queue.get()
            await self._generate_response(text)
            self.transcription_queue.task_done()

    async def _generate_response(self, text):
        def _get_llm_response(llm, tokenizer, messages, *, verbose=False):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return generate_text(llm, tokenizer, prompt, verbose=verbose).strip()

        try:
            logger.info("Generating response...")

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. You always respond with short sentences and never use punctuation like parentheses or colons that wouldn't appear in conversational speech.",
                },
                {"role": "user", "content": text},
            ]
            async with self.mlx_lock:
                response_text = await asyncio.to_thread(
                    _get_llm_response, self.llm, self.tokenizer, messages, verbose=False
                )

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

        def _tts_stream(tts, txt, rate, queue, cancel_ev: asyncio.Event):
            # This runs in a worker thread, so we *must* poll a thread‑safe flag.
            for chunk in tts.generate(
                txt,
                sample_rate=rate,
                stream=True,
                streaming_interval=self.streaming_interval,
                verbose=False,
            ):
                if cancel_ev.is_set():  # <-- stop immediately
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk.audio)

        try:
            async with self.mlx_lock:
                await asyncio.to_thread(
                    _tts_stream,
                    self.tts,
                    text,
                    self.output_sample_rate,
                    self.output_audio_queue,
                    cancel_event,
                )
        except asyncio.CancelledError:
            # The coroutine itself was cancelled from outside → just exit cleanly.
            pass
        except Exception as exc:
            logger.error("Speech synthesis error: %s", exc)

    async def _audio_output_processor(self):
        self.player = AudioPlayer(sample_rate=self.output_sample_rate)

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
        default="mlx-community/whisper-large-v3-turbo",
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
    parser.add_argument("--vad_mode", type=int, default=3, help="VAD mode")
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
        vad_mode=args.vad_mode,
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
