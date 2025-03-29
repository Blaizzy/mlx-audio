import argparse
import logging
import os
import queue
import threading
import time
import wave

import mlx.core as mx
import mlx_whisper
import numpy as np
import pyaudio
import webrtcvad
from mlx_lm.utils import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model as load_tts

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        silence_threshold=0.03,  # Threshold for determining silence
        silence_duration=1.5,  # Duration of silence to trigger end of speech
        sample_rate=16000,  # Audio sample rate
        frame_duration_ms=30,  # Duration of each audio frame in ms
        interruptible=True,  # Allow interruptions during generation
        vad_mode=3,  # WebRTC VAD aggressiveness (0-3)
        stt_model="mlx-community/whisper-large-v3-turbo",
        llm_model="Qwen/Qwen2.5-0.5B-Instruct",
        tts_model="mlx-community/Kokoro-82M-bf16",
    ):

        # Audio parameters
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.interruptible = interruptible

        # Models
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model

        # WebRTC Voice Activity Detection
        self.vad = webrtcvad.Vad(vad_mode)

        # Audio player
        self.player = None

        # Audio queues
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # Status flags
        self.recording = False
        self.generating = False
        self.speaking = False
        self.interrupted = False

        # Initialize audio components
        self.p = pyaudio.PyAudio()

        # Initialize speech recognition and generation models
        self._init_models()

        # Thread locks
        self.audio_lock = threading.Lock()
        self.generation_lock = threading.Lock()

    def _init_models(self):
        """Initialize speech-to-text and text-to-speech models"""

        logger.info("Loading text generation model...")
        # Using a smaller model for quick responses
        self.llm, self.tokenizer = load_llm(self.llm_model)

        logger.info("Loading text-to-speech model...")
        # Using TTS for speech synthesis
        try:
            self.tts = load_tts(self.tts_model)
        except Exception as e:
            logger.error(f"TTS model initialization failed: {e}")
            raise e

    def _is_silent(self, audio_data):
        """Detect if an audio chunk is silent"""
        # Convert bytes to numpy array
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_np = audio_data

        # Normalize
        audio_np = audio_np.astype(np.float32) / 32768.0

        # Calculate energy
        energy = np.sqrt(np.mean(audio_np**2))

        return energy < self.silence_threshold

    def _voice_activity_detection(self, frame):
        """Use WebRTC VAD to detect voice activity"""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except:
            # Fallback to energy-based detection
            return not self._is_silent(frame)

    def listen(self):
        """Start listening for voice input"""
        self.recording = True

        # Calculate frame size based on frame duration
        frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))

        # Open audio stream
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frame_size,
        )

        logger.info("Listening for voice input...")

        frames = []
        silent_frames = 0
        speaking_detected = False

        try:
            while self.recording:
                frame = stream.read(frame_size, exception_on_overflow=False)

                # Check if this frame contains speech
                is_speech = self._voice_activity_detection(frame)

                if is_speech:
                    speaking_detected = True
                    silent_frames = 0
                    frames.append(frame)
                elif speaking_detected:
                    silent_frames += 1
                    frames.append(frame)

                    # If silence duration threshold is reached, process the audio
                    if (
                        silent_frames
                        > (self.silence_duration * 1000) / self.frame_duration_ms
                    ):
                        logger.info("Silence detected, processing speech...")
                        if self.player:
                            self.player.stop(force=True)

                        # Process the recorded audio
                        if frames:
                            # Save frames to a temporary WAV file
                            temp_wav = self._save_audio_frames(frames)

                            # Transcribe the audio
                            text = self._transcribe_audio(temp_wav)

                            # Clear the cache
                            mx.metal.clear_cache()

                            if text.strip():
                                logger.info(f"Transcribed: {text}")

                                # Queue the text for response generation
                                self.audio_queue.put(text)

                                # Start generation in a separate thread
                                if not self.generating:
                                    threading.Thread(
                                        target=self._generate_response
                                    ).start()

                            # Clean up temporary file
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)

                        # Reset for next utterance
                        frames = []
                        speaking_detected = False
                        silent_frames = 0

                # Check if there's an interrupt request from generation
                if self.interrupted:
                    logger.info("Recording interrupted for response playback")
                    # Wait until speaking is done
                    while self.speaking:
                        time.sleep(0.1)
                    self.interrupted = False

        except KeyboardInterrupt:
            logger.info("Stopping recording...")
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
            self.recording = False

    def _save_audio_frames(self, frames):
        """Save audio frames to a temporary WAV file"""
        temp_file = "temp_recording.wav"
        wf = wave.open(temp_file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        return temp_file

    def _transcribe_audio(self, audio_file):
        """Transcribe audio to text using Whisper"""
        try:
            result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=self.stt_model)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _generate_response(self):
        """Generate a response from the transcribed text"""
        with self.generation_lock:
            self.generating = True

            try:
                # Get the latest input text
                text = self.audio_queue.get()

                logger.info("Generating response...")

                # Check for an interruption during generation setup
                if self.interrupted:
                    logger.info("Generation interrupted before starting")
                    self.generating = False
                    self.interrupted = False
                    return

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate response text
                response_text = generate_text(
                    self.llm, self.tokenizer, prompt, verbose=True
                )
                # Clear the cache
                mx.metal.clear_cache()

                # Clean up the generated text
                response_text = response_text.strip()

                logger.info(f"Generated response: {response_text}")

                # Queue the response for speech synthesis
                if response_text and not self.interrupted:
                    self.response_queue.put(response_text)
                    threading.Thread(target=self._speak_response).start()

            except Exception as e:
                logger.error(f"Generation error: {e}")
            finally:
                self.generating = False

    def _speak_response(self):
        """Convert the generated text to speech and play it"""
        try:
            self.speaking = True

            # Get the response text
            response_text = self.response_queue.get()

            logger.info("Converting response to speech...")

            # Synthesize speech
            temp_file = "temp_response.wav"

            # generate audio
            self.player = generate_audio(
                response_text,
                model_path=self.tts,
                file_prefix=temp_file,
                sample_rate=24000,
                play=True,
                return_player=True,
            )

            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

            logger.info("Response playback complete")

            # Clear the cache
            mx.metal.clear_cache()

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
        finally:
            self.speaking = False

    def start(self):
        """Start the voice interface"""
        logger.info("Starting voice interface...")

        try:
            # Start listening in a separate thread
            listen_thread = threading.Thread(target=self.listen)
            listen_thread.daemon = True
            listen_thread.start()

            # Keep the main thread alive
            while True:
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Stopping voice interface...")
            self.recording = False
            self.p.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description="Voice interface")
    parser.add_argument(
        "--silence_threshold", type=float, default=0.02, help="Silence threshold"
    )
    parser.add_argument(
        "--silence_duration", type=float, default=1.2, help="Silence duration"
    )
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model",
    )
    parser.add_argument(
        "--llm_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="LLM model"
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        default="mlx-community/Kokoro-82M-bf16",
        help="TTS model",
    )
    # TODO: Add output directory for audio files
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    return parser.parse_args()


def main():
    """Main function to start the voice interface"""
    # Create voice interface with default parameters
    args = parse_args()
    pipeline = VoicePipeline(
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        interruptible=True,
        vad_mode=3,
        stt_model=args.stt_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
    )

    # Start the interface
    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Try to clean up resources
        if hasattr(pipeline, "p") and pipeline.p:
            pipeline.p.terminate()
    finally:
        print("Voice pipeline stopped")


if __name__ == "__main__":
    main()
