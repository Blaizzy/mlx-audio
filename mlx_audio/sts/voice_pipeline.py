import argparse
import asyncio
import json
import logging
import os
import wave
from pathlib import Path

import mlx.core as mx
import mlx_whisper
import numpy as np
import pyaudio
import webrtcvad
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, TcpSocketSignaling, add_signaling_arguments
from av import AudioFrame
from mlx_lm.utils import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model as load_tts

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioTransformTrack(MediaStreamTrack):
    """
    A media stream track that transforms audio frames from a source track.
    This can be used to process audio in real-time for WebRTC connections.
    """
    kind = "audio"

    def __init__(self, track, pipeline):
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.buffer = []
        self.silent_frames = 0
        self.speaking_detected = False
        self.frame_duration_ms = pipeline.frame_duration_ms
        self.silence_duration = pipeline.silence_duration
        self.sample_rate = pipeline.sample_rate
        self._start_time = None

    async def recv(self):
        # Get a frame from the source track
        frame = await self.track.recv()

        # Initialize start time on first frame
        if self._start_time is None:
            self._start_time = frame.time

        # Convert frame to PCM format for VAD
        pcm_frame = frame.to_ndarray().flatten().astype(np.int16).tobytes()

        # Process the audio frame
        is_speech = self.pipeline._voice_activity_detection(pcm_frame)

        if is_speech:
            self.speaking_detected = True
            self.silent_frames = 0
            self.buffer.append(pcm_frame)
        elif self.speaking_detected:
            self.silent_frames += 1
            self.buffer.append(pcm_frame)

            # If silence duration threshold is reached, process the audio
            if self.silent_frames > (self.silence_duration * 1000) / self.frame_duration_ms:
                logger.info("Silence detected in WebRTC stream, processing speech...")

                # Save and process audio
                if self.buffer:
                    # Process in a non-blocking way
                    audio_data = b"".join(self.buffer)
                    asyncio.create_task(self.pipeline._process_webrtc_audio(audio_data))

                # Reset for next utterance
                self.buffer = []
                self.speaking_detected = False
                self.silent_frames = 0

        # Return the original frame (pass-through)
        return frame


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
        webrtc_enabled=False,  # Enable WebRTC functionality
        signaling_server=None,  # Signaling server for WebRTC
        output_dir="output",
        signaling_port=8080
    ):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

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

        # Async queues
        self.audio_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()

        # Status flags
        self.recording = False
        self.generating = False
        self.speaking = False
        self.interrupted = False

        # WebRTC
        self.webrtc_enabled = webrtc_enabled
        self.signaling_server = signaling_server
        self.signaling_port = 8080
        self.peer_connection = None
        self.signaling = None

        # Initialize audio components if not WebRTC only
        if not webrtc_enabled or (webrtc_enabled and not signaling_server):
            self.p = pyaudio.PyAudio()

        # Lock for model initialization
        self.init_lock = asyncio.Lock()
        self.models_initialized = False

        # Locks for audio and generation
        self.audio_lock = asyncio.Lock()
        self.generation_lock = asyncio.Lock()

    async def init_models(self):
        """Initialize speech-to-text and text-to-speech models asynchronously"""
        if self.models_initialized:
            return

        async with self.init_lock:
            if self.models_initialized:
                return

            logger.info("Loading text generation model...")
            # Using a smaller model for quick responses
            # Run in executor to avoid blocking the event loop
            self.llm, self.tokenizer = await asyncio.get_event_loop().run_in_executor(
                None, lambda: load_llm(self.llm_model)
            )

            logger.info("Loading text-to-speech model...")
            # Using TTS for speech synthesis
            try:
                self.tts = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: load_tts(self.tts_model)
                )
            except Exception as e:
                logger.error(f"TTS model initialization failed: {e}")
                raise e

            self.models_initialized = True

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

    async def listen(self):
        """Start listening for voice input asynchronously"""
        await self.init_models()

        self.recording = True

        # Calculate frame size based on frame duration
        frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))

        # Open audio stream through executor to avoid blocking
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=frame_size,
            )
        )

        logger.info("Listening for voice input...")

        frames = []
        silent_frames = 0
        speaking_detected = False

        # Start the response generator
        asyncio.create_task(self._response_processor())

        try:
            while self.recording:
                # Read frame non-blockingly
                frame = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: stream.read(frame_size, exception_on_overflow=False)
                )

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
                            # Stop player asynchronously
                            await asyncio.get_event_loop().run_in_executor(
                                None, lambda: self.player.stop(force=True) if self.player else None
                            )

                        # Process the recorded audio
                        if frames:
                            # Save frames to a temporary WAV file
                            temp_wav = await self._save_audio_frames(frames)

                            # Transcribe the audio
                            text = await self._transcribe_audio(temp_wav)

                            # Clear the cache (run in executor to avoid blocking)
                            await asyncio.get_event_loop().run_in_executor(
                                None, mx.metal.clear_cache
                            )

                            if text.strip():
                                logger.info(f"Transcribed: {text}")

                                # Queue the text for response generation
                                await self.audio_queue.put(text)

                            # Clean up temporary file asynchronously
                            if os.path.exists(temp_wav):
                                await asyncio.get_event_loop().run_in_executor(
                                    None, os.remove, temp_wav
                                )

                        # Reset for next utterance
                        frames = []
                        speaking_detected = False
                        silent_frames = 0

                # Check if there's an interrupt request from generation
                if self.interrupted:
                    logger.info("Recording interrupted for response playback")
                    # Wait until speaking is done
                    while self.speaking:
                        await asyncio.sleep(0.1)
                    self.interrupted = False

                # Small sleep to yield to other tasks
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Stopping recording...")
        finally:
            # Clean up asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: (stream.stop_stream(), stream.close())
            )
            self.recording = False

    async def _save_audio_frames(self, frames, prefix="temp_recording"):
        """Save audio frames to a temporary WAV file asynchronously"""
        timestamp = int(asyncio.get_event_loop().time())
        temp_file = os.path.join(self.output_dir, f"{prefix}_{timestamp}.wav")

        # Run file operations in executor to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._write_wav_file(temp_file, frames)
        )

        return temp_file

    def _write_wav_file(self, filepath, frames):
        """Helper method to write WAV file synchronously (to be run in executor)"""
        wf = wave.open(filepath, "wb")
        wf.setnchannels(1)

        # Handle PyAudio if available
        if hasattr(self, 'p'):
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        else:
            # Default to 2 bytes (16-bit) if PyAudio not available
            wf.setsampwidth(2)

        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

    async def _process_webrtc_audio(self, audio_data):
        """Process audio received from WebRTC asynchronously"""
        # Save audio to a temporary file
        temp_wav = await self._save_audio_frames([audio_data], prefix="webrtc_recording")

        # Transcribe the audio
        text = await self._transcribe_audio(temp_wav)

        # Clear the cache
        await asyncio.get_event_loop().run_in_executor(None, mx.metal.clear_cache)

        if text.strip():
            logger.info(f"WebRTC Transcribed: {text}")

            # Queue the text for response generation
            await self.audio_queue.put(text)

        # Clean up temporary file
        if os.path.exists(temp_wav):
            await asyncio.get_event_loop().run_in_executor(None, os.remove, temp_wav)

    async def _transcribe_audio(self, audio_file):
        """Transcribe audio to text using Whisper asynchronously"""
        try:
            # Run transcription in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: mlx_whisper.transcribe(audio_file, path_or_hf_repo=self.stt_model)
            )
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def _response_processor(self):
        """Process audio queue and generate responses continuously"""
        while True:
            # Wait for next text input in queue
            text = await self.audio_queue.get()

            # Generate response
            await self._generate_response(text)

            # Mark task as done
            self.audio_queue.task_done()

    async def _generate_response(self, text):
        """Generate a response from the transcribed text asynchronously"""
        async with self.generation_lock:
            self.generating = True

            try:
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

                # Apply chat template in executor
                prompt = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )

                # Generate response text in executor
                response_text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: generate_text(self.llm, self.tokenizer, prompt, verbose=True)
                )

                # Clear the cache
                await asyncio.get_event_loop().run_in_executor(None, mx.metal.clear_cache)

                # Clean up the generated text
                response_text = response_text.strip()

                logger.info(f"Generated response: {response_text}")

                # Queue the response for speech synthesis
                if response_text and not self.interrupted:
                    await self.response_queue.put(response_text)

                    # For WebRTC, handle differently
                    if self.webrtc_enabled and self.peer_connection:
                        await self._handle_webrtc_response(response_text)
                    else:
                        # Immediately start speaking response
                        asyncio.create_task(self._speak_response())

            except Exception as e:
                logger.error(f"Generation error: {e}")
            finally:
                self.generating = False

    async def _handle_webrtc_response(self, response_text):
        """Handle response for WebRTC clients asynchronously"""
        try:
            temp_wav = await self._generate_tts_async(response_text)

            # TODO: Stream the audio back through WebRTC
            # This would typically involve creating a MediaPlayer and adding it as a track
            # to the peer connection, but that requires more complex WebRTC handling

            logger.info("Generated audio response for WebRTC client")

        except Exception as e:
            logger.error(f"WebRTC response handling error: {e}")

    async def _generate_tts_async(self, text):
        """Generate TTS asynchronously"""
        timestamp = int(asyncio.get_event_loop().time())
        temp_file = os.path.join(self.output_dir, f"response_{timestamp}")

        # Run TTS in executor to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_audio(
                text,
                model_path=self.tts,
                file_prefix=temp_file,
                sample_rate=24000,
                play=False,
                return_player=False
            )
        )

    async def _speak_response(self):
        """Convert the generated text to speech and play it asynchronously"""
        try:
            self.speaking = True

            # Get the response text
            response_text = await self.response_queue.get()

            logger.info("Converting response to speech...")

            # Synthesize speech
            timestamp = int(asyncio.get_event_loop().time())
            temp_file = os.path.join(self.output_dir, f"response_{timestamp}")

            # Generate audio in executor
            self.player = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_audio(
                    response_text,
                    model_path=self.tts,
                    file_prefix=temp_file,
                    sample_rate=24000,
                    play=True,
                    return_player=True,
                )
            )

            # Wait for playback to complete
            while self.player and self.player.is_playing():
                await asyncio.sleep(0.1)

            logger.info("Response playback complete")

            # Clear the cache
            await asyncio.get_event_loop().run_in_executor(None, mx.metal.clear_cache)

            # Mark task as done
            self.response_queue.task_done()

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
        finally:
            self.speaking = False

    async def _create_peer_connection(self):
        """Create and setup WebRTC peer connection asynchronously"""
        pc = RTCPeerConnection()

        # Handle ICE connection state change
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()

        # Setup audio track handling
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received {track.kind} track from remote peer")
            if track.kind == "audio":
                # Create a transform track that will process the audio
                transform = AudioTransformTrack(track, self)
                # We add a recorder to save the audio (optional)
                recorder = MediaRecorder(os.path.join(self.output_dir, "input.wav"))
                recorder.addTrack(transform)
                recorder.start()

        return pc

    async def start_webrtc_server(self):
        """Start WebRTC server with signaling"""
        # Initialize models before starting server
        await self.init_models()

        # Create signaling
        self.signaling = TcpSocketSignaling(self.signaling_server, port=self.signaling_port)
        logger.info(f"WebRTC server started on {self.signaling_server}")

        # Main WebRTC connection loop
        while True:
            try:
                obj = await self.signaling.receive()

                if isinstance(obj, RTCSessionDescription):
                    # Create a new peer connection for each offer
                    self.peer_connection = await self._create_peer_connection()

                    # Set the remote description
                    await self.peer_connection.setRemoteDescription(obj)

                    # Create an answer
                    if obj.type == "offer":
                        # Prepare media for sending
                        # TODO: Create a MediaPlayer for sending TTS output

                        # Create answer
                        answer = await self.peer_connection.createAnswer()
                        await self.peer_connection.setLocalDescription(answer)

                        # Send answer
                        await self.signaling.send(self.peer_connection.localDescription)

                elif obj is BYE:
                    logger.info("Received BYE from client")
                    if self.peer_connection:
                        await self.peer_connection.close()

            except Exception as e:
                logger.error(f"WebRTC server error: {e}")
                # Brief pause before continuing
                await asyncio.sleep(1)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Voice Pipeline")
    parser.add_argument(
        "--webrtc", action="store_true", help="Enable WebRTC functionality"
    )
    parser.add_argument(
        "--signaling", type=str, default="0.0.0.0:8080", help="Signaling server for WebRTC"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory for audio files"
    )
    parser.add_argument(
        "--stt_model", type=str, default="mlx-community/whisper-large-v3-turbo", help="STT model"
    )
    parser.add_argument(
        "--tts_model", type=str, default="mlx-community/Kokoro-82M-bf16", help="TTS model"
    )
    parser.add_argument(
        "--llm_model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="LLM model"
    )
    parser.add_argument(
        "--vad_mode", type=int, default=3, help="VAD mode"
    )
    parser.add_argument(
        "--silence_duration", type=float, default=3.0, help="Silence duration"
    )

    args = parser.parse_args()

    # Create voice pipeline
    pipeline = VoicePipeline(
        webrtc_enabled=args.webrtc,
        signaling_server=args.signaling if args.webrtc else None,
        output_dir=args.output_dir,
        stt_model=args.stt_model,
        tts_model=args.tts_model,
        llm_model=args.llm_model,
        vad_mode=args.vad_mode,
        silence_duration=args.silence_duration
    )

    if args.webrtc:
        # Start WebRTC server
        await pipeline.start_webrtc_server()
    else:
        # Start voice input processing
        await pipeline.listen()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopping application...")