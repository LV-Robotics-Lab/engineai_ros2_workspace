"""
Full audio capture and recording example.
Async microphone recording with PyAudio; saves a wav file when recording ends.
"""
import argparse
import queue
import numpy as np
import pyaudio
import wave
import time

class Recorder:
    def __init__(self, config):
        """Initialize audio recorder
        Args:
            config (dict): configuration dictionary
        """
        # Save configuration
        self.config = config
        # Thread-safe queue: buffers audio frames from the background capture thread
        self.recorder_queue = queue.Queue()
        # Recording device index; select a specific microphone
        self.recorder_device_index = config.get("recorder_device_index", None)
        # Sample rate, default 16000 Hz (common for speech recognition)
        self.recorder_rate = config.get("recorder_audio_rate", 16000)
        # Frames per buffer for each capture
        self.recorder_chunk_size = config.get("recorder_chunk_size", 1024)
        # Total hardware channels of the microphone
        self.recorder_channels = config.get("recorder_channels", 1)
        # Channel indices to extract
        self.recorder_pickup_channels = config["recorder_pickup_channels"]
        # Playback flag: set True while playing audio to mute mic and avoid feedback
        self.is_playing = False

        # Initialize PyAudio instance
        self.p = pyaudio.PyAudio()
        # Open microphone input stream with async callback
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.recorder_channels,
            rate=self.recorder_rate,
            input=True,
            input_device_index=self.recorder_device_index,
            frames_per_buffer=self.recorder_chunk_size,
            stream_callback=self.stream_callback,
        )

    def stream_callback(self, in_data, frame_count, time_info, status):
        """Background callback fired when the audio buffer is full (runs in a worker thread)."""
        # Skip recording while playing to avoid feedback / echo
        if self.is_playing:
            return (None, pyaudio.paContinue)

        try:
            # Convert raw bytes to int16 array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Reshape to (frames_per_buffer, total_channels)
            audio_reshaped = audio_data.reshape(
                self.recorder_chunk_size, self.recorder_channels
            )
            # Normalize: int16 [-32768, 32767] -> float [-1.0, 1.0]
            audio_float = audio_reshaped.astype(np.float32) / 32768.0
            # Extract configured channels
            audio_picked = audio_float[:, self.recorder_pickup_channels]
            # Non-blocking put; drop frame if queue is full to avoid blocking the sound card
            self.recorder_queue.put(audio_picked, block=False)
        except queue.Full:
            # Drop frame when queue is full; do not raise
            pass

        # Tell the sound card to continue capturing
        return (None, pyaudio.paContinue)

    def get(self):
        """Called from the main thread: get one float audio frame from the queue."""
        try:
            # Wait up to 0.5 s for audio
            data = self.recorder_queue.get(timeout=0.5)
            return data
        except queue.Empty:
            # Return silence zeros as fallback when no audio is available
            return np.zeros(self.recorder_chunk_size, dtype=np.float32)

    def clear_queue(self):
        """Clear all buffered audio frames from the queue."""
        while not self.recorder_queue.empty():
            try:
                self.recorder_queue.get_nowait()
            except queue.Empty:
                break

    def stop_recording(self):
        """Stop the audio capture stream."""
        self.stream.stop_stream()
        self.stream.close()

    def close(self):
        """Release all audio hardware resources."""
        self.stop_recording()
        self.p.terminate()

def save_wav(audio_frames, sample_rate, channels, save_path):
    """
    Save captured float audio arrays as a wav file.
    :param audio_frames: list of audio frames
    :param sample_rate: sample rate
    :param channels: number of channels to save
    :param save_path: output file path
    """
    # Concatenate all audio segments
    all_audio = np.concatenate(audio_frames, axis=0)
    # Convert float back to int16
    audio_int16 = (all_audio * 32768).astype(np.int16)

    # Create wav writer
    wf = wave.open(save_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit audio uses 2 bytes per sample
    wf.setframerate(sample_rate)
    # Write binary audio data
    wf.writeframes(audio_int16.tobytes())
    wf.close()
    print(f"✅ Recording saved successfully: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record from microphone and save as wav")
    parser.add_argument(
        "-o", "--output",
        default="./my_record.wav",
        help="Output wav file path (default: ./my_record.wav)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=5,
        help="Recording duration in seconds (default: 5)",
    )
    args = parser.parse_args()

    # ===================== Recording config =====================
    recorder_config = {
        "recorder_device_index": None,        # Mic device index; None uses the system default
        "recorder_audio_rate": 16000,          # Sample rate 16000 Hz
        "recorder_chunk_size": 1024,           # Frames per capture
        "recorder_channels": 1,                # Hardware channels: mono mic
        "recorder_pickup_channels": [0]        # Record channel 0
    }
    record_seconds = args.duration
    save_file_path = args.output

    # 1. Initialize recorder
    recorder = Recorder(recorder_config)
    print(f"🎙️ Starting recording for {record_seconds} seconds...")

    # Buffer all captured audio frames
    audio_buffer = []
    start_time = time.time()

    # 2. Capture loop
    while time.time() - start_time < record_seconds:
        frame_data = recorder.get()
        audio_buffer.append(frame_data)

    # 3. Stop recording and release hardware
    recorder.close()
    print("Recording finished, saving audio...")

    # 4. Save audio to a local wav file
    save_wav(
        audio_frames=audio_buffer,
        sample_rate=recorder_config["recorder_audio_rate"],
        channels=len(recorder_config["recorder_pickup_channels"]),
        save_path=save_file_path
    )
