"""audio_capture/capture.py — rolling 30-second mic chunks via sounddevice."""
from __future__ import annotations
import threading
import numpy as np
import sounddevice as sd
from typing import Callable, Optional

SAMPLE_RATE    = 16_000
CHUNK_SECONDS  = 30
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SECONDS


class AudioCapture:
    """Continuously records microphone audio and fires a callback for each 30-second chunk.

    Callback signature: on_chunk(audio: np.ndarray)  — float32 mono, 16 kHz.
    """

    def __init__(
        self,
        on_chunk: Callable[[np.ndarray], None],
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = CHUNK_SECONDS,
        device: Optional[int] = None,
    ) -> None:
        self.on_chunk      = on_chunk
        self.sample_rate   = sample_rate
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.device        = device
        self._buf          = np.empty((0,), dtype=np.float32)
        self._lock         = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self.running       = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._cb,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> None:
        self.running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # flush tail
        with self._lock:
            tail = self._buf.copy()
            self._buf = np.empty((0,), dtype=np.float32)
        if len(tail) > 128:
            self.on_chunk(tail)

    def _cb(self, indata: np.ndarray, frames: int, _time, _status) -> None:
        mono = indata[:, 0]
        with self._lock:
            self._buf = np.concatenate([self._buf, mono])
            while len(self._buf) >= self.chunk_samples:
                chunk          = self._buf[: self.chunk_samples].copy()
                self._buf      = self._buf[self.chunk_samples :]
                threading.Thread(target=self.on_chunk, args=(chunk,), daemon=True).start()

    @staticmethod
    def list_devices() -> None:
        print(sd.query_devices())
