import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import torchaudio
import numpy as np  # CRITICAL FIX: Added numpy import


class MultiDatasetAudioEngine(Dataset):
    def __init__(self, base_dir: Path, sample_rate: int = 16000):
        self.base_dir = Path(base_dir)
        self.sample_rate = sample_rate
        self.audio_files = []
        self.dataset_metadata = []

        print(f"\n[AuraSync Data] Initializing Multi-Dataset Evaluation Hub...")
        self._discover_datasets()

    def _discover_datasets(self):
        """
        Dynamically scans the data directory for all valid .wav files
        and automatically pairs them with their .txt transcript if it exists.
        """
        data_path = self.base_dir / "data"

        # Fallback to current directory if running flat
        if not data_path.exists():
            data_path = self.base_dir

        print(f"[AuraSync Data] Scanning directory: {data_path}")

        # Scan for ANY .wav file anywhere in the data folder or subfolders
        found_wavs = sorted(list(data_path.glob("**/*.wav")))

        for wav_path in found_wavs:
            self.audio_files.append(wav_path)

            # Automatically check if a matching .txt transcript exists
            txt_path = wav_path.with_suffix(".txt")
            transcript = ""
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()

            # Infer dataset origin based on the folder path
            origin = "unknown"
            if "wsj0" in str(wav_path).lower():
                origin = "wsj0"
            elif "svarah" in str(wav_path).lower():
                origin = "svarah"
            else:
                origin = "custom_upload"

            self.dataset_metadata.append({
                "origin": origin,
                "transcript": transcript,
                "file_name": wav_path.name
            })

        print(f"[AuraSync Data] -> Located {len(self.audio_files)} valid audio evaluation files.")

    def __len__(self):
        return len(self.audio_files)

    def apply_echoset_rir(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulates room reverberation (Optional Stress Test).
        """
        # Kept deterministic for consistent benchmarking
        torch.manual_seed(42)
        mock_rir = torch.randn(1, 1000) * 0.05
        reverb_waveform = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            mock_rir.unsqueeze(0),
            padding=500
        ).squeeze(0)
        return reverb_waveform[:, :waveform.shape[1]]

    def __getitem__(self, idx):
        if len(self.audio_files) == 0:
            print("[Warning] No audio files found. Falling back to generated noise.")
            return {
                "waveform": torch.randn(1, 16000 * 5),
                "origin": "fallback",
                "transcript": "No transcript available.",
                "file_name": "noise.wav"
            }

        file_path = self.audio_files[idx]
        meta = self.dataset_metadata[idx]

        # Use Scipy to handle raw files safely
        try:
            sr, audio_np = wavfile.read(str(file_path))
        except Exception as e:
            print(f"[Error] Could not read {file_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Fallback to next file

        # Convert to standard normalized tensor
        if audio_np.dtype == np.int16:
            audio_np = audio_np / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np / 2147483648.0

        waveform = torch.tensor(audio_np, dtype=torch.float32)

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return {
            "waveform": waveform,
            "origin": meta["origin"],
            "transcript": meta["transcript"],
            "file_name": meta["file_name"]
        }