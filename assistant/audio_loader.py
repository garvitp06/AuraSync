import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset


class WindowsCPUAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000):
        self.data_dir = Path(data_dir).resolve()
        self.sample_rate = sample_rate
        self.audio_files = []

        # Recursively search through all nested subdirectories
        print(f"[AuraSync Data] Scanning directory tree: {self.data_dir}")
        for p in self.data_dir.rglob("*"):
            # Check if it is a file and not a directory container
            if p.is_file() and not p.name.startswith('.'):
                self.audio_files.append(p)

        print(f"[AuraSync Data] Total nested files located: {len(self.audio_files)}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]

        try:
            # Force torchaudio to load file path safely as a string
            waveform, sr = torchaudio.load(str(file_path))

            # Downsample/Upsample to target rate to optimize CPU memory usage
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            return waveform
        except Exception as e:
            # Fallback block if some non-audio files or system logs are in the folders
            # Return a 1-second silence chunk so the chunk engine doesn't crash
            return torch.zeros(1, self.sample_rate)


if __name__ == "__main__":
    # Robust relative locator logic
    current_dir = Path(__file__).parent.resolve()

    # Checks if 'data' folder is directly beside it or if it's running outside
    if (current_dir / "data").exists():
        KAG_PATH = current_dir / "data" / "wsj0"
    else:
        KAG_PATH = current_dir / "assistant" / "data" / "wsj0"

    dataset = WindowsCPUAudioDataset(data_dir=KAG_PATH)
    if len(dataset) > 0:
        sample_wave = dataset[0]
        print(f"[AuraSync Data] Success! Loaded sample audio tensor shape: {sample_wave.shape}")