import torch
import torch.nn as nn
import torchaudio
from asteroid.models import BaseModel


class TIGERSeparator(nn.Module):
    def __init__(self, num_speakers=2):
        super().__init__()
        print("[TIGER Engine] Deploying Parallel Convolutional Matrix (ConvTasNet)...")
        self.model = BaseModel.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")

        # ==========================================================
        # ACOUSTIC ALIGNMENT: 8kHz <-> 16kHz BRIDGE
        # ConvTasNet expects 8kHz. Vosk ASR expects 16kHz.
        # We bridge them dynamically in memory.
        # ==========================================================
        self.model_sr = getattr(self.model, 'sample_rate', 8000)
        self.pipeline_sr = 16000

        if self.model_sr != self.pipeline_sr:
            print(f"[TIGER Engine] Architecture uses {self.model_sr}Hz. Injecting dynamic resamplers...")
            self.downsample = torchaudio.transforms.Resample(self.pipeline_sr, self.model_sr)
            self.upsample = torchaudio.transforms.Resample(self.model_sr, self.pipeline_sr)
        else:
            self.downsample = nn.Identity()
            self.upsample = nn.Identity()

    def forward(self, mixture):
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)

        # 1. Downsample for Neural Matrix compatibility
        mixture = self.downsample(mixture)

        # 2. STANDARDIZATION
        std = torch.std(mixture, dim=-1, keepdim=True) + 1e-8
        mean = torch.mean(mixture, dim=-1, keepdim=True)
        mixture_normalized = (mixture - mean) / std

        # 3. Parallel Neural Separation
        with torch.inference_mode():
            separated_normalized = self.model(mixture_normalized)

        # 4. DENORMALIZATION
        separated_audio = (separated_normalized * std) + mean

        # 5. Upsample back to 16kHz for Kaldi ASR compatibility
        separated_audio = self.upsample(separated_audio)

        return [separated_audio[0, 0, :], separated_audio[0, 1, :]]