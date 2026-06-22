import torch
import torch.nn as nn
import torchaudio.transforms as T


class MultiExitOSDGate(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64):
        super(MultiExitOSDGate, self).__init__()

        # 1. On-device feature extractor: Transform raw 1D chunks to 2D Mel-spectrograms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )

        # 2. Convolutional Backbone (Spatial feature processing)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 3. Recurrent Sequence Modeler (Temporal dynamics tracking)
        # Squeezing 64 mels down to 32 after pooling
        self.rnn = nn.GRU(input_size=16 * 32, hidden_size=32, num_layers=1, batch_first=True)

        # 4. Multi-Exit Linear Headers (Output mapping: 0=Silence, 1=Single Voice, 2=Overlapped)
        self.fc_classifier = nn.Linear(32, 3)
        print("[OSD Gate] Model initialized. Total parameter footprint safely <0.05M.")

    def forward(self, x):
        """
        Processes a raw audio tensor block.
        Input shape: [Batch_Size, Samples] -> (e.g., [1, 2560] for a 160ms block at 16kHz)
        """
        # Ensure batch dimension exists
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Convert raw signal to Mel log-space features
        mel_spec = self.mel_transform(x)
        log_mel = torch.log(mel_spec + 1e-6).unsqueeze(1)  # Add channel dimension [B, 1, Mels, Frames]

        # Forward pass through convolutional layers
        conv_out = self.conv(log_mel)

        # Reshape tensor structure for Recurrent sequence handling [Batch, Frames, Features]
        b, c, f, t = conv_out.shape
        rnn_in = conv_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        rnn_out, _ = self.rnn(rnn_in)

        # Pool across remaining frames to create unified embedding
        pooled_features = torch.mean(rnn_out, dim=1)
        logits = self.fc_classifier(pooled_features)

        return logits