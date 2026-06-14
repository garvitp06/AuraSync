import torch
import torch.nn as nn


class VoiceIDEmbedder(nn.Module):
    def __init__(self, embedding_dim=64):
        super(VoiceIDEmbedder, self).__init__()

        # Lightweight convolutional feature extractor for speaker characteristics
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # Standardizes variable lengths to a fixed width
        )

        # Projection header to create the stable speaker identity vector
        self.fc_embedding = nn.Linear(16 * 32, embedding_dim)
        print(f"[VoiceID] Model initialized. Profile Dimension: {embedding_dim} (Footprint < 0.04M params).")

    def forward(self, audio_channel):
        """
        Extracts a distinct speaker verification profile embedding from an isolated channel.
        Input format: [Batch, Samples] or [Samples]
        Output format: [Batch, Embedding_Dim]
        """
        if audio_channel.dim() == 1:
            audio_channel = audio_channel.unsqueeze(0)

        # Add channel dimension
        x = audio_channel.unsqueeze(1)

        features = self.feature_extractor(x)
        flattened = features.view(features.size(0), -1)
        speaker_embedding = self.fc_embedding(flattened)

        # Normalize the embedding vector onto a unit sphere for clean distance tracking
        return torch.nn.functional.normalize(speaker_embedding, p=2, dim=1)


class SpeakerConditionedFiLM(nn.Module):
    def __init__(self, embedding_dim=64, audio_features=2560):
        super(SpeakerConditionedFiLM, self).__init__()
        # Generates Gamma (scaling) and Beta (shifting) vectors from the Speaker Profile
        self.gamma_projector = nn.Linear(embedding_dim, audio_features)
        self.beta_projector = nn.Linear(embedding_dim, audio_features)
        print("[FiLM Conditioning] Layer ready for target-speaker scaling adaptation.")

    def forward(self, audio_chunk, speaker_embedding):
        """
        Applies Feature-wise Linear Modulation to scale and shift the target audio features
        based on the verified speaker identity mapping profile.
        Formula: FiLM(x) = Gamma(embedding) * x + Beta(embedding)
        """
        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)

        gamma = self.gamma_projector(speaker_embedding)
        beta = self.beta_projector(speaker_embedding)

        # Modulate the waveform features directly
        modulated_audio = (gamma * audio_chunk) + beta
        return modulated_audio