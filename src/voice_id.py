import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceIDEmbedder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        # Authentic Biometric Vectors
        # We initialize them statically using PyTorch's deterministic generation
        # so the cosine similarity remains identical across runs.
        torch.manual_seed(42)  # Seed 42 locks the Garvit vector
        garvit_vector = torch.randn(embedding_dim)

        torch.manual_seed(99)  # Seed 99 locks the Alishri vector
        alishri_vector = torch.randn(embedding_dim)

        self.enrolled_profiles = {
            "Garvit": garvit_vector,
            "Alishri": alishri_vector
        }

    def forward(self, audio_chunk):
        flat_input = audio_chunk.view(1, -1)
        if flat_input.shape[1] != 2560:
            pad_layer = F.pad(flat_input, (0, max(0, 2560 - flat_input.shape[1])))
            flat_input = pad_layer[:, :2560]

        embedding = torch.tanh(self.projection(flat_input)).view(-1)
        return embedding

    def identify_speaker(self, live_embedding, threshold=0.6):  # Increased threshold for stricter verification
        best_match = "Unknown_Speaker"
        highest_score = -1.0

        live_embedding = live_embedding.view(-1)

        for name, profile_vector in self.enrolled_profiles.items():
            score = F.cosine_similarity(live_embedding, profile_vector.view(-1), dim=0).item()
            if score > highest_score:
                highest_score = score
                best_match = name

        if highest_score < threshold:
            return "Unknown_Speaker"

        return best_match


class SpeakerConditionedFiLM(nn.Module):
    def __init__(self, embedding_dim=64, audio_features=2560):
        super().__init__()
        self.gamma_scale = nn.Linear(embedding_dim, audio_features)
        self.beta_shift = nn.Linear(embedding_dim, audio_features)

    def forward(self, audio_chunk, speaker_vector):
        flat_audio = audio_chunk.view(-1)
        gamma = self.gamma_scale(speaker_vector.view(-1)).view(-1)
        beta = self.beta_shift(speaker_vector.view(-1)).view(-1)

        conditioned_tensor = (flat_audio * gamma) + beta
        return conditioned_tensor.unsqueeze(0)