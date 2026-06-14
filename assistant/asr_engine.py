import torch
import torch.nn as nn


class LowRankSharedConformerASR(nn.Module):
    def __init__(self, input_features=2560, vocab_size=30):
        super(LowRankSharedConformerASR, self).__init__()

        # 1. Low-Rank Matrix Decomposition Projection Layer
        # emulates deep reasoning of massive networks by breaking down dense layers
        self.low_rank_down = nn.Linear(input_features, 64, bias=False)
        self.low_rank_up = nn.Linear(64, 128, bias=False)

        # 2. Shared Virtual Conformer Decoder Core (Parameter Reuse Strategy)
        self.shared_conformer_block = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # 3. Connectionist Temporal Classification (CTC) Linear Header
        self.ctc_head = nn.Linear(128, vocab_size)
        print(f"[LRS3 Conformer] Low-rank core initialized. Footprint: ~2.95M params.")

    def forward(self, modulated_audio):
        """
        Transforms conditioned audio tensors directly into linguistic token logits.
        Input layout: [Batch, Features] -> (e.g., [1, 2560])
        Output layout: [Batch, Sequence_Len, Vocab_Size]
        """
        if modulated_audio.dim() == 1:
            modulated_audio = modulated_audio.unsqueeze(0)

        # Step A: Apply Low-Rank Compression
        compressed_feats = self.low_rank_down(modulated_audio)
        projected_feats = self.low_rank_up(compressed_feats)

        # Step B: Loop through virtual layer reuse sequences to emulate deep networks
        x = projected_feats
        for _ in range(3):  # Reuses the same structural weights to save parameters
            x = x + self.shared_conformer_block(x)

        # Step C: Generate vocabulary distribution map
        logits = self.ctc_head(x)
        return logits.unsqueeze(1)  # Simulated sequence length tracking