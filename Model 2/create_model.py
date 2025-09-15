import torch
from torch import nn


# -------------------------------------------------------
# MusicTransformer Model
# -------------------------------------------------------
class MusicTransformer(nn.Module):
    """
    A simple Transformer model for music token prediction.
    Takes a sequence of token IDs (from MIDI tokenization)
    and predicts the next token in the sequence.
    """

    def __init__(self, 
                vocab_size: int,     # number of unique tokens (notes, events, etc.)
                embed_dim: int = 256, 
                n_heads: int = 4, 
                n_layers: int = 4,
                ff_dim: int = 512,   # feedforward dimension inside Transformer
                dropout: float = 0.1,
                max_seq_len: int = 1024):
        """
        Args:
            vocab_size: total number of tokens in vocabulary
            embed_dim: embedding dimension
            n_heads: number of attention heads
            n_layers: number of transformer encoder layers
            ff_dim: feedforward layer hidden size
            dropout: dropout rate
            max_seq_len: maximum sequence length (for positional encoding)
        """
        super().__init__()

        # ---- Embeddings ----
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Learnable positional encodings (not sinusoidal)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True   # keeps shape as (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- Output Projection ----
        self.fc = nn.Linear(embed_dim, vocab_size)

    # -------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) of token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embed(x)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Pass through Transformer
        out = self.transformer(x)  # (batch, seq_len, embed_dim)

        # Project to vocabulary size
        return self.fc(out)  # (batch, seq_len, vocab_size)
