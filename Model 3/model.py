import torch
from torch import nn
import math

# Sinusoidal Positional Encoding
# -------------------------------
def sine_positional_encoding(max_len, embed_dim):
    """
    - Sinusoidal positional encoding adds position-dependent sine and cosine values to token embeddings 
    so transformers can capture word order without recurrence.
    
    - It uses sinusoidal functions with different frequencies to encode the position of each element in a sequence.
    
    - The sine and cosine functions are chosen because they have a cyclic nature, 
    which allows them to return information about the position of a token in a way that is easy for the model to learn
    
    - shape: (1, max_len, embed_dim)
    """
    
    pos_encoding = torch.zeros(max_len, embed_dim) # will later hold sine and cosine values for each position.
    
    # Create a column vector representing token positions [1, 2, .., max_len-1]
    positions_vector = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape after unsqueezing: (1, max_len)
    
    # The formula makes the sine and cosine waves vary in frequency across dimensions (low-frequency for smaller indices)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
    )
    
    # Fill even embedding indices with sin and odd ones with cos.
    pos_encoding[:, 0::2] = torch.sin(positions_vector * div_term)
    pos_encoding[:, 1::2] = torch.cos(positions_vector * div_term)
    
    return pos_encoding.unsqueeze(0) # (1, max_len, embed_dim)
    

# Model defination
# -----------------
class ProjectMozart(nn.Module):
    def __init__(self,
                vocab_size:int,
                embed_dim:int = 512, # the size of each token’s vector representation
                
                n_heads:int = 8, # number of independent attention mechanisms running in parallel
                n_layers:int = 8, # depth of the network
                
                feedforward_dim:int = 2048, # the hidden layer size of the MLP
                dropout:float = 0.1,
                max_seq_len:int = 2048):
        super().__init__()
        
        # Used in the forward function
        self.embed_dim = embed_dim
        
        # Embeddings
        # -----------
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=embed_dim)
        
        # Postional encoding tensor (using buffer)
        # ----------------------------------------
        self.register_buffer(
            name= "positional_encoding",
            tensor= sine_positional_encoding(max_seq_len, embed_dim),
            persistent= False # no need to save it (can be recomputed easily)
        )
        
        # Transformer
        # ------------
        # One encoder layer defination
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model= embed_dim,
            nhead=n_heads,
            dim_feedforward= feedforward_dim,
            dropout=dropout,
            batch_first=True # (batch, seq, feature)
        )
        # The whole encoder, which is n_layers number of encoder layers
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
        # LayerNorm and output
        # ---------------------
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size) # fully connected layer: maps each token’s hidden vector (embed_dim) to the output space (vocab_size)
        
    def _generate_causal_mask(self, seq_len):
        """
        Creates an upper-triangular mask to prevent attention to future positions.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1 # don't mask the diagonal
        ).bool()
        
        return mask
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) of token IDs (from our dataloader) -> embedded into (batch, seq_len, embed_dim)
        Returns:
            logits: (batch, seq_len, vocab_size) (where vocab_size dimension is the scores of each word in the vocab)
        """
        
        _batch, seq_len = x.size()
        device = x.device
        
        # Embedding + position
        # ---------------------
        # For training stability multiply by embed_scale = (embed_dim)^(1/2)
        x = self.embed_layer(x) * (self.embed_dim ** 0.5)
        
        # Select only the first seq_len positions from the stored positional encoding
        pos_enc = self.positional_encoding[:, :seq_len, :].to(device)
        
        x = x + pos_enc
        
        # Causal mask
        # ------------
        mask = self._generate_causal_mask(seq_len)
        
        # Transformer
        # -----------
        out = self.transformer(x, mask=mask)
        
        # LayerNorm and projection
        # -------------------------
        out = self.norm(out)
        logits = self.fc(out)
        
        return logits