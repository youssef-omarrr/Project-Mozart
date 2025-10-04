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
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Create a column vector representing token positions [1, 2, .., max_len-1]
    # shape after unsqueezing: (1, max_len)

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
        
        # Embeddings
        # -----------
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=embed_dim)
        
        # Postional encoding (using buffer)
        # ----------------------------------
        self.register_buffer(
            name= "positional_encoding",
            tensor= sine_positional_encoding(max_seq_len, embed_dim)
        )
        