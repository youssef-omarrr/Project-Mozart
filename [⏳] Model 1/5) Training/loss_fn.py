############################################################
# IMPORTS
############################################################
import torch
import torch.nn as nn

from filter import is_music_token

############################################################
# CONSTANTS / CONFIG
############################################################
# Custom loss weights
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 10         # Minimum notes per sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################################
# Custom Loss Functions
############################################################
class MusicTokenEnforcementLoss(nn.Module):
    def __init__(self, tokenizer, music_token_ids, non_music_penalty=100.0):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.music_token_ids = music_token_ids
        self.non_music_penalty = non_music_penalty
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # These will be initialized during first forward pass
        self.model_vocab_size = None
        self.non_music_mask = None
        self.valid_music_ids = [i for i in music_token_ids if i >= 0]
        
        print(f"Initialized MusicTokenEnforcementLoss with {len(self.valid_music_ids)} valid music token IDs")
        
        # Store special token IDs for later use
        self.special_token_ids = []
        
        # Collect special token IDs
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            self.special_token_ids.append(tokenizer.pad_token_id)
            
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            self.special_token_ids.append(tokenizer.eos_token_id)
            
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.special_token_ids.append(tokenizer.bos_token_id)
            
        if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.special_token_ids.append(tokenizer.mask_token_id)
        
        # Handle any other special tokens that start with '<' and end with '>'
        try:
            vocab = tokenizer.get_vocab()
            for token, token_id in vocab.items():
                if token.startswith('<') and token.endswith('>'):
                    if token_id not in self.special_token_ids:
                        self.special_token_ids.append(token_id)
        except Exception as e:
            print(f"Warning: Could not process special tokens: {e}")
        
        # Identify rest token IDs
        self.rest_token_ids = []
        try:
            vocab = tokenizer.get_vocab()
            for token, token_id in vocab.items():
                if (token.lower().startswith('rest') or token == 'r' or token == 'R') and is_music_token(token):
                    self.rest_token_ids.append(token_id)
        except Exception as e:
            print(f"Warning: Could not process rest tokens: {e}")
        
        print(f"Found {len(self.rest_token_ids)} rest tokens and {len(self.special_token_ids)} special tokens")
    
    def _initialize_mask(self, actual_vocab_size, device):
        """Initialize the non-music mask based on actual vocabulary size"""
        print(f"Initializing mask with vocab size: {actual_vocab_size}")
        
        # Create mask for non-music tokens using actual model vocab size
        self.non_music_mask = torch.ones(actual_vocab_size, device=device)
        
        # Set music tokens to 0 (no penalty)
        valid_music_ids = [i for i in self.valid_music_ids if 0 <= i < actual_vocab_size]
        if valid_music_ids:
            self.non_music_mask[valid_music_ids] = 0
            print(f"Set {len(valid_music_ids)} music tokens to no penalty")
        
        # Set special tokens to 0 (no penalty)
        valid_special_ids = [i for i in self.special_token_ids if 0 <= i < actual_vocab_size]
        if valid_special_ids:
            self.non_music_mask[valid_special_ids] = 0
            print(f"Set {len(valid_special_ids)} special tokens to no penalty")
        
        self.model_vocab_size = actual_vocab_size
    
    def forward(self, logits, labels, attention_mask=None):
        # Standard cross entropy loss - only compute where labels != -100
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Initialize mask if not done yet or if vocab size changed
        actual_vocab_size = logits.size(-1)
        if self.non_music_mask is None or self.model_vocab_size != actual_vocab_size:
            self._initialize_mask(actual_vocab_size, logits.device)
        
        # Heavy penalty for non-music token predictions
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate probability mass assigned to non-music tokens
        non_music_probs = probs * self.non_music_mask.unsqueeze(0).unsqueeze(0)
        non_music_mass = non_music_probs.sum(dim=-1)
        
        # Only apply penalty where we have valid labels (not -100)
        valid_positions = (labels != -100).float()
        
        # Apply penalty only where model actually predicts non-music tokens
        penalty_mask = (non_music_mass > 0.01).float() * valid_positions
        non_music_penalty_loss = (non_music_mass * penalty_mask * self.non_music_penalty).mean()
        
        total_loss = ce_loss + non_music_penalty_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'non_music_penalty': non_music_penalty_loss.item(),
            'non_music_predictions': (penalty_mask > 0).sum().item()
        }