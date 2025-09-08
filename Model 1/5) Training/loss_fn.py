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
        
        # Create mask for non-music tokens
        valid_music_ids = [i for i in music_token_ids if 0 <= i < len(tokenizer)]
        self.non_music_mask = torch.ones(len(tokenizer))
        self.non_music_mask[valid_music_ids] = 0  # Music tokens get 0 penalty

        self.non_music_mask[tokenizer.pad_token_id] = 0  # Ignore padding
        self.non_music_mask[tokenizer.eos_token_id] = 0  # Ignore EOS
        
        # Handle special tokens like <MASK>
        if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.non_music_mask[tokenizer.mask_token_id] = 0
        
        # Handle any other special tokens that start with '<'
        for token, token_id in tokenizer.get_vocab().items():
            if token.startswith('<') and token.endswith('>'):
                if 0 <= token_id < len(tokenizer):
                    self.non_music_mask[token_id] = 0
        
        self.non_music_mask = self.non_music_mask.to(DEVICE)
        
        # Identify rest token IDs
        self.rest_token_ids = []
        for token, token_id in tokenizer.get_vocab().items():
            if (token.lower().startswith('rest') or token == 'r' or token == 'R') and is_music_token(token):
                self.rest_token_ids.append(token_id)
        
        print(f"Found {len(self.rest_token_ids)} rest tokens in vocabulary")
    
    def forward(self, logits, labels, attention_mask=None):
        # Standard cross entropy loss - only compute where labels != -100
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Heavy penalty for non-music token predictions
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate probability mass assigned to non-music tokens
        non_music_probs = probs * self.non_music_mask.to(logits.device).unsqueeze(0).unsqueeze(0)
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