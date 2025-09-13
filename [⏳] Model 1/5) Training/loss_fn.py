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
        
        # USE THE CANONICAL LIST OF SPECIAL TOKENS FROM THE TOKENIZER
        # This is more reliable than manually searching
        self.special_token_ids = tokenizer.all_special_ids
        
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
    
    def forward(self, logits, labels, attention_mask=None, mask_positions=None, top_k=5):        # Standard cross entropy loss - only compute where labels != -100
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Initialize mask if not done yet or if vocab size changed
        actual_vocab_size = logits.size(-1)
        if self.non_music_mask is None or self.model_vocab_size != actual_vocab_size:
            self._initialize_mask(actual_vocab_size, logits.device)
        
        # ****************** ROBUST VALID POSITIONS MASK ******************
        # This is the most accurate way to find positions that are real data, not padding.
        # 1. Start with the attention mask (1=real token, 0=padding)
        if attention_mask is None:
            # If no mask provided, assume all positions are valid
            valid_positions = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid_positions = (attention_mask == 1)

        # 2. IMPORTANT: Also ignore positions where labels are -100 (the ignore_index)
        # This is a standard practice in Hugging Face models.
        valid_positions = valid_positions & (labels != -100)
        
        # ****************** CRITICAL: ONLY AT MASK POSITIONS ******************
        if mask_positions is not None:
            # Ensure mask_positions is on the same device and has same shape
            mask_positions = mask_positions.to(valid_positions.device)
            if mask_positions.shape != valid_positions.shape:
                # Handle potential shape mismatch from padding
                mask_positions = torch.nn.functional.pad(
                    mask_positions, 
                    (0, valid_positions.shape[1] - mask_positions.shape[1]), 
                    value=False
                )
            valid_positions = valid_positions & mask_positions
            
        # ****************** TOP-K PENALTY LOGIC ******************
        # Get the top-k predicted token IDs and their values
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # shapes: [batch, seq, k]
        
        # Check if any of the top-k predictions are non-music tokens
        expanded_mask = self.non_music_mask.unsqueeze(0).unsqueeze(0)
        topk_is_non_music = expanded_mask[:, :, topk_indices]
        
        # Check if ANY of the top-k predictions are non-music: [batch, seq]
        any_topk_non_music = topk_is_non_music.any(dim=-1)
        
        # Penalize only on valid positions (real tokens, not ignored by labels)
        penalty_positions = (any_topk_non_music & valid_positions)

        # Calculate the penalty loss based on the highest non-music probability in top-k
        probs = torch.softmax(logits, dim=-1)
        
        # Get the maximum probability among non-music tokens in the top-k
        non_music_in_topk = topk_is_non_music.float()
        topk_probs = torch.softmax(topk_values, dim=-1)
        max_non_music_prob = (topk_probs * non_music_in_topk).max(dim=-1)[0]
        # Replace any zeros (where no non-music was in top-k) with a small value to avoid log(0)
        max_non_music_prob = torch.clamp(max_non_music_prob, min=1e-12)
        
        # Apply penalty
        non_music_penalty_loss = (-torch.log(max_non_music_prob) * penalty_positions * self.non_music_penalty).mean()

        total_loss = ce_loss + non_music_penalty_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'non_music_penalty': non_music_penalty_loss.item(),
            'non_music_predictions': penalty_positions.sum().item()
        }