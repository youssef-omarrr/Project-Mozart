############################################################
# IMPORTS
############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from filter import is_musical_token

############################################################
# CONSTANTS / CONFIG
############################################################
# Custom loss weights - only penalize consecutive rests
CONSECUTIVE_REST_PENALTY = 5.0  # Heavy penalty for consecutive rests
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 10         # Minimum notes per sequence


############################################################
# Custom Loss Functions - Penalize Consecutive Rests Only
############################################################
class ConsecutiveRestLoss(nn.Module):
    def __init__(self, tokenizer, consecutive_penalty=CONSECUTIVE_REST_PENALTY,
                                    density_reward=NOTE_DENSITY_REWARD):
        super().__init__()
        self.tokenizer = tokenizer
        self.consecutive_penalty = consecutive_penalty
        self.density_reward = density_reward
        
        
        # Identify rest token IDs from the musical tokens only
        self.rest_token_ids = []
        for token, token_id in tokenizer.base_tokenizer.get_vocab().items():
            if ('Rest' in token or 'rest' in token) and is_musical_token(token):
                self.rest_token_ids.append(token_id)
        
        print(f"Found {len(self.rest_token_ids)} rest tokens in musical vocabulary")


    def detect_consecutive_rests(self, sequence):
        """Detect positions with consecutive rest notes"""
        if len(self.rest_token_ids) == 0:
            return torch.zeros_like(sequence, dtype=torch.bool)
        
        # Create mask for rest tokens
        is_rest = torch.isin(sequence, torch.tensor(self.rest_token_ids).to(sequence.device))
        
        # Find consecutive rests (current token is rest AND previous token is rest)
        consecutive_rests = torch.zeros_like(is_rest, dtype=torch.bool)
        consecutive_rests[:, 1:] = is_rest[:, 1:] & is_rest[:, :-1]
        
        return consecutive_rests


    def forward(self, logits, labels, 
                attention_mask=None, 
                musical_attention_mask=None):
        
        # Use musical attention mask if provided
        if musical_attention_mask is not None:
            effective_mask = musical_attention_mask
        else:
            effective_mask = attention_mask
            
        # Standard cross entropy loss (use the model's built-in loss calculation)
        # We'll compute this properly by shifting
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        
        if effective_mask is not None:
            shift_mask = effective_mask[..., 1:].contiguous()  # Shift the mask too
            
            # Calculate loss without reduction first
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                reduction='none'
            )
            
            # Reshape and apply mask - FIXED THE SHAPE ISSUE
            ce_loss = ce_loss.view(shift_labels.size())
            valid_positions = shift_mask.view(-1) > 0
            ce_loss = ce_loss.view(-1)[valid_positions].mean()
            
        else:
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                reduction='mean'
            )
        
        
        # --- Penalties ---
        # Penalize invalid tokens in predictions
        invalid_token_penalty = 0.0
        if self.invalid_token_ids:
            with torch.no_grad():
                # Get predicted tokens (greedy)
                pred_tokens = torch.argmax(logits, dim=-1)
                
                # Create mask for invalid tokens
                is_invalid = torch.isin(pred_tokens, torch.tensor(self.invalid_token_ids).to(pred_tokens.device))
                
                if is_invalid.any():
                    # Get probabilities for invalid tokens
                    probs = F.softmax(logits, dim=-1)
                    invalid_probs = probs[..., self.invalid_token_ids].sum(dim=-1)
                    
                    # Apply penalty to invalid token positions
                    invalid_token_penalty = (invalid_probs * is_invalid.float()).sum() * 5.0  # Heavy penalty
        
        # Penalize only consecutive rests in predictions
        # Initialize loss components as tensors on the same device
        consecutive_penalty_loss = torch.tensor(0.0, device=logits.device)
        density_reward = torch.tensor(0.0, device=logits.device)
        consecutive_rests_detected = 0
        
        if len(self.rest_token_ids) > 0:
            with torch.no_grad():
                # Get predicted tokens (greedy)
                pred_tokens = torch.argmax(logits, dim=-1)
                
                # Detect consecutive rests in predictions
                consecutive_rests_mask = self.detect_consecutive_rests(pred_tokens)
                consecutive_rests_detected = consecutive_rests_mask.sum().item()
                
                # Apply penalty only to positions with consecutive rests
                if consecutive_rests_mask.any():
                    # Simple penalty based on count of consecutive rests
                    # This avoids the complex indexing that was causing errors
                    consecutive_penalty_loss = consecutive_rests_mask.sum().float() * self.consecutive_penalty * 0.001
                
                # Reward note density (encourage more non-rest notes)
                if attention_mask is not None:
                    # Count non-rest tokens in labels (ignore padding and -100)
                    non_rest_mask = ~torch.isin(labels, torch.tensor(self.rest_token_ids).to(labels.device))
                    valid_mask = (labels != self.tokenizer.pad_token_id) & (labels != -100)
                    note_count = (non_rest_mask & valid_mask).sum(dim=-1).float()
                    
                    # Reward sequences with more notes
                    density_reward = -torch.relu(MIN_NOTES_THRESHOLD - note_count).mean() * self.density_reward
        
        total_loss = ce_loss + consecutive_penalty_loss + density_reward + invalid_token_penalty
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'consecutive_rest_penalty': consecutive_penalty_loss.item(),
            'density_reward': density_reward.item(),
            'invalid_token_penalty': invalid_token_penalty.item(),
            'consecutive_rests_detected': consecutive_rests_detected
        }