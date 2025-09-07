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
        
        # Identify rest token IDs
        self.rest_token_ids = []
        self.rest_token_tensor = None  # Will be created on the proper device later
        
        # Populate rest token IDs
        for token, token_id in tokenizer.base_tokenizer.get_vocab().items():
            if any(rest_word in token for rest_word in ['Rest', 'rest', 'REST']) and is_musical_token(token):
                self.rest_token_ids.append(token_id)
        
        print(f"Found {len(self.rest_token_ids)} rest tokens in musical vocabulary")

    def _get_rest_token_tensor(self, device):
        """Get rest token tensor on the correct device"""
        if self.rest_token_tensor is None or self.rest_token_tensor.device != device:
            self.rest_token_tensor = torch.tensor(self.rest_token_ids, device=device)
        return self.rest_token_tensor

    def detect_consecutive_rests(self, sequence):
        """Detect positions with consecutive rest notes"""
        if len(self.rest_token_ids) == 0:
            return torch.zeros_like(sequence, dtype=torch.bool)
        
        # Create mask for rest tokens
        rest_tensor = self._get_rest_token_tensor(sequence.device)
        is_rest = torch.isin(sequence, rest_tensor)
        
        # Find consecutive rests
        consecutive_rests = torch.zeros_like(is_rest, dtype=torch.bool)
        consecutive_rests[:, 1:] = is_rest[:, 1:] & is_rest[:, :-1]
        
        return consecutive_rests

    def forward(self, logits, labels, attention_mask=None, musical_attention_mask=None):
        # Use musical attention mask if provided
        effective_mask = musical_attention_mask if musical_attention_mask is not None else attention_mask
        
        # Standard cross entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        if effective_mask is not None:
            shift_mask = effective_mask[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                reduction='none'
            )
            
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
        consecutive_penalty_loss = torch.tensor(0.0, device=logits.device)
        density_reward = torch.tensor(0.0, device=logits.device)
        consecutive_rests_detected = 0
        
        if self.rest_token_ids:
            with torch.no_grad():
                # Get predicted tokens
                pred_tokens = torch.argmax(logits, dim=-1)
                
                # Detect consecutive rests
                consecutive_rests_mask = self.detect_consecutive_rests(pred_tokens)
                consecutive_rests_detected = consecutive_rests_mask.sum().item()
                
                if consecutive_rests_mask.any():
                    consecutive_penalty_loss = consecutive_rests_mask.sum().float() * self.consecutive_penalty * 0.001
                
                # Reward note density
                if attention_mask is not None:
                    rest_tensor = self._get_rest_token_tensor(labels.device)
                    non_rest_mask = ~torch.isin(labels, rest_tensor)
                    valid_mask = (labels != self.tokenizer.pad_token_id) & (labels != -100)
                    note_count = (non_rest_mask & valid_mask).sum(dim=-1).float()
                    
                    density_reward = -torch.relu(MIN_NOTES_THRESHOLD - note_count).mean() * self.density_reward
        
        total_loss = ce_loss + consecutive_penalty_loss + density_reward
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'consecutive_rest_penalty': consecutive_penalty_loss.item(),
            'density_reward': density_reward.item(),
            'consecutive_rests_detected': consecutive_rests_detected
        }