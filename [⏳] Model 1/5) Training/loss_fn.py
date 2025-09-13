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
    
    def forward(self, logits, labels, attention_mask=None, mask_positions=None, top_k=5):
        """
        logits: [B, S, V]
        labels: [B, S]  (-100 where ignored)
        mask_positions: [B, S] bool indicating generation/mask positions
        """

        # Initialize mask if not done yet or if vocab size changed
        actual_vocab_size = logits.size(-1)
        if self.non_music_mask is None or self.model_vocab_size != actual_vocab_size:
            self._initialize_mask(actual_vocab_size, logits.device)

        # Ensure non_music_mask is on the same device & dtype
        non_music_mask = self.non_music_mask.to(logits.device)
        # non_music_mask is True where penalty applies (1) and False for allowed tokens (0)
        if non_music_mask.dtype != torch.bool:
            non_music_mask = non_music_mask != 0

        # ****************** ROBUST VALID POSITIONS MASK ******************
        if attention_mask is None:
            valid_positions = torch.ones_like(labels, dtype=torch.bool, device=logits.device)
        else:
            valid_positions = (attention_mask == 1)

        valid_positions = valid_positions & (labels != -100)

        if mask_positions is not None:
            mask_positions = mask_positions.to(valid_positions.device)
            if mask_positions.shape != valid_positions.shape:
                mask_positions = torch.nn.functional.pad(
                    mask_positions,
                    (0, valid_positions.shape[1] - mask_positions.shape[1]),
                    value=False
                )
            valid_positions = valid_positions & mask_positions

        # ----------------- HARD LOGITS MASK (disallow non-music tokens at valid positions) -----------------
        # allowed_mask: True for allowed (music + specials), False for disallowed
        allowed_mask = (~non_music_mask).to(torch.bool)  # shape: [V]

        # Create a copy of logits to avoid modifying original in-place
        logits_masked = logits.clone()

        # FIXED: Apply masking more aggressively
        B, S, V = logits.shape
        for b in range(B):
            for s in range(S):
                if valid_positions[b, s]:
                    # Set non-music tokens to extremely negative logits
                    logits_masked[b, s, ~allowed_mask] = -1e10  # More extreme than -1e9

        # ----------------- Cross-Entropy on masked logits (ignores -100) -----------------
        ce_loss = self.ce_loss(logits_masked.view(-1, logits_masked.size(-1)), labels.view(-1))

        # ----------------- TOP-K penalty logic (robust version) on ORIGINAL logits -----------------
        topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # Use ORIGINAL logits for penalty

        # topk_is_non_music: True where top-k token is non-music (disallowed)
        topk_is_non_music = non_music_mask[topk_indices]  # shape: [B, S, k]

        any_topk_non_music = topk_is_non_music.any(dim=-1)  # [B, S]

        # Probabilities among the k predictions (softmax across k)
        topk_probs = torch.softmax(topk_values, dim=-1)  # [B, S, k]

        non_music_mask_float = topk_is_non_music.float()
        non_music_probs = topk_probs * non_music_mask_float
        max_non_music_prob = non_music_probs.max(dim=-1)[0]  # [B, S]
        max_non_music_prob = torch.where(any_topk_non_music, max_non_music_prob, torch.ones_like(max_non_music_prob))
        max_non_music_prob = torch.clamp(max_non_music_prob, min=1e-12)

        penalty_positions = any_topk_non_music & valid_positions  # [B, S]
        penalty_per_pos = -torch.log(max_non_music_prob) * self.non_music_penalty
        penalty_per_pos = penalty_per_pos * penalty_positions.float()

        num_penalized = penalty_positions.sum()
        if num_penalized.item() > 0:
            non_music_penalty_loss = penalty_per_pos.sum() / num_penalized.float()
        else:
            non_music_penalty_loss = torch.tensor(0.0, device=logits.device)

        total_loss = ce_loss + non_music_penalty_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'non_music_penalty': non_music_penalty_loss.item(),
            'non_music_predictions': int(num_penalized.item())
        }
