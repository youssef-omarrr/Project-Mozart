import re

def is_musical_token(token):
    """Check if token is a musical token (enhanced version)"""
    # Full musical tokens pattern - more comprehensive
    full_pattern = r'^[A-G][#b♯♭]?[0-9]+_[a-z0-9/\.]+$|^Rest_[a-z0-9/\.]+$|^[A-G][#b♯♭]?[0-9]+$'
    
    # Subword components that are part of musical tokens
    note_parts = ['C', 'D', 'E', 'F', 'G', 'A', 'B', '#', 'b', '♯', '♭']
    duration_parts = ['_q', '_e', '_h', '_w', '_s', '_t',
                    '_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9',
                    '_0.25', '_0.5', '_0.75', '_1.0', '_1.5', '_2.0', '_3.0', '_4.0',
                    '_1/4', '_1/8', '_1/16', '_3/4', '_3/8', '_1/3', '_2/3', '_1/6']
    rest_parts = ['Rest', 'rest', 'REST', 'r']
    
    # Special tokens
    special_tokens = ['<|startofpiece|>', '<|endofpiece|>', '<TRACKS>', '<TRACKSEP>', 
                        '<NAME=', '<BPM=', '<DURATION_BEATS=', '<DURATION_MINUTES=', '<MASK>']
    
    # Octave numbers
    octave_parts = [str(i) for i in range(0, 10)]
    
    # Check if token matches any pattern
    if (re.match(full_pattern, token) or
        token in note_parts or
        token in duration_parts or
        token in rest_parts or
        token in special_tokens or
        token in octave_parts or
        any(token.startswith(prefix) for prefix in ['<NAME=', '<BPM=', '<DURATION_BEATS=', '<DURATION_MINUTES='])):
        return True
    
    # Additional check for instrument names
    if token.endswith(':') and len(token) > 1:
        return True
        
    return False

def add_musical_tokens_to_tokenizer(tokenizer):
    """Add common musical tokens to the tokenizer's vocabulary"""
    musical_tokens = []
    
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    durations = ['q', 'e', 'h', 'w', 's', '0.25', '0.5', '0.75', '1.0', '1.5', '2.0',
                '1/4', '1/8', '1/16', '3/4', '3/8', '1/3', '2/3', '1/6']
    
    # Notes with durations
    for note in notes:
        for octave in range(1, 8):
            for duration in durations:
                # Generate potential tokens
                tokens_to_check = [
                    f"{note}{octave}_{duration}",
                    f"{note}#{octave}_{duration}",
                    f"{note}b{octave}_{duration}"
                ]
                
                # Only add tokens that are recognized as musical
                for token in tokens_to_check:
                    if is_musical_token(token):
                        musical_tokens.append(token)
    
    # Rests with durations
    for duration in durations:
        rest_token = f"Rest_{duration}"
        if is_musical_token(rest_token):
            musical_tokens.append(rest_token)
    
    # Filter out any tokens that might already exist in the tokenizer
    existing_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in musical_tokens if token not in existing_vocab]
    
    # Add tokens to tokenizer
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"Added {len(new_tokens)} new musical tokens to vocabulary")
        print(f"Total musical tokens now: {len([t for t in tokenizer.get_vocab().keys() if is_musical_token(t)])}")
    else:
        print("No new musical tokens to add (they may already exist in vocabulary)")
    
    return tokenizer


class MusicalTokenizerWrapper:
    """Wrapper around GPT-2 tokenizer that filters non-musical tokens"""
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.musical_token_ids = self._identify_musical_tokens()
        
    def _identify_musical_tokens(self):
        """Identify which token IDs correspond to musical tokens"""
        musical_token_ids = []
        for token, token_id in self.base_tokenizer.get_vocab().items():
            if is_musical_token(token):
                musical_token_ids.append(token_id)
        print(f"Found {len(musical_token_ids)} musical tokens out of {len(self.base_tokenizer)} total")
        return musical_token_ids
    
    def __call__(self, *args, **kwargs):
        """Tokenize text but only keep musical tokens"""
        result = self.base_tokenizer(*args, **kwargs)
        
        # Create mask for musical tokens
        musical_mask = []
        for input_ids in result["input_ids"]:
            mask = [1 if token_id in self.musical_token_ids else 0 for token_id in input_ids]
            musical_mask.append(mask)
        
        result["musical_attention_mask"] = musical_mask
        return result
    
    def decode(self, *args, **kwargs):
        """Pass through to base tokenizer"""
        return self.base_tokenizer.decode(*args, **kwargs)
    
    def __len__(self):
        return len(self.base_tokenizer)
    
    # Pass through other necessary methods
    def __getattr__(self, name):
        return getattr(self.base_tokenizer, name)
    
    # Add properties to maintain compatibility
    @property
    def pad_token(self):
        return self.base_tokenizer.pad_token
    
    @property
    def eos_token(self):
        return self.base_tokenizer.eos_token
    
    @property
    def vocab_size(self):
        return self.base_tokenizer.vocab_size
    
    def get_vocab(self):
        return self.base_tokenizer.get_vocab()
    
    def save_pretrained(self, *args, **kwargs):
        return self.base_tokenizer.save_pretrained(*args, **kwargs)