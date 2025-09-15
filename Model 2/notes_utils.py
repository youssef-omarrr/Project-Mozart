# ------------------------------------------------------------
# NOTE MAPPINGS
# ------------------------------------------------------------

# note name -> semitone offset (C4 = 60 in MIDI)
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}


# ------------------------------------------------------------
# VOCAB HELPERS
# ------------------------------------------------------------

def build_vocab_maps(tokenizer):
    # tokenizer.vocab is a dict token->id
    vocab = tokenizer.vocab
    id_to_token = {int(v): k for k, v in vocab.items()}
    return vocab, id_to_token


def find_token_key_starting_with(vocab, prefix):
    # return token key matching prefix exactly or starting with prefix
    if prefix in vocab:
        return prefix
    for k in vocab:  # e.g. Duration_1.0.8
        if k.startswith(prefix):
            return k
    return None


def choose_velocity_token(vocab, target_vel=90):
    # collect velocity tokens and pick closest to target
    vel_tokens = []
    for k in vocab:
        if k.startswith("Velocity_"):
            try:
                n = int(k.split("_", 1)[1])
                vel_tokens.append((n, k))
            except Exception:
                pass
    if not vel_tokens:
        return None
    vel_tokens.sort(key=lambda x: abs(x[0] - target_vel))
    return vel_tokens[0][1]


# ------------------------------------------------------------
# NOTE PARSING
# ------------------------------------------------------------

def note_name_to_midi(note):
    """
    Convert note name (e.g. C4, C#4, Db4) -> MIDI number
    """
    import re
    m = re.match(r'^([A-G][#b]?)(-?\d+)$', note)
    if not m:
        raise ValueError(f"Invalid note name: {note}")
    name, octave = m.group(1), int(m.group(2))
    return 12 * (octave + 1) + NOTE_TO_SEMITONE[name]


def note_token_ids_from_symbolic(tokenizer, note_symbol, default_velocity=90):
    """
    Convert a symbolic note (e.g. 'C4_q') -> [Pitch, Velocity, Duration] token IDs
    Duration codes supported: w,h,q,e,s (whole, half, quarter, eighth, sixteenth)
    """
    vocab, _ = build_vocab_maps(tokenizer)

    if "_" not in note_symbol:
        raise ValueError("Symbolic note must be like 'C4_q'")

    note_part, dur_code = note_symbol.split("_", 1)

    # --- pitch ---
    midi = note_name_to_midi(note_part)   # 60 for C4
    pitch_key = f"Pitch_{midi}"
    pitch_token = find_token_key_starting_with(vocab, pitch_key)
    if pitch_token is None:
        raise RuntimeError(f"Pitch token for MIDI {midi} not found in vocab")

    # --- duration ---
    dur_map = {
        'w': '4.0',   # whole
        'h': '2.0',   # half
        'q': '1.0',   # quarter
        'e': '0.5',   # eighth
        's': '0.25',  # sixteenth
    }
    dur_value = dur_map.get(dur_code, dur_code)  # fallback: pass through

    duration_prefix = f"Duration_{dur_value}"
    duration_token = find_token_key_starting_with(vocab, duration_prefix)
    if duration_token is None:
        duration_token = find_token_key_starting_with(vocab, "Duration_1.0")
        if duration_token is None:
            raise RuntimeError("No suitable Duration token found in vocab")

    # --- velocity ---
    velocity_token = choose_velocity_token(vocab, target_vel=default_velocity)
    if velocity_token is None:
        velocity_token = find_token_key_starting_with(vocab, "Velocity_95")
        if velocity_token is None:
            raise RuntimeError("No Velocity token found in vocab")

    # return token IDs
    return [int(vocab[pitch_token]), int(vocab[velocity_token]), int(vocab[duration_token])]


# ------------------------------------------------------------
# RANDOM NOTE GENERATION
# ------------------------------------------------------------

import random

DUR_CODES = ["w", "h", "q", "e", "s"]  # whole, half, quarter, eighth, sixteenth

def random_note_symbol(octave_range=(3, 5)):
    """
    Generate a random symbolic note like 'C4_q' or 'F#5_e'
    """
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    note = random.choice(note_names)
    octave = random.randint(octave_range[0], octave_range[1])
    dur = random.choice(DUR_CODES)
    return f"{note}{octave}_{dur}"


def random_start_symbols(n=3, octave_range=(3, 5)):
    """
    Generate N random symbolic notes
    """
    return [random_note_symbol(octave_range) for _ in range(n)]


# ------------------------------------------------------------
# TOKEN TYPE AND SEQUENCE DETECTION
# ------------------------------------------------------------

def token_type(token_name):
    """
    Identify the type of token (Pitch, Velocity, Duration, Bar, Position)
    """
    for t in ["Pitch", "Velocity", "Duration", "Bar", "Position"]:
        if token_name.startswith(t):
            return t
    return None

def validate_token_sequence(tokenizer, token_ids, max_check=100):
    """
    Validate that a token sequence follows proper musical structure:
    Bar -> Position -> Pitch -> Velocity -> Duration -> (Position or Bar) -> ...
    """
    vocab, id_to_token = build_vocab_maps(tokenizer)
    
    check_ids = token_ids[:min(len(token_ids), max_check)]
    valid_transitions = 0
    invalid_transitions = 0
    
    # State machine for musical structure
    expected_states = {
        "Bar": ["Position"],
        "Position": ["Pitch"],
        "Pitch": ["Velocity"], 
        "Velocity": ["Duration"],
        "Duration": ["Position", "Bar"]
    }
    
    current_state = None
    
    for i, token_id in enumerate(check_ids):
        token_name = id_to_token.get(token_id, f"Unknown_{token_id}")
        
        if token_name.startswith(("EOS", "PAD")):
            continue
            
        token_type_val = token_type(token_name)  # Use existing function
        
        if token_type_val:
            if current_state is None:
                current_state = token_type_val
                valid_transitions += 1
            elif token_type_val in expected_states.get(current_state, []):
                valid_transitions += 1
                current_state = token_type_val
            else:
                invalid_transitions += 1
                expected = expected_states.get(current_state, [])
                print(f"Invalid transition at {i}: {current_state} -> {token_type_val} (expected: {expected})")
                current_state = token_type_val
    
    total = valid_transitions + invalid_transitions
    accuracy = valid_transitions / total if total > 0 else 0
    
    return {
        "valid_transitions": valid_transitions,
        "invalid_transitions": invalid_transitions,
        "accuracy": accuracy,
        "total_checked": len(check_ids)
    }