from miditok import REMI, TokenizerConfig
from pathlib import Path
from tqdm import tqdm

def get_tokenizer():
    config = TokenizerConfig(
        # ----- Instruments -----
        use_programs=True,                         # Distinguish instruments via program numbers
        one_token_stream_for_programs=True,       # Merge instruments into one stream (if False → one stream per program)
        program_changes=True,                       # Add ProgramChange tokens

        # ----- Musical structure -----
        use_time_signatures=True,                   # Add TimeSignature tokens
        use_tempos=True,                            # Add Tempo tokens
        use_chords=True,                            # Detect and include chord tokens
        use_rests=True,                             # Represent silences as Rest tokens

        # ----- Velocity and duration -----
        include_velocity=True,                      # Include velocity tokens (for dynamics)
    )

    tokenizer = REMI(config)
    
    return tokenizer
    

def get_tokenized_data(midi_path:str):
    
    # 1. get tokenizer
    tokenizer = get_tokenizer()
    
    # 2. change midi_path string to Path objects
    midi_path = Path(midi_path)
    tokenized = []
    
    # 3. loop through every midi file in the midi_path
    for midi in tqdm(list(midi_path.glob("*.mid")), desc="Tokenizing MIDI files"):
        # 3.1. tokenize midi file
        tokens = tokenizer(midi)
        # 3.2. append the ids only to the tokenized list
        tokenized.append(tokens.ids)    
        
    return tokenized
        