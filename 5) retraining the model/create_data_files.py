import json
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

DATA_DIR = Path("../dataset/text")
TRAIN_OUT = Path("../dataset/train_file.txt")
TEST_OUT = Path("../dataset/test_file.txt")
MASK_PROBABILITY = 0.3  # Probability of masking an instrument

def flatten_tracks(tracks: dict, mask_instruments: bool = False) -> str:
    """
    Converts the dictionary of tracks into a single string with optional masking.
    
    Args:
        tracks (dict): A dictionary where keys are instrument names and
                        values are lists of musical tokens.
        mask_instruments (bool): Whether to randomly mask some instruments
    Returns:
        str: A single-line string representation of all tracks.
    """
    lines = []
    
    for name, tokens in tracks.items():
        # Apply masking with probability
        if mask_instruments and random.random() < MASK_PROBABILITY:
            # Mask the instrument by replacing notes with <MASK> token
            lines.append(f"{name}: <MASK>")
        else:
            # Format: "InstrumentName:token token token..."
            lines.append(f"{name}:" + " ".join(tokens))
        
    # Join all track strings with a separator token.
    return " <TRACKSEP> ".join(lines)

def make_example(data, mask_instruments=False):
    """
    Constructs a single training example string with optional instrument masking.
    
    Args:
        data (dict): The dictionary loaded from a JSON/TXT file.
        mask_instruments (bool): Whether to mask some instruments in this example
        
    Returns:
        str: The fully formatted training example string.
    """
    # Safely get metadata, defaulting to empty strings if not found.
    name = data.get("name", "")
    bpm = data.get("bpm", "")
    
    dur_beats = data.get("duration_beats", "")
    dur_minutes = data.get("duration_minutes", "")
    
    # Flatten the 'tracks' dictionary into a single string.
    tracks = flatten_tracks(data.get("tracks", {}), mask_instruments)
    
    # Assemble the final string using an f-string for clarity.
    ex = (
        "<|startofpiece|>"
        f"<NAME={name}>"
        f"<BPM={bpm}>"
        f"<DURATION_BEATS={dur_beats}>"
        f"<DURATION_MINUTES={dur_minutes}>"
        "<TRACKS>"
        f"{tracks}"
        "<|endofpiece|>"
    )
    return ex

def create_training_data():
    # Find all .txt files in the data directory.
    files = list(DATA_DIR.glob("*.txt"))
    
    # Split files into train and test sets (80% train, 20% test)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    print(f"Found {len(files)} files, splitting into {len(train_files)} train and {len(test_files)} test files")
    
    # Create training data with masked examples
    with TRAIN_OUT.open("w", encoding="utf-8") as train_fo:
        for p in train_files:
            try:
                txt = p.read_text(encoding="utf-8")
                data = json.loads(txt)
                
                # Create multiple masked versions for each training piece
                for _ in range(3):  # Create 3 masked versions per training piece
                    masked_example = make_example(data, mask_instruments=True)
                    train_fo.write(masked_example.strip() + "\n")
                    
            except Exception as e:
                print("skip training file", p, e)
    
    # Create test data with original examples
    with TEST_OUT.open("w", encoding="utf-8") as test_fo:
        for p in test_files:
            try:
                txt = p.read_text(encoding="utf-8")
                data = json.loads(txt)
                
                # Create original (non-masked) examples for testing
                original_example = make_example(data, mask_instruments=False)
                test_fo.write(original_example.strip() + "\n")
                    
            except Exception as e:
                print("skip test file", p, e)

# Specify special tokens used in the data (to be used in your training script)
special_tokens = {
    "bos_token": "<|startofpiece|>",
    "eos_token": "<|endofpiece|>",
    "pad_token": "<pad>",
    "additional_special_tokens": [
        "<TRACKS>", "<TRACKSEP>", "<NAME=", "<BPM=", 
        "<DURATION_BEATS=", "<DURATION_MINUTES=", "<MASK>"
    ],
}

if __name__ == "__main__":
    create_training_data()
    print(f"Created training data with instrument masking at {TRAIN_OUT}")
    print(f"Created test data with original examples at {TEST_OUT}")