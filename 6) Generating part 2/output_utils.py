import os
import re
import json


def save_generated_piece(metadata: dict, generated_text: str,
                        save_dir: str = "../example_outputs/decoded_pieces/") -> str:
    """
    Parse generated_text and write JSON file:
    {
        "name": ...,
        "bpm": ...,
        "duration_beats": ...,
        "duration_minutes": ...,
        "tracks": { "piano": ["Rest_q", ...], "cello": [...], ... }
    }
    
    """
    os.makedirs(save_dir, exist_ok=True)

    # Clean string
    s = (generated_text or "").strip()
    s = re.sub(r'\s+', ' ', s)  # normalize spaces

    # Extract the generated part (after the prompt)
    prompt_end = s.find("<TRACKS>") + len("<TRACKS>")
    if prompt_end > 0:
        s = s[prompt_end:].strip()

    tracks = {}

    # Handle masked instrument format: "Instrument: <MASK>" should become "Instrument: generated_notes"
    # Regex to find instrument patterns with potential masking
    pattern = re.compile(r'([A-Za-z][A-Za-z0-9_#\- ]{0,40}?):\s*([^:<]+|<?MASK>?)(?=\s+[A-Za-z][A-Za-z0-9_#\- ]{0,40}?:|$)')
    matches = pattern.findall(s)

    for inst, notes in matches:
        inst_key = inst.strip().lower()
        # If notes contain <MASK> or are empty, provide default notes
        if '<MASK>' in notes or not notes.strip() or notes.strip() == '<MASK>':
            # Provide some default musical content instead of empty/masked
            tokens = ["C4_q", "D4_q", "E4_q", "F4_q"]  # Simple C major scale
        else:
            tokens = [tok for tok in notes.strip().split() if tok]
        tracks[inst_key] = tokens

    # 3. Build final JSON structure
    final_output = {
        "name": metadata["name"],
        "bpm": metadata["bpm"],
        "duration_beats": metadata["duration_beats"],
        "duration_minutes": metadata["duration_minutes"],
        "tracks": tracks,
    }

    # auto-number files
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("piece_") and f.endswith(".txt")]
    if existing_files:
        nums = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        next_num = max(nums) + 1
    else:
        next_num = 1
    
    # Save the final output
    filepath = os.path.join(save_dir, f"piece_{next_num}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved piece to {filepath}")
    
    return filepath