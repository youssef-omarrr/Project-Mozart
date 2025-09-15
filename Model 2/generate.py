import os
import subprocess
from pydub import AudioSegment
import torch
from miditok import TokSequence

from notes_utils import *

# ------------------------------------------------------------
# FILE NAMING HELPER
# ------------------------------------------------------------
def name_file(output_dir, safe_name, extension, numerate=True):
    """
    Generate a unique filename in output_dir using safe_name and extension.
    Ensures incrementing suffix (_1, _2, ...) if duplicates exist.
    """
    if not extension.startswith("."):
        extension = "." + extension

    os.makedirs(output_dir, exist_ok=True)

    if numerate:
        existing_files = [
            f for f in os.listdir(output_dir) 
            if f.startswith(safe_name) and f.endswith(extension)
        ]
        numbers = []
        for f in existing_files:
            parts = f[:-len(extension)].split("_")  # strip extension, split
            if len(parts) > 1 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        next_number = max(numbers, default=0) + 1
        return os.path.join(output_dir, f"{safe_name}_{next_number}{extension}")
    else:
        return os.path.join(output_dir, f"{safe_name}{extension}")


# ------------------------------------------------------------
# MIDI → WAV CONVERSION
# ------------------------------------------------------------
def midi_to_wav(
    midi_path,
    soundfont_path="../soundfonts/AegeanSymphonicOrchestra-SND.sf2",
    output_dir="model_outputs/wav_files/",
    output_name=None,
    sample_rate=44100,
    gain=2.0,
    normalize=True,
    fluidsynth_path=r"C:\tools\fluidsynth\bin\fluidsynth.exe",
    print_details=True
):
    """
    Convert a MIDI file to WAV using FluidSynth and a specified SoundFont.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use helper to pick name
    if output_name is None:
        base = os.path.splitext(os.path.basename(midi_path))[0]
    else:
        base = output_name
    output_wav = name_file(output_dir, base, ".wav")

    if print_details:
        print(f"Rendering '{midi_path}' to WAV...")
        print(f"Using SoundFont: {soundfont_path}")
        print(f"Output path: {output_wav}")

    # Run FluidSynth
    cmd = [
        fluidsynth_path,
        "-a", "file",
        "-F", output_wav,
        "-ni", soundfont_path,
        "-r", str(sample_rate),
        "-g", str(gain),
        midi_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error running FluidSynth: {e}")
        raise

    # Normalize audio if requested
    if normalize:
        sound = AudioSegment.from_wav(output_wav)
        normalized = sound.normalize()
        normalized.export(output_wav, format="wav")

    if print_details:
        print(f"✅ Successfully converted MIDI to WAV: {output_wav}")

    return output_wav


# ------------------------------------------------------------
# GENERATION LOOP
# ------------------------------------------------------------
def generate_with_rhythm(
    model, tokenizer,
    start_symbols=["Bar_None", "Position_0", "C4_q", "E4_q", "G4_q"],  # Start with bar structure
    max_len=2000,
    output_dir="model_outputs/midi_files",
    base_name="generated_rhythm",
    max_input_len_window=1024,
    default_velocity=90,
    stop_on_eos=True,
    temperature=0.8
):
    """
    Generate a sequence of tokens with proper Bar, Position, Pitch, Velocity, Duration structure.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Convert start symbols to token ids
    start_ids = []
    for sym in start_symbols:
        if sym.startswith(("Bar_", "Position_")):
            # Handle Bar and Position tokens directly
            vocab, _ = build_vocab_maps(tokenizer)
            if sym in vocab:
                start_ids.append(int(vocab[sym]))
        else:
            # Handle note symbols
            start_ids.extend(
                note_token_ids_from_symbolic(tokenizer, sym, default_velocity)
            )

    ids = start_ids.copy()
    vocab, id_to_token = build_vocab_maps(tokenizer)
    eos_id = int(vocab["EOS_None"]) if "EOS_None" in vocab else None
    
    # Track expected token sequence
    # After Bar -> expect Position
    # After Position -> expect Pitch  
    # After Pitch -> expect Velocity
    # After Velocity -> expect Duration
    # After Duration -> expect Position or Bar (new measure)
    expected_next = "Position"  # Start expecting position after initial bar

    with torch.no_grad():
        for step in range(max_len):
            window = ids[-max_input_len_window:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, device=device)

            # Apply temperature
            logits = logits[0, -1] / temperature
            
            # Get valid candidates based on expected token type
            valid_candidates = []
            
            for token_id, token_name in id_to_token.items():
                is_valid = False
                
                if expected_next == "Bar" and token_name.startswith("Bar_"):
                    is_valid = True
                elif expected_next == "Position" and token_name.startswith("Position_"):
                    is_valid = True
                elif expected_next == "Pitch" and token_name.startswith("Pitch_"):
                    is_valid = True
                elif expected_next == "Velocity" and token_name.startswith("Velocity_"):
                    is_valid = True
                elif expected_next == "Duration" and token_name.startswith("Duration_"):
                    is_valid = True
                # Special case: after Duration, allow both Position and Bar
                elif expected_next == "Position_or_Bar" and (token_name.startswith("Position_") or token_name.startswith("Bar_")):
                    is_valid = True
                
                if is_valid:
                    valid_candidates.append(token_id)
            
            if not valid_candidates:
                print(f"No valid candidates for expected_next='{expected_next}' at step {step}")
                # Fallback: allow any reasonable token
                for token_id, token_name in id_to_token.items():
                    if token_name.startswith(("Bar_", "Position_", "Pitch_")):
                        valid_candidates.append(token_id)
                        break
                if not valid_candidates:
                    break
            
            # Filter logits to only valid candidates
            filtered_logits = torch.full_like(logits, float('-inf'))
            for candidate_id in valid_candidates:
                if candidate_id < len(filtered_logits):
                    filtered_logits[candidate_id] = logits[candidate_id]
            
            # Sample from filtered distribution
            probs = torch.softmax(filtered_logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            token_name = id_to_token.get(next_id, "")
            
            ids.append(next_id)
            
            # Update expected next token based on what we just generated
            if token_name.startswith("Bar_"):
                expected_next = "Position"
            elif token_name.startswith("Position_"):
                expected_next = "Pitch"
            elif token_name.startswith("Pitch_"):
                expected_next = "Velocity"
            elif token_name.startswith("Velocity_"):
                expected_next = "Duration"
            elif token_name.startswith("Duration_"):
                # After duration, we can have either a new position in same bar or new bar
                expected_next = "Position_or_Bar"
            
            if stop_on_eos and eos_id is not None and next_id == eos_id:
                print(f"Hit EOS at step {step+1}")
                break

            if (step + 1) % 500 == 0:
                print(f"Generated {step+1} tokens (current length {len(ids)})...")

    print(f"Final sequence length: {len(ids)}")
    
    # Print first 50 tokens to verify structure
    print("First 50 generated tokens:")
    for i, token_id in enumerate(ids[:50]):
        token_name = id_to_token.get(token_id, f"Unknown_{token_id}")
        print(f"{i} {token_name}")

    # Validate the generated sequence
    validation_results = validate_token_sequence(tokenizer, ids)
    
    print(f"\nSequence validation:")
    print(f"Valid transitions: {validation_results['valid_transitions']}")
    print(f"Invalid transitions: {validation_results['invalid_transitions']}")
    print(f"Accuracy: {validation_results['accuracy']:.2%}")

    # --- Save MIDI ---
    out_midi = name_file(output_dir, base_name, ".mid")
    seq = TokSequence(ids=ids)
    decoded = tokenizer.decode([seq])
    midi_obj = decoded[0] if isinstance(decoded, (list, tuple)) else decoded

    if hasattr(midi_obj, "dump_midi"):
        midi_obj.dump_midi(out_midi)
    elif hasattr(midi_obj, "dumps_midi"):
        data = midi_obj.dumps_midi()
        with open(out_midi, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode())
    else:
        raise RuntimeError("Decoded object has no known dump method")

    print(f"✅ Saved MIDI to {out_midi}")

    # --- Render WAV ---
    try:
        midi_to_wav(out_midi, output_dir=output_dir)
    except Exception as e:
        print(f"Warning: Could not render WAV: {e}")

    return ids
