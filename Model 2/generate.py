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
    start_symbols=["C4_q", "E4_q", "G4_q"],
    max_len=2000,
    output_dir="model_outputs/midi_files",
    base_name="generated_rhythm",
    max_input_len_window=1024,
    default_velocity=90,
    stop_on_eos=True
):
    """
    Generate a sequence of tokens from a trained model and save MIDI + WAV.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Convert start symbols to token ids
    start_ids = []
    for sym in start_symbols:
        start_ids.extend(
            note_token_ids_from_symbolic(tokenizer, sym, default_velocity)
        )

    ids = start_ids.copy()
    vocab, id_to_token = build_vocab_maps(tokenizer)

    eos_id = int(vocab["EOS_None"]) if "EOS_None" in vocab else None

    with torch.no_grad():
        for step in range(max_len):
            window = ids[-max_input_len_window:]
            x = torch.tensor([window], dtype=torch.long, device=device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, device=device)

            next_id = int(torch.argmax(logits[0, -1]).item())
            token_name = id_to_token.get(next_id, "")

            # --- Smarter Rhythm Filtering ---
            last_note_tokens = [
                id_to_token.get(t, "") for t in reversed(ids)
                if not id_to_token.get(t, "").startswith(("Bar", "Position"))
            ]
            last_type = token_type(last_note_tokens[0]) if last_note_tokens else None

            if last_type == "Pitch":
                expected = "Velocity"
            elif last_type == "Velocity":
                expected = "Duration"
            else:
                expected = "Pitch"

            if token_name.startswith(("Bar", "Position")) or token_name.startswith(expected):
                ids.append(next_id)

            if stop_on_eos and eos_id is not None and next_id == eos_id:
                print(f"Hit EOS at step {step+1}")
                break

            if (step + 1) % 500 == 0:
                print(f"Generated {step+1} tokens (current length {len(ids)})...")

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
    midi_to_wav(out_midi, output_dir=output_dir)

    return ids
