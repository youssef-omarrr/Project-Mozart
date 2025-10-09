import torch
import torch.nn.functional as F
from random import randint
from tqdm import tqdm
from miditok import TokSequence
from data.starting_seq import first_six

# Load model and tokenizer and generate data
# -------------------------------------------
def generate_music(model,
                    tokenizer,
                    continue_gen_midi:str = None,
                    continue_gen_tensor:torch.Tensor = None,
                    file_name:str = "Generated_Mozart",
                    max_new_tokens:int = 2048,
                    temperature:float = 0.95,
                    top_k:int = 20,
                    ):
    """
    Generate music tokens from ProjectMozart model and save MIDI/WAV outputs.

    Parameters
    - model: A PyTorch model with a positional_encoding attribute (used to determine max sequence length).
    - tokenizer: A tokenizer object (REMI).
    - continue_gen_midi (str, optional): Path to a MIDI file to continue generation from.
        > If provided, it is tokenized and used as the starting context. Mutually exclusive with continue_gen_tensor.
    - continue_gen_tensor (torch.Tensor, optional): Tensor of token ids to continue generation from.
        > If provided, it is used as the starting context. Mutually exclusive with continue_gen_midi.
    - file_name (str): Base name used when saving generated files.
    - max_new_tokens (int): Number of new tokens to generate.
    - temperature (float): Sampling temperature to control randomness.
    - top_k (int): If >0, sample from top_k tokens; if 0, use greedy decoding.

    Notes
    - Only one of continue_gen_tensor or continue_gen_midi should be active at a time.
        If both are provided, continue_gen_midi takes precedence.
    - Side effects: saves a MIDI file and a WAV file into model_outputs/* directories.
    - Returns: torch.Tensor of generated token ids with shape (1, total_length).
    """
    
    # 1. Put model in eval mode and to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    
    # 2.0. Select starting tokens according to given parameters
    # 2.1. Continue from given midi file
    if continue_gen_midi is not None:
        tokens = tokenizer(continue_gen_midi)
        tokens = torch.tensor(tokens.ids).unsqueeze(0).to(device)
        print("[Loaded] Continuing generation from given midi file...")
        
    # 2.2. Continue from given sequence of tokens
    elif continue_gen_tensor is not None:
        tokens = continue_gen_tensor.to(device)
        print("[Loaded] Continuing generation from given tensor...")
        
    # 2.3. Select a starting sequence randomly    
    else:
        tokens = first_six[randint(0, 39)].unsqueeze(0).to(device)
        
    
    # 2.4. Create another tensor to concat all the tokens 
    # if we wanted to generate above the model's limit (512)
    total_tokens = tokens
    
    # 2.5. The model's max_seq_len
    model_max_seq_len = model.positional_encoding.size(1)
    
    # 3. Start generating
    with torch.inference_mode():
    
        for _ in tqdm(range(max_new_tokens), desc="Generating..."):
            
            # 4. Trim the tokens if their length exceeds the length of the max_seq_len of the model
            if (tokens.size(1) > model_max_seq_len):
                tokens = tokens[:, -model_max_seq_len:]
            
            # 5. Forward pass and take the last token only (the predicted token)
            logits = model(tokens) # (1, seq_len, vocab_size)
            next_token = logits[:, -1, :]  # Take the last token only
            probs = F.softmax(next_token / temperature, dim=-1) # Apply temperature for more random results
            
            # 6. Choose randomly from the top_k probs
            if top_k:
                # keep only top_k tokens
                top_probs, top_idx = torch.topk(probs, top_k) 
                # normalize
                top_probs = top_probs/torch.sum(top_probs) 
                # take one sample token from the top_k probability distribution.
                next_token = torch.multinomial(top_probs, 1)
                next_token = top_idx.gather(-1, next_token)
            else:
                # only sample the highest prob (greedy decoding)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                
            # 7.1. Concat the newly added token to the sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            # 7.2. concat total_tokens seperatly as 'tokens' can be trimmed
            total_tokens = torch.cat([total_tokens, next_token], dim=1)
        
    # 8. move them to cpu and adjust thier shape
    total_tokens = total_tokens.squeeze(0).to("cpu")
    
    # 9. Convert them to toksequence to be able to use 'complete_sequence' function
    seq = TokSequence(ids=total_tokens.tolist())
    tokenizer.complete_sequence(seq)
    
    # 10. Decode and save file
    midi = tokenizer.decode(seq)
    midi_name = name_file("model_outputs/midi_files", file_name, ".mid")
    midi.dump_midi(midi_name)
    print("midi created")
    
    # 11. Convert midi to wav
    wav_name = name_file("model_outputs/wav_files", file_name, ".wav")
    midi_to_wav(midi_name, output_name=file_name)

    print(f"Saved: {midi_name} and {wav_name}")
    
    return total_tokens.unsqueeze(0)


# ------------------------------------------------------------ #
# ==========  HELPER FUNCTIONS  ============================== #
# ------------------------------------------------------------ #

import os
import subprocess
from pydub import AudioSegment

# FILE NAMING HELPER
# --------------------
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
# MIDI qo WAV CONVERSION
# ------------------------------------------------------------
def midi_to_wav(
    midi_path,
    soundfont_path="../soundfonts/AegeanSymphonicOrchestra-SND.sf2",
    output_dir="model_outputs/wav_files/",
    output_name=None,
    sample_rate=44100,
    gain=3.0,
    normalize=True,
    fluidsynth_path=r"C:\tools\fluidsynth\bin\fluidsynth.exe",
    print_details=False
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
        print(f"Successfully converted MIDI to WAV: {output_wav}")

    return output_wav