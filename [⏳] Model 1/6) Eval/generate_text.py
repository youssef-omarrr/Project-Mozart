from prompt_utils import *
from output_utils import save_generated_piece
from filter import is_music_token  
from load_model_tokenizer import load_model_and_tokenizer

def clean_generated_text(text: str) -> str:
    """
    Post-process generated text to ensure only musical tokens remain in tracks
    """
    # Find the tracks section
    tracks_start = text.find("<TRACKS>")
    if tracks_start == -1:
        return text
        
    before_tracks = text[:tracks_start + len("<TRACKS>")]
    after_tracks = text[tracks_start + len("<TRACKS>"):]
    
    # Process the tracks section
    lines = []
    for line in after_tracks.split('<TRACKSEP>'):
        if ':' in line:
            inst, notes = line.split(':', 1)
            # Filter tokens
            clean_tokens = []
            for token in notes.split():
                if token != '<MASK>' and is_music_token(token):
                    clean_tokens.append(token)
            
            if clean_tokens:  # Only add if we have musical tokens
                lines.append(f"{inst.strip()}: {' '.join(clean_tokens)}")
        elif line.strip():  # Handle lines without colons
            clean_tokens = []
            for token in line.split():
                if token != '<MASK>' and is_music_token(token):
                    clean_tokens.append(token)
            if clean_tokens:
                lines.append(' '.join(clean_tokens))
    
    clean_after_tracks = ' <TRACKSEP> '.join(lines)
    return before_tracks + clean_after_tracks

def generate_text(
    prompt: str = "Say Hello World!",
    saved_txt_dir: str = "../../example_outputs/model_output/",
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    
    # 1. Get model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. Prepare prompt
    out_dict = parse_user_instruction(prompt)
    
    # Use masked prompt
    prompt, metadata = build_structured_prompt(out_dict)
    
    print("Generated prompt:", prompt)
    
    # 3. Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. Generate text - optimized for music generation
    outputs = model.generate(
        **inputs,
        do_sample=do_sample,      
        temperature=temperature,     
        top_p=top_p,
        max_new_tokens=256,  # Reduced - music tokens are more compact
        min_length=100,       # Reduced min length
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Higher penalty for repetition
        no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
        early_stopping=True      # Stop when EOS is generated
    )      

    # 5. Decode
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Clean the generated text to remove non-musical tokens
    text = clean_generated_text(text)
    
    print("Generated text:", text)
    
    # 6. Save output
    file_path = save_generated_piece(
        metadata=metadata,
        generated_text=text,
        save_dir=saved_txt_dir
    )
    
    return file_path

