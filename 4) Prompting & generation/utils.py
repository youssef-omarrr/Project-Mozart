import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

from prompt_utils import *

DEFAULT_BASE_MODEL = "../MODELS/gpt2-medium-local/"            # local base model folder (saved_pretrained)
MY_MODEL_DIR = "../MODELS/Project_Mozart_gpt2-medium"        # where LoRA adapters live
CACHE_DIR = "../MODELS/"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    my_model_dir: str = MY_MODEL_DIR,
    base_model_dir: str = DEFAULT_BASE_MODEL,
    cashed_dir: str = CACHE_DIR,
    device: str =   DEVICE):
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
                                            pretrained_model_name_or_path = my_model_dir,
                                            local_files_only = True,
                                            cache_dir = cashed_dir,
                                            )
    
    # 2. Load model
    base_model  = AutoModelForCausalLM.from_pretrained(
                                            pretrained_model_name_or_path = base_model_dir,
                                            cache_dir=cashed_dir,
                                            dtype=torch.float16,
                                        ).to(device)

    # Resize embeddings if tokenizer has added tokens
    base_model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)

    # 3. Apply LoRA
    model = PeftModel.from_pretrained(
                                base_model,
                                my_model_dir, 
                                local_files_only=True
                                ).to(device)

    model.eval()
    return model, tokenizer


def generate_text (prompt:str = "Say Hello World!",
                    saved_txt_dir:str = "../example_outputs/model_output/",
                    do_sample:bool =True,       # sampling instead of greedy
                    temperature:float =0.7,      # randomness
                    top_p:float =0.9,            # nucleus sampling
                    ):
    
    # 1. Get model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. Prepare prompt
    out_dict = parse_user_instruction(prompt)
    prompt, _ = build_structured_prompt(out_dict)
    
    # 3. Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 4. Generate text
    outputs = model.generate(
        **inputs,
        do_sample=do_sample,      
        temperature=temperature,     
        top_p=top_p,
    )         

    # 5. Decode
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    # 6. Make sure output directory exists
    os.makedirs(saved_txt_dir, exist_ok=True)
    

    # 7. Find the next file number
    existing_files = [f for f in os.listdir(saved_txt_dir) if f.startswith("output_") and f.endswith(".txt")]
    if existing_files:
        numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        next_num = max(numbers) + 1
    else:
        next_num = 1

    file_path = os.path.join(saved_txt_dir, f"output_{next_num}.txt")

    # 8. Save text
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved generated text to {file_path}")
    return text
    
        