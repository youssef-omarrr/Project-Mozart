import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from prompt_utils import *
from output_utils import save_generated_piece
from filter import MusicalTokenizerWrapper

DEFAULT_BASE_MODEL = "../MODELS/gpt2-medium-local/"            # local base model folder (saved_pretrained)
MY_MODEL_DIR = "../MODELS/Project_Mozart_gpt2-medium"        # where LoRA adapters live
CACHE_DIR = "../MODELS/"   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    my_model_dir: str = MY_MODEL_DIR,
    base_model_dir: str = DEFAULT_BASE_MODEL,
    cashed_dir: str = CACHE_DIR,
    device: str =   DEVICE,
    for_training:bool = False):
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
                                            pretrained_model_name_or_path = my_model_dir,
                                            local_files_only = True,
                                            cache_dir = cashed_dir,
                                            )
    

    # tokenizer = MusicalTokenizerWrapper(base_tokenizer)

    # 2. Load model
    base_model  = AutoModelForCausalLM.from_pretrained(
                                            pretrained_model_name_or_path = base_model_dir,
                                            cache_dir=cashed_dir,
                                            dtype=torch.float16,
                                        ).to(device)

    # Resize embeddings if tokenizer has added tokens
    base_model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)
    
    # 3. Try to load LoRA with ignore_mismatched_sizes
    try:
        model = PeftModel.from_pretrained(
            base_model,
            my_model_dir,
            local_files_only=True,
            ignore_mismatched_sizes=True  # This allows size mismatches
        ).to(device)
        
    except Exception as e:
        print(f"Could not load adapters: {e}. Creating new adapters...")
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        
    # Resize embeddings if tokenizer has added tokens
    model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)

    
    # Only set to eval mode if NOT for training
    if not for_training:
        model.eval()
    else:
        model.train()  # Set to training mode
        # Ensure LoRA parameters are trainable
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    
    return model, tokenizer

def generate_text(
    prompt: str = "Say Hello World!",
    saved_txt_dir: str = "../example_outputs/model_output/",
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
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 4. Generate text - increase max tokens for music generation
    outputs = model.generate(
        **inputs,
        do_sample=do_sample,      
        temperature=temperature,     
        top_p=top_p,
        max_new_tokens=512,  # Increased for music generation
        min_length=50,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1  # Reduce repetition
    )         

    # 5. Decode
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("Generated text:", text)
    
    # 6. Save output
    file_path = save_generated_piece(
        metadata=metadata,
        generated_text=text,
        save_dir=saved_txt_dir
    )
    
    return file_path
