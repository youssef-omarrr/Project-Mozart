import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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


def update_model_with_mask_token():
    """Update existing model and tokenizer with MASK token"""
    
    # 1. Load and update tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MY_MODEL_DIR,
        local_files_only=True,
        cache_dir=CACHE_DIR,
    )
    
    # Add MASK token if not already present
    if "<MASK>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<MASK>"])
        print(f"Added <MASK> token. New vocab size: {len(tokenizer)}")
    else:
        print("<MASK> token already exists in vocabulary")
    
    print("Loading base model with correct vocab size...")
    # Load base model first and resize embeddings
    base_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE_MODEL,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
    )
    
    # Resize base model embeddings FIRST
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Now load the LoRA adapters
    print("Loading LoRA adapters...")
    try:
        # Try to load with ignore_mismatched_sizes
        model = PeftModel.from_pretrained(
            base_model,
            MY_MODEL_DIR,
            local_files_only=True,
            ignore_mismatched_sizes=True  # This is key!
        )
    except:
        print("Could not load adapters with mismatched sizes. Creating new adapters...")
        # If that fails, create new adapters from scratch
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    
    model = model.to(DEVICE)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("Saving updated model...")
    model.save_pretrained(MY_MODEL_DIR)
    tokenizer.save_pretrained(MY_MODEL_DIR)
    
    print("Update complete!")