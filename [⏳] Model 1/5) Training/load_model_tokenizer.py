import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


DEFAULT_BASE_MODEL = "facebook/bart-base"              # local base model folder (saved_pretrained)
MY_MODEL_DIR = "../../MODELS/Project_Mozart_bart"        # where LoRA adapters live
CACHE_DIR = "../../MODELS/"   
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
                                            cache_dir = cashed_dir,)
    


    # 2. Load model
    base_model  = AutoModelForSeq2SeqLM.from_pretrained(
                                            pretrained_model_name_or_path = base_model_dir,
                                            cache_dir=cashed_dir,
                                            dtype=torch.float16,
                                        ).to(device)

    # Resize embeddings if tokenizer has added tokens
    base_model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)
    
    # 3. Load LoRA with ignore_mismatched_sizes
    model = PeftModel.from_pretrained(
        base_model,
        my_model_dir,
        local_files_only=True,
        ignore_mismatched_sizes=True  # This allows size mismatches
        ).to(device)
            
                
    # Resize embeddings if tokenizer has added tokens
    model.resize_token_embeddings(len(tokenizer),
                                        mean_resizing=False)

    
    # 5. Set mode
    model.train() if for_training else model.eval()
    
    return model, tokenizer