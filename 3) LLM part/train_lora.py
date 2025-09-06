############################################################
# Imports
# - Hugging Face Transformers: model, trainer, args, collator
# - PEFT: for LoRA fine-tuning with parameter-efficient methods
# - Datasets: dataset loading utilities
# - Torch: for tensor operations & dtype settings
############################################################
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from datasets import load_dataset
import torch


############################################################
# Constants / Configurations
############################################################
MODEL_NAME = "distilbert/distilgpt2"      # Base model (small GPT-2 distilled)
DOWNLOAD_PATH = "../MODELS/"              # Cache directory for model
TRAIN_FILE = "../dataset/train.txt"       # Training text dataset
OUTPUT_DIR = "../MODELS/Project_Mozart_distilgpt2"  # Output path for fine-tuned model


############################################################
# Step 1: Prepare Model, Dataset, and Training Setup
############################################################
def prepare_units(tokenizer):
    """
    Prepares the model, dataset, training arguments, and data collator.
    Args:
        tokenizer: The tokenizer used for text preprocessing.
    Returns:
        model: The LoRA-wrapped model ready for training.
        training_args: TrainingArguments for Hugging Face Trainer.
        tokenized: Tokenized dataset for training.
        data_collator: Collator for language modeling.
    """

    # 1) Load base model in 8-bit precision (memory efficient)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=DOWNLOAD_PATH,
        torch_dtype=torch.float16,   # Reduce memory usage with half precision
    )
    
    # move to cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Prepare model for k-bit (quantized) training
    model = prepare_model_for_kbit_training(model)
    
    # 2) Resize embeddings to match tokenizer vocabulary
    model.resize_token_embeddings(len(tokenizer))
    
    # 3) LoRA configuration (lightweight fine-tuning)
    lora_config = LoraConfig(
        r=8,                         # Rank (low-rank adaptation)
        lora_alpha=16,               # Scaling factor
        target_modules=["c_attn", "c_proj"],  # GPT-2 layers where LoRA is applied
        lora_dropout=0.05,           # Dropout for regularization
        bias="none",                 # Bias not trained
        task_type="CAUSAL_LM",       # Task: causal language modeling
    )
    
    # Wrap base model with LoRA
    model = get_peft_model(model, lora_config)
    
    # 4) Load dataset (expects a plain text file)
    dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
    
    # 5) Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,          # Truncate to max length
            max_length=1024           # Maximum sequence length
        )
    
    # Apply tokenization across dataset
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,                # Process multiple examples at once
        remove_columns=["text"]      # Remove raw text to save memory
    )
    
    # Data collator: groups tokenized samples into batches
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False                     # No masked LM; causal LM task
    )
    
    # 6) Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,        # Where to save checkpoints/models
        
        per_device_train_batch_size=1,   # Small batch size due to memory limits
        gradient_accumulation_steps=8,   # Simulates larger batch size (1*8=8)
        
        num_train_epochs=10,          # Number of training epochs
        save_strategy="epoch",        # Save model after each epoch
        logging_steps=50,             # Log training progress every 50 steps
        fp16=True,                    # Mixed precision (faster + memory efficient)
        
        optim="paged_adamw_8bit",     # Optimizer compatible with 8-bit training
        learning_rate=2e-4,           # Higher LR for LoRA fine-tuning
        weight_decay=0.01,            # Regularization
        
        lr_scheduler_type="cosine",   # Cosine decay schedule
        warmup_steps=50,              # Warmup to stabilize training
        
        gradient_checkpointing=True,  # Saves memory by recomputing gradients
        push_to_hub=False,            # Disable auto-upload
        report_to="none",             # No logging integration (e.g., wandb)
    )
    
    return model, training_args, tokenized, data_collator
        

############################################################
# Step 2: Train LoRA Model
############################################################
def train_lora(tokenizer):
    """
    Trains the model with LoRA fine-tuning.
    Args:
        tokenizer: The tokenizer used for preprocessing the dataset.
    """
    
    # Prepare model, data, and training arguments
    model, args, tokenized, data_collator = prepare_units(tokenizer)
    
    # 7) Initialize Trainer (Hugging Face training loop)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=data_collator
    )
    
    # Start training
    trainer.train()
    
    # Save trained LoRA model to output directory
    trainer.save_model(OUTPUT_DIR)
