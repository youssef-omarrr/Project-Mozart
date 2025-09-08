############################################################
# IMPORTS
############################################################
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

from transformers import get_linear_schedule_with_warmup

from filter import get_unique_tokens
from loss_fn import MusicTokenEnforcementLoss
from prepare_dataset import prepare_dataset

############################################################
# CONSTANTS / CONFIG
############################################################
OUTPUT_DIR = "../../MODELS/Project_Mozart_bart-small"

# Enhanced training hyperparams
EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

MAX_GRAD_NORM = 1.0
USE_FP16 = True
SAVE_EVERY_N_STEPS = 1000
EVAL_EVERY_N_STEPS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# Custom loss weights
NOTE_DENSITY_REWARD = 0.3       # Mild reward for more notes
MIN_NOTES_THRESHOLD = 50         # Minimum notes per sequence

MASK_TOKEN = "<MASK>"
MUSIC_NOTES = get_unique_tokens()
MUSIC_TOKEN_IDS = None  # Will be populated during initialization


############################################################
# UTILITIES
############################################################
def get_music_token_ids(tokenizer):
    """Convert music tokens to token IDs"""
    ids = []
    missing_tokens = []
    for note in MUSIC_NOTES:
        tid = tokenizer.convert_tokens_to_ids(note)
        if tid is None or tid < 0 or tid >= tokenizer.vocab_size:
            missing_tokens.append((note, tid))
        else:
            ids.append(int(tid))
    if missing_tokens:
        print(f"[WARN] Some MUSIC_NOTES missing in tokenizer: {missing_tokens[:10]} (showing up to 10)")
    return ids


def collate_fn(samples, tokenizer):
    """
    Collate function for input-target pairs.
    is_input: True for masked inputs, False for unmasked targets
    """
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]
    attention_mask = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in samples]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def create_training_batch(input_batch, target_batch, tokenizer):
    """
    Create training batch from input (masked) and target (unmasked) batches.
    """
    # Use masked inputs for model input
    batch = {
        "input_ids": input_batch["input_ids"],
        "attention_mask": input_batch["attention_mask"]
    }
    
    # Use unmasked targets as labels
    labels = target_batch["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
    batch["labels"] = labels
    
    return batch


############################################################
# TRAIN / EVAL LOOP
############################################################

def train_loop(model, tokenizer, tokenized):
    
    global MUSIC_TOKEN_IDS
    MUSIC_TOKEN_IDS = get_music_token_ids(tokenizer)

    # Loss function
    custom_loss_fn = MusicTokenEnforcementLoss(tokenizer, MUSIC_TOKEN_IDS, non_music_penalty=100.0)

    # Separate dataloaders for inputs and targets
    train_input_dataloader = DataLoader(
        tokenized["train_input"], 
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False,  # Important: don't shuffle to maintain input-target correspondence
        collate_fn=lambda s: collate_fn(s, tokenizer, is_input=True), 
        pin_memory=True
    )
    
    train_target_dataloader = DataLoader(
        tokenized["train_target"], 
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False,  # Important: don't shuffle to maintain input-target correspondence
        collate_fn=lambda s: collate_fn(s, tokenizer, is_input=False), 
        pin_memory=True
    )
    
    val_input_dataloader = DataLoader(
        tokenized["test_input"], 
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False, 
        collate_fn=lambda s: collate_fn(s, tokenizer, is_input=True), 
        pin_memory=True
    )
    
    val_target_dataloader = DataLoader(
        tokenized["test_target"], 
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False, 
        collate_fn=lambda s: collate_fn(s, tokenizer, is_input=False), 
        pin_memory=True
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler and Scaler
    total_steps = len(train_input_dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=total_steps)

    scaler = torch.amp.GradScaler(device=DEVICE) if USE_FP16 and DEVICE == "cuda" else None

    metrics_history = defaultdict(list)
    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        steps_in_epoch = 0
        
        # Zip the input and target dataloaders
        paired_dataloader = zip(train_input_dataloader, train_target_dataloader)
        total_batches = min(len(train_input_dataloader), len(train_target_dataloader))
        
        pbar = tqdm(paired_dataloader, total=total_batches, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for step, (input_batch, target_batch) in enumerate(pbar):
            # Create training batch from input-target pair
            batch = create_training_batch(input_batch, target_batch, tokenizer)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_loss, loss_components = custom_loss_fn(
                    outputs.logits, 
                    batch["labels"],
                    batch["attention_mask"]
                )
                loss_reduced = total_loss / GRAD_ACCUM_STEPS

            # Backward
            if scaler is not None:
                scaler.scale(loss_reduced).backward()
            else:
                loss_reduced.backward()

            epoch_loss += total_loss.item()
            steps_in_epoch += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'ce': f'{loss_components["ce_loss"]:.4f}',
                'penalty': f'{loss_components["non_music_penalty"]:.4f}'
            })

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % SAVE_EVERY_N_STEPS == 0:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    print(f"\nSaved checkpoint at step {global_step}")

                if global_step % EVAL_EVERY_N_STEPS == 0:
                    val_metrics = evaluate_model(
                        model, custom_loss_fn, 
                        val_input_dataloader, val_target_dataloader, tokenizer
                    )
                    print(f"\nValidation at step {global_step}: {val_metrics}")
                    metrics_history['val_loss'].append(val_metrics['val_loss'])

        avg_train_loss = epoch_loss / max(1, steps_in_epoch)
        metrics_history['train_loss'].append(avg_train_loss)
        
        val_metrics = evaluate_model(
            model, custom_loss_fn, 
            val_input_dataloader, val_target_dataloader, tokenizer
        )
        print(f"\nEpoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save model after each epoch
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model and tokenizer saved at {OUTPUT_DIR}")

    return metrics_history


def evaluate_model(model, loss_fn, input_dataloader, target_dataloader, tokenizer):
    """
    Evaluate model using paired input and target dataloaders.
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_non_music_predictions = 0

    with torch.no_grad():
        paired_dataloader = zip(input_dataloader, target_dataloader)
        for input_batch, target_batch in tqdm(paired_dataloader, desc="Evaluating"):
            batch = create_training_batch(input_batch, target_batch, tokenizer)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss, loss_components = loss_fn(
                    outputs.logits, 
                    batch["labels"], 
                    batch["attention_mask"]
                )
                total_loss += loss.item()
                total_non_music_predictions += loss_components.get('non_music_predictions', 0)
                total_steps += 1

    model.train()
    return {
        'val_loss': total_loss / max(1, total_steps),
        'non_music_predictions': total_non_music_predictions
    }


############################################################
# ENTRY POINT
############################################################
def train_lora(model, tokenizer):
    """Main training function"""
    model.train()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

    # Prepare dataset - now returns separate input and target datasets
    tokenized = prepare_dataset(tokenizer)
    
    # Verify we have all expected datasets
    expected_keys = ["train_input", "train_target", "test_input", "test_target"]
    for key in expected_keys:
        if key not in tokenized:
            raise ValueError(f"Missing dataset key: {key}")
        print(f"{key}: {len(tokenized[key])} samples")
    
    metrics_history = train_loop(model, tokenizer, tokenized)
    
    return metrics_history