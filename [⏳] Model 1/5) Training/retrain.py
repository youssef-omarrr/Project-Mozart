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
    
    # Get the actual vocabulary size that includes added tokens
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    print(f"\nTokenizer vocab_size: {tokenizer.vocab_size}, added tokens: {len(tokenizer.get_added_vocab())}, total: {vocab_size}")
    
    for note in MUSIC_NOTES:
        tid = tokenizer.convert_tokens_to_ids(note)
        if tid is None or tid < 0 or tid >= vocab_size:
            missing_tokens.append((note, tid))
        else:
            ids.append(int(tid))
    if missing_tokens:
        print(f"[WARN] Some MUSIC_NOTES missing in tokenizer: {missing_tokens[:10]} (showing up to 10), total of {len(missing_tokens)}")
    return ids


def collate_fn(samples, tokenizer):
    """
    Collate function for input-target pairs.
    """
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in samples]
    attention_mask = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in samples]
    
    labels = [torch.tensor(s["labels"], dtype=torch.long) for s in samples]
    mask_positions = [torch.tensor(s["mask_positions"], dtype=torch.long) for s in samples]
    
    # Pad to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    mask_positions = torch.nn.utils.rnn.pad_sequence(mask_positions, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mask_positions": mask_positions
    }


############################################################
# TRAIN / EVAL LOOP
############################################################

def train_loop(model, tokenizer, tokenized):
    
    global MUSIC_TOKEN_IDS
    MUSIC_TOKEN_IDS = get_music_token_ids(tokenizer)
    print(f"Found {len(MUSIC_TOKEN_IDS)} music token IDs\n")

    # Loss function
    custom_loss_fn = MusicTokenEnforcementLoss(
        tokenizer, MUSIC_TOKEN_IDS, non_music_penalty=5000.0
    )

    # Unified dataloaders
    train_dataloader = DataLoader(
        tokenized["train"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda s: collate_fn(s, tokenizer),
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        tokenized["test"],
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda s: collate_fn(s, tokenizer),
        pin_memory=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Scheduler and Scaler
    total_steps = len(train_dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(device=DEVICE) if USE_FP16 and DEVICE == "cuda" else None

    metrics_history = defaultdict(list)
    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        steps_in_epoch = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss, loss_components = custom_loss_fn(
                    outputs.logits,
                    batch["labels"],
                    batch["attention_mask"],
                    mask_positions=batch["mask_positions"],
                )

                    # --- DEBUG sanity check ---
                if step == 0 and epoch == 0:  # only print once at the very start
                    pred_ids = outputs.logits.argmax(dim=-1)
                    mask_pos = batch["mask_positions"].bool()

                    # Slice to first 20 predictions/labels at mask positions
                    preds = tokenizer.batch_decode(pred_ids[mask_pos][:20].tolist())
                    labels = tokenizer.batch_decode(batch["labels"][mask_pos][:20].tolist())

                    print("\n=== DEBUG SANITY CHECK ===")
                    print("Predicted tokens:", preds)
                    print("Label tokens:    ", labels)
                    print("==========================\n")


            # Backward
            loss_reduced = total_loss / GRAD_ACCUM_STEPS
            if scaler is not None:
                scaler.scale(loss_reduced).backward()
            else:
                loss_reduced.backward()

            epoch_loss += total_loss.item()
            steps_in_epoch += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "ce": f"{loss_components['ce_loss']:.4f}",
                    "penalty": f"{loss_components['non_music_penalty']:.4f}",
                }
            )

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
                    print(f"\nSaved Model and tokenizer checkpoint at step {global_step}")

                if global_step % EVAL_EVERY_N_STEPS == 0:
                    val_metrics = evaluate_model(model, custom_loss_fn, val_dataloader)

                    print(f"\nValidation at step {global_step}:")
                    print(f"  Total Loss: {val_metrics['val_loss']:.4f}")
                    
                    print(f"  CE Loss: {val_metrics['val_ce_loss']:.4f}")
                    print(f"  Penalty Loss: {val_metrics['val_penalty_loss']:.4f}")
                    print(
                        f"  Non-Music Errors: {val_metrics['non_music_predictions']}/{val_metrics['total_mask_positions']}"
                    )
                    
                    print(f"  Non-music rate: {val_metrics['non_music_rate']:.3%}")
                    print(f"  Music rate: {val_metrics['music_rate']:.3%}")
                    print(f"  Exact-match rate: {val_metrics['exact_match_rate']:.3%}")


                    metrics_history["val_loss"].append(val_metrics["val_loss"])

        avg_train_loss = epoch_loss / max(1, steps_in_epoch)
        metrics_history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate_model(model, custom_loss_fn, val_dataloader)

        print(f"\nEpoch {epoch+1} finished.")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        
        print(f"  Val Total Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val CE Loss: {val_metrics['val_ce_loss']:.4f}")
        print(f"  Val Penalty Loss: {val_metrics['val_penalty_loss']:.4f}")
        
        print(f"  Non-music rate: {val_metrics['non_music_rate']:.3%}")
        print(f"  Music rate: {val_metrics['music_rate']:.3%}")
        print(f"  Exact-match rate: {val_metrics['exact_match_rate']:.3%}")

        # Save model after each epoch
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model and tokenizer saved at {OUTPUT_DIR}")

    return metrics_history


def evaluate_model(model, loss_fn, dataloader, tokenizer=None):
    """
    Evaluate model using a single dataloader (already contains input/labels).
    Compare greedy predictions at mask positions to labels and to MUSIC_TOKEN_IDS.
    """
    model.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_penalty_loss = 0.0
    total_steps = 0

    total_mask_positions = 0
    total_non_music_preds = 0      # preds that are NOT music tokens
    total_exact_matches = 0        # pred == label at mask pos
    total_music_preds = 0          # preds that are music tokens (regardless of exact match)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Count masked positions (ground-truth positions we care about)
            batch_mask_count = int(batch["mask_positions"].sum().item())
            total_mask_positions += batch_mask_count

            with torch.amp.autocast(enabled=USE_FP16, device_type=DEVICE):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss, loss_components = loss_fn(
                    outputs.logits,
                    batch["labels"],
                    batch["attention_mask"],
                    mask_positions=batch["mask_positions"],
                )

            total_loss += loss.item()
            total_ce_loss += loss_components.get("ce_loss", 0.0)
            total_penalty_loss += loss_components.get("non_music_penalty", 0.0)
            total_steps += 1

            # ---- METRICS FROM PREDICTIONS ----
            # outputs.logits: (batch, seq_len, vocab)
            pred_ids = outputs.logits.argmax(dim=-1)  # shape: [B, S]
            labels = batch["labels"]                  # shape: [B, S]
            mask_pos = batch["mask_positions"].bool() # shape: [B, S]

            # Ensure MUSIC_TOKEN_IDS set (use python set for fast membership)
            music_id_set = set(MUSIC_TOKEN_IDS)

            # Vectorized checks per batch
            B, S = pred_ids.shape
            # Flatten to iterate only masked positions
            pred_flat = pred_ids[mask_pos]
            label_flat = labels[mask_pos]

            if pred_flat.numel() > 0:
                # Count how many predictions are music tokens
                # convert pred_flat to cpu numpy for membership check (fast for large vocab)
                pred_np = pred_flat.detach().cpu().numpy()
                is_music_pred = [int(p in music_id_set) for p in pred_np]
                music_preds_count = sum(is_music_pred)
                total_music_preds += music_preds_count

                # non-music predictions = masked positions - music_preds_count
                total_non_music_preds += (pred_flat.numel() - music_preds_count)

                # exact matches where label != -100
                # convert label_flat to cpu numpy and compare
                label_np = label_flat.detach().cpu().numpy()
                exact_matches = 0
                for p, l in zip(pred_np, label_np):
                    if int(l) == -100:
                        continue
                    if int(p) == int(l):
                        exact_matches += 1
                total_exact_matches += exact_matches

    model.train()

    # compute metrics (safeguard division)
    mask_pos_total = total_mask_positions if total_mask_positions > 0 else 1
    music_rate = total_music_preds / mask_pos_total
    non_music_rate = total_non_music_preds / mask_pos_total
    exact_match_rate = total_exact_matches / mask_pos_total

    return {
        "val_loss": total_loss / max(1, total_steps),
        "val_ce_loss": total_ce_loss / max(1, total_steps),
        "val_penalty_loss": total_penalty_loss / max(1, total_steps),
        "non_music_predictions": total_non_music_preds,
        "total_mask_positions": total_mask_positions,
        "music_rate": music_rate,               # fraction of mask positions predicted as music tokens
        "non_music_rate": non_music_rate,
        "exact_match_rate": exact_match_rate,   # fraction where pred == label at mask pos
        "error_rate": non_music_rate
    }




############################################################
# ENTRY POINT
############################################################
def train_lora(model, tokenizer):
    """Main training function"""
    model.train()

    # Prepare dataset - now returns separate input and target datasets
    dataset = prepare_dataset(tokenizer)
    
    # Verify we have all expected datasets
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    metrics_history = train_loop(model, tokenizer, dataset)
    
    return metrics_history