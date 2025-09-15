# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from notes_utils import validate_token_sequence


# -------------------------------------------------------
# Training Function
# -------------------------------------------------------
def train(model,
        data_loader,
        tokenizer,
        epochs: int = 5,
        save_dir: str = "checkpoints/",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000) -> None:
    """
    Train a MusicTransformer model with improved training strategy.

    Args:
        model (nn.Module): The transformer model to train.
        data_loader (DataLoader): DataLoader providing (x, y) token sequences.
        epochs (int, optional): Number of epochs to train. Default = 5.
        save_dir (str, optional): Directory to save checkpoints. Default = "checkpoints/".
        learning_rate (float): Learning rate for optimizer.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (len(data_loader) * epochs - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function with label smoothing to reduce overconfidence
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    global_step = 0
    best_loss = float('inf')

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(enumerate(data_loader),
                            desc=f"Epoch {epoch+1}/{epochs}",
                            total=len(data_loader),
                            leave=True)

        for step, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)
            
            # Validate sequence structure (basic check)
            if step == 0 and epoch == 0:
                print("Sample input sequence (first 20 tokens):")
                sample_tokens = x[0, :20].cpu().numpy()
                print(sample_tokens)
                
                # Add this validation check:
                results = validate_token_sequence(tokenizer, sample_tokens)
                print(f"Training data structure validation: {results['accuracy']:.2%}")
                if results['invalid_transitions'] > 0:
                    print(f"Found {results['invalid_transitions']} invalid transitions in sample")

            # Forward pass
            logits = model(x[:, :-1])
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = y[:, 1:].reshape(-1)
            
            loss = loss_fn(logits_flat, targets_flat)

            # Backward pass with gradient accumulation for larger effective batch size
            loss = loss / 2  # Simulate 2x gradient accumulation
            loss.backward()
            
            if (step + 1) % 2 == 0:  # Update every 2 steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * 2  # Account for scaling
            avg_loss = total_loss / (step + 1)
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "step": f"{step+1}/{len(data_loader)}",
                "loss": f"{loss.item() * 2:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            global_step += 1

        print(f"Epoch {epoch+1} finished | Average Loss: {avg_loss:.4f}")

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    print(f"Training completed! Best loss: {best_loss:.4f}")
