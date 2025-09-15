# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from notes_utils import validate_token_sequence, build_vocab_maps

# -------------------------------------------------------
# Enhanced Training Function with Structural Awareness
# -------------------------------------------------------
def train(model,
        train_loader,
        val_loader,
        tokenizer,
        epochs: int = 5,
        save_dir: str = "checkpoints/",
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        structural_weight: float = 0.3) -> None:
    """
    Train a MusicTransformer model with structural awareness to enforce proper token ordering.
    
    Args:
        model (nn.Module): The transformer model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        tokenizer: Tokenizer for sequence validation.
        epochs (int, optional): Number of epochs to train. Default = 5.
        save_dir (str, optional): Directory to save checkpoints. Default = "checkpoints/".
        learning_rate (float): Learning rate for optimizer.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
        structural_weight (float): Weight for structural loss component (0-1).
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Build vocab maps for structural analysis
    _, id_to_token = build_vocab_maps(tokenizer)
    
    # Define expected token transitions
    expected_transitions = {
        "Bar": ["Position"],
        "Position": ["Pitch"],
        "Pitch": ["Velocity"], 
        "Velocity": ["Duration"],
        "Duration": ["Position", "Bar"]
    }

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
            progress = (step - warmup_steps) / (len(train_loader) * epochs - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    main_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    structural_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    best_loss = float('inf')
    
    # Precompute token type mappings
    token_type_map = {}
    for token_id, token_name in id_to_token.items():
        for token_type in expected_transitions:
            if token_name.startswith(f"{token_type}_"):
                token_type_map[token_id] = token_type
                break
        else:
            token_type_map[token_id] = None

    # Precompute valid next token masks for each token type
    valid_next_masks = {}
    for current_type, next_types in expected_transitions.items():
        mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        for token_id, token_name in id_to_token.items():
            if any(token_name.startswith(f"{t}_") for t in next_types):
                mask[token_id] = True
        valid_next_masks[current_type] = mask

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_struct_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}",
                            total=len(train_loader),
                            leave=True)

        for step, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x[:, :-1])
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = y[:, 1:].reshape(-1)
            
            # Main prediction loss
            main_loss = main_loss_fn(logits_flat, targets_flat)

            # Structural awareness loss - only apply where we have valid transitions
            struct_loss = torch.tensor(0.0, device=device)
            if structural_weight > 0:
                # Get current token types for the input sequence
                current_types = torch.zeros_like(x[:, :-1])
                for i in range(x[:, :-1].shape[0]):
                    for j in range(x[:, :-1].shape[1]):
                        current_types[i, j] = token_type_map.get(x[i, j].item(), -1)
                
                # Create structural mask (only positions where we have expected transitions)
                struct_mask = (current_types != -1) & (current_types != None)
                
                # Apply structural loss only to valid positions
                if struct_mask.sum() > 0:
                    # Get the valid next token masks for each position
                    valid_masks = torch.stack([valid_next_masks.get(current_types[i, j].item(), 
                                                torch.zeros(vocab_size, device=device, dtype=torch.bool))
                                            for i in range(struct_mask.shape[0])
                                            for j in range(struct_mask.shape[1]) if struct_mask[i, j]])
                    
                    # Get the logits for masked positions
                    masked_logits = logits_flat[struct_mask.view(-1)]
                    
                    # Create target: maximize probability of valid next tokens
                    struct_loss = -torch.log((masked_logits.softmax(dim=1) * valid_masks.float()).sum(dim=1) + 1e-10).mean()
                    
            
            # Combined loss
            loss = main_loss + structural_weight * struct_loss

            # Backward pass with gradient accumulation
            loss = loss / 2  # Simulate 2x gradient accumulation
            loss.backward()
            
            if (step + 1) % 2 == 0:  # Update every 2 steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * 2  # Account for scaling
            total_main_loss += main_loss.item()
            total_struct_loss += struct_loss.item() if structural_weight > 0 else 0
            avg_loss = total_loss / (step + 1)
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({
                "step": f"{step+1}/{len(train_loader)}",
                "loss": f"{loss.item() * 2:.4f}",
                "main": f"{main_loss.item():.4f}",
                "struct": f"{struct_loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            global_step += 1

        # Print epoch summary
        epoch_main_loss = total_main_loss / len(train_loader)
        epoch_struct_loss = total_struct_loss / len(train_loader) if structural_weight > 0 else 0
        print(f"Epoch {epoch+1} finished | Main Loss: {epoch_main_loss:.4f} | Struct Loss: {epoch_struct_loss:.4f} | Total: {avg_loss:.4f}")

        # ---- Validation and checkpoint ----
        # Run validation on the actual validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x[:, :-1])
                
                # Calculate validation loss
                batch_size, seq_len, vocab_size = val_logits.shape
                val_logits_flat = val_logits.reshape(-1, vocab_size)
                val_targets_flat = val_y[:, 1:].reshape(-1)
                val_loss += main_loss_fn(val_logits_flat, val_targets_flat).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
        model.train()
        
        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": avg_loss,
            "val_loss": avg_val_loss,
            "main_loss": epoch_main_loss,
            "struct_loss": epoch_struct_loss,
        }
        torch.save(checkpoint, ckpt_path)
        
        # Save best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {best_loss:.4f}")

    print(f"Training completed! Best validation loss: {best_loss:.4f}")