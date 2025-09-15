# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import os
import torch
import torch.nn as nn
from tqdm import tqdm


# -------------------------------------------------------
# Training Function
# -------------------------------------------------------
def train(model,
        data_loader,
        epochs: int = 5,
        save_dir: str = "checkpoints/") -> None:
    """
    Train a MusicTransformer model on a given dataset.

    Args:
        model (nn.Module): The transformer model to train.
        data_loader (DataLoader): DataLoader providing (x, y) token sequences.
        epochs (int, optional): Number of epochs to train. Default = 5.
        save_dir (str, optional): Directory to save checkpoints. Default = "checkpoints/".

    Saves:
        - Checkpoints for each epoch inside `save_dir` as:
            checkpoint_epoch{epoch}.pt
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer & loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

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

            # Forward pass (shifted targets)
            logits = model(x[:, :-1])
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y[:, 1:].reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track & update
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)

            progress_bar.set_postfix({
                "step": f"{step+1}/{len(data_loader)}",
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}"
            })

        print(f"Epoch {epoch+1} finished | Average Loss: {avg_loss:.4f}")

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)
