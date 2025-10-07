import torch
from math import cos, pi
from training_loop import train_one_epoch, validate
import os

# Helper lr_lambda function for the scheudler
# --------------------------------------------
def make_lr_lambda(step):
    def lr_lambda(warmup_steps, train_len, epochs):
        """
        This defines how the learning rate changes as the number of training steps increases.
        
        The function returns a multiplier (not the actual learning rate).
        PyTorch will multiply this value by the initial learning rate that 
        you passed to the optimizer.
        
        This means early in training you learn fast, 
        and later, training slows down smoothly to fine-tune weights.
        """
        
        # 1. warmup phase 
        # (lr gradually increases linearly)
        # from 0 → 1 × learning_rate
        # ----------------------------------
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        
        # 2. cosine decay phase 
        # (lr gradually decreases following a cosine curve down to zero.)
        # When progress = 0 → cos(0) = 1 → return = 1 → LR = full learning rate
        # When progress = 1 → cos(π) = -1 → return = 0 → LR = 0
        # ----------------------------------------------------------------
        else:
            progress = (step - warmup_steps) / (train_len * epochs - warmup_steps)
            return 0.5 * (1 + cos(pi * progress))
        
    return lr_lambda


# Main Training and evaluation function
# --------------------------------------
def train_model(
    model:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    learning_rate:float = 5e-5,
    epochs:int = 10,
    check_point_dir:str = "checkpoints/",
    load_pretrained:str = None,
    ):
    
    # 0. init device and put model in it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 1. init losses, and warmup steps
    total_train_losses = []
    total_val_losses = []
    warmup_steps = 2000
    
    # 2. init loss function, optimizer, and scheduler
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 0) # pad_id token
    optim = torch.optim.AdamW(
                            params=model.parameters(),
                            lr = learning_rate,
                            weight_decay= 0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, 
                                                lr_lambda=
                                                make_lr_lambda(warmup_steps,
                                                                len(train_dataloader),
                                                                epochs))
    
    # 3. load pretrained model if available
    if not load_pretrained:
        # see if it is available
        if os.path.exists(load_pretrained):
            # load checkpoint
            checkpoint = torch.load(load_pretrained, map_location="cpu")
            # load model
            model.load_state_dict(checkpoint["model_state_dict"])
            # load optim and scheduler
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            print(f"Loaded pretrained model (val_loss={checkpoint['val_loss']:.4f})")
        else:
            print ("[ERROR] Failed to load model. Starting a new save...")
            
    # 4. Full training loop
    for epoch in range(epochs):
        print(f"Training epoch no.{epoch+1} / {epochs}")
        print("-"*35)
        
        # 5. train
        train_loss = train_one_epoch(model,
                                    train_dataloader,
                                    loss_fn,
                                    optim,
                                    scheduler,
                                    device)
        total_train_losses.append(train_loss)
        
        # 6. validate
        print("Validating....")
        val_loss = validate(model,
                            val_dataloader,
                            loss_fn,
                            device)
        total_val_losses.append(val_loss)
        
        # 7. save checkpoints of the model
        torch.save({
            "trained_epochs" : epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss
        }, os.path.join(check_point_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # 8. print epoch summary
        print(f"Epoch no.{epoch+1} / {epochs} summary")
        print("-"*35)
        print(f"Average train losses = {train_loss}")
        print(f"Average validation losses = {val_loss}")
        print("="*35)
    
    # 9. return losses lits
    return total_train_losses, total_val_losses