import torch
from tqdm import tqdm

# Training one epoch Fucntion
# ----------------------------
def train_one_epoch(
            model:torch.nn.Module,
            train_dataloader:torch.utils.data.DataLoader,
            loss_fn:torch.nn.Module,
            optimizer:torch.optim,
            scheduler,
            device,
            ):
    
    # 0. put model in train mode
    model.train()
    
    # 1. Loop through train_dataloader 
    pbar = tqdm(enumerate(train_dataloader), 
                total= train_dataloader,
                desc=f"Training...")
    for step, (x, y) in pbar:
        
        # 2. put tensors to device
        x, y = x.to(device), y.to(device)
        
        # 3. forward pass
        logits = model(x)
        
        # 4. calculate the loss
        loss = loss_fn(logits, y)
        
        # 5. zero grad
        optimizer.zero_grad()
        
        # 6. loss backard
        loss.backward()
        
        # 7. optimizer and scheduler step
        optimizer.step()
        scheduler.step()
        
        # 8. update the progress bar
        pbar.set_postfix({
            "train_losses": f"{loss/ len(train_dataloader):.4f}",
            "lr": f"{scheduler.get_last:lr()[0]:.2e}"
        })
    
    # 9. return loss
    return loss/ len(train_dataloader)
    
    
# Validating one epoch Fucntion
# ------------------------------
def validate(
    model:torch.nn.Module,
    val_dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    device
):
    # 0. put model to eval mode
    model.eval()
    
    # 1. go to infernce mode
    with torch.inference_mode():
        
        # 2. Loop through val_dataloader
        pbar = tqdm( total= val_dataloader,
                    desc=f"Testing...")
        for x,y in pbar:
            
            # 3. put tensors to device
            x, y = x.to(device), y.to(device)
            
            # 4. forward pass
            logits = model(x)
            
            # 5. calculate the loss
            loss = loss_fn(logits, y)
            
            # 6. update the progress bar
            pbar.set_postfix({
                "val_losses": f"{loss/ len(val_dataloader):.4f}",
            })
            
        # 7. return loss
        return loss/len(val_dataloader)