import csv
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from data_loading import H5Dataset, create_equal_dataloaders

def train_model(model,criterion,dataset_list,lr=1e-4,epochs=100,save_dir='training_logs'):

    train_loader,val_loader = create_equal_dataloaders([H5Dataset(d) for d in dataset_list],30)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    training_log = save_dir / "training_log.csv"
    with open(training_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss'])
        
    validation_log = save_dir / "validation_log.csv"
    with open(validation_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'val_loss'])

    # Freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Move model to device
    model = model.to(device)

    # Initialize optimizer and loss function
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    best_val_loss = float('inf')
    step = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            images, depths = batch
            images, depths = images.to(device), depths.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_depths = model(images)
            
            # Compute loss
            loss = criterion(pred_depths, 1/depths)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'train_loss': train_loss / train_batches})
            step += 1
            if step % 100 == 0:
                with open(training_log, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, train_loss / train_batches])
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, depths = batch
                images, depths = images.to(device), depths.to(device)
                
                # Forward pass
                pred_depths = model(images)
                
                # Compute loss
                loss = criterion(pred_depths, depths)
                
                # Update metrics
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Save losses to CSV
        with open(validation_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, avg_val_loss])
        
        # Print epoch results
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / 'best_depth_model.pth')
            print('Saved new best model!')
        