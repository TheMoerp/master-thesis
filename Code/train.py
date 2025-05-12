import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from monai.utils import set_determinism

from data_preparation import prepare_ribfrac_dataset
from model import AutoEncoder3D, AnomalyDetector

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-5,
    checkpoint_dir="checkpoints",
):
    """
    Train the 3D autoencoder model.
    
    Args:
        model: The autoencoder model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            images = batch["image"].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructions = model(images)
            
            # Calculate loss
            loss = criterion(reconstructions, images)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                
                # Forward pass
                reconstructions = model(images)
                
                # Calculate loss
                loss = criterion(reconstructions, images)
                
                # Update statistics
                val_loss += loss.item()
                val_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        history["val_loss"].append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print("Saved best model checkpoint.")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history["train_loss"][-1],
        'val_loss': history["val_loss"][-1],
    }, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    return model, history

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def main(args):
    # Set deterministic behavior for reproducibility
    set_determinism(seed=args.seed)
    
    # For now, use CPU as Conv3D is not supported on MPS
    device = torch.device("cpu")
    print("Using CPU (Conv3D operations are not yet supported on MPS)")
    
    # Prepare datasets and data loaders
    print("Preparing dataset...")
    train_loader, val_loader, test_loader = prepare_ribfrac_dataset(
        data_dir="datasets",
        batch_size=args.batch_size,
        cache_rate=0.0
    )
    
    # Create model
    print("Creating model...")
    model = AutoEncoder3D(
        in_channels=1,
        out_channels=1,
        features=[16, 32, 64, 128, 256]
    )
    
    # Train model
    print("Training model...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(args.checkpoint_dir, "training_history.png")
    )
    
    # Create and calibrate anomaly detector
    print("Calibrating anomaly detector...")
    detector = AnomalyDetector(
        model=trained_model,
        threshold_percentile=args.threshold_percentile
    )
    
    threshold = detector.calibrate(val_loader, device=device)
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # Save anomaly detector threshold
    np.save(os.path.join(args.checkpoint_dir, "anomaly_threshold.npy"), threshold)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Autoencoder for RibFrac")
    parser.add_argument("--data_dir", type=str, default="datasets",
                      help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                      help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true",
                      help="Disable CUDA training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                      help="Directory to save checkpoints")
    parser.add_argument("--threshold_percentile", type=float, default=95, help="Percentile for anomaly threshold")
    
    args = parser.parse_args()
    main(args) 