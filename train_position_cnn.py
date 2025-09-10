#!/usr/bin/env python3
"""
Train CNN model for position bar digit recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from pathlib import Path
import json
import random
from typing import Tuple, List
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.hybrid_position_detector import PositionBarCNN


class PositionDigitDataset(Dataset):
    """Dataset for position bar digits"""
    
    def __init__(self, data_dir: str, annotations_file: str, transform=None, train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Filter labeled samples only
        self.samples = [(ann['path'], ann['label']) 
                       for ann in annotations 
                       if ann['label'] is not None and ann['label'] > 0]
        
        if len(self.samples) == 0:
            logger.warning("No labeled samples found! Run collect_training_data.py first.")
            
        # Split into train/val
        if len(self.samples) > 0:
            train_samples, val_samples = train_test_split(
                self.samples, test_size=0.2, random_state=42, 
                stratify=[s[1] for s in self.samples] if len(self.samples) > 20 else None
            )
            
            self.samples = train_samples if train else val_samples
            
        logger.info(f"{'Train' if train else 'Val'} dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Return a blank image if file not found
            image = np.zeros((32, 32), dtype=np.uint8)
        
        # Ensure 32x32 size
        if image.shape != (32, 32):
            image = cv2.resize(image, (32, 32))
        
        # Convert to tensor
        image = torch.FloatTensor(image).unsqueeze(0) / 255.0
        
        # Apply transforms
        if self.transform and self.train:
            image = self.transform(image)
        
        # Label should be 0-indexed for CrossEntropyLoss
        label = label - 1  # Convert 1-20 to 0-19
        
        return image, label


class DataAugmentation:
    """Custom augmentation for digit images"""
    
    def __call__(self, img):
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, factor)
        
        # Random noise
        if random.random() > 0.5:
            noise = torch.randn_like(img) * 0.05
            img = torch.clamp(img + noise, 0, 1)
        
        # Random blur
        if random.random() > 0.5:
            img = TF.gaussian_blur(img, kernel_size=3)
        
        return img


class CNNTrainer:
    """Trainer for position bar CNN"""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = torch.device(device if device else 
                                 ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class statistics
                for label, pred in zip(labels, predicted):
                    label_val = label.item()
                    if label_val not in class_total:
                        class_total[label_val] = 0
                        class_correct[label_val] = 0
                    class_total[label_val] += 1
                    if label == pred:
                        class_correct[label_val] += 1
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        # Print per-class accuracy
        logger.info("Per-digit accuracy:")
        for digit in sorted(class_total.keys()):
            digit_acc = 100 * class_correct[digit] / class_total[digit]
            logger.info(f"  Digit {digit+1}: {digit_acc:.1f}% ({class_correct[digit]}/{class_total[digit]})")
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int = 50, lr: float = 0.001):
        """Full training loop"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("models/position_cnn_best.pth")
                logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        logger.info(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save final model
        self.save_model("models/position_cnn_final.pth")
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_model(self, path: str):
        """Save model weights"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_curves.png')
        logger.info("Training curves saved to models/training_curves.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN for position bar digits")
    parser.add_argument('--data-dir', type=str, default='data/position_digits',
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=20,
                       help='Number of classes (max horse number)')
    
    args = parser.parse_args()
    
    # Check if annotations exist
    annotations_file = Path(args.data_dir) / "annotations.json"
    if not annotations_file.exists():
        logger.error(f"No annotations found at {annotations_file}")
        logger.error("Run: python collect_training_data.py --video <video_file> --label")
        return
    
    # Create datasets
    transform = DataAugmentation()
    
    train_dataset = PositionDigitDataset(
        args.data_dir, annotations_file, transform=transform, train=True
    )
    
    val_dataset = PositionDigitDataset(
        args.data_dir, annotations_file, transform=None, train=False
    )
    
    if len(train_dataset) == 0:
        logger.error("No training samples found! Collect and label data first.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    # Create model
    model = PositionBarCNN(num_classes=args.num_classes)
    
    # Train
    trainer = CNNTrainer(model)
    trainer.train(train_loader, val_loader, num_epochs=args.epochs, lr=args.lr)
    
    logger.info("\nTraining complete!")
    logger.info("To use the trained model, update hybrid_position_detector.py to load:")
    logger.info("  models/position_cnn_best.pth")


if __name__ == "__main__":
    main()