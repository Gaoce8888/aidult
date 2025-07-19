#!/usr/bin/env python3
"""
AI Model Training Pipeline for Screenshot Verification
Supports multiple model types, data augmentation, and distributed training
"""
import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model_enhancements import (
    create_vision_transformer, create_dual_stream_network,
    create_adversarial_detector, create_model_ensemble
)
from config.settings import settings


class ScreenshotDataset(Dataset):
    """Custom Dataset for Screenshot Verification"""
    
    def __init__(self, data_dir: str, transform=None, is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Load data paths and labels
        self.samples = self._load_samples()
        
        logging.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and labels"""
        samples = []
        
        # Authentic screenshots (label 1)
        authentic_dir = self.data_dir / "authentic"
        if authentic_dir.exists():
            for img_path in authentic_dir.glob("*.png"):
                samples.append((str(img_path), 1))
        
        # Fake screenshots (label 0)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.png"):
                samples.append((str(img_path), 0))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        import cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_transforms(img_size: int = 224, is_training: bool = True) -> A.Compose:
    """Get data augmentation transforms"""
    if is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class ModelTrainer:
    """Model Training Class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=config['tensorboard_dir'])
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        logging.info(f"Model trainer initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_type = self.config['model_type']
        num_classes = self.config.get('num_classes', 2)
        
        if model_type == 'vision_transformer':
            model = create_vision_transformer(num_classes=num_classes)
        elif model_type == 'dual_stream':
            model = create_dual_stream_network(num_classes=num_classes)
        elif model_type == 'efficientnet':
            from torchvision import models
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_type == 'ensemble':
            model_configs = self.config['ensemble_configs']
            model = create_model_ensemble(model_configs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        num_epochs = self.config['num_epochs']
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 30)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10)
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, create_dual_stream_network.__class__):
                # For dual-stream network, we need both RGB and frequency domain
                freq_data = self._to_frequency_domain(data)
                output = self.model(data, freq_data)
            else:
                output = self.model(data)
            
            # Calculate loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                logging.info(f'Epoch {epoch}: [{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f}, '
                           f'Accuracy: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if isinstance(self.model, create_dual_stream_network.__class__):
                    freq_data = self._to_frequency_domain(data)
                    output = self.model(data, freq_data)
                else:
                    output = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Store predictions for metrics
                all_predictions.extend(output.softmax(dim=1)[:, 1].cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, [1 if p > 0.5 else 0 for p in all_predictions], average='binary'
        )
        auc = roc_auc_score(all_targets, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _to_frequency_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to frequency domain"""
        # FFT for each channel
        freq_domain = torch.fft.fft2(x, dim=(-2, -1))
        
        # Get magnitude spectrum
        magnitude = torch.abs(freq_domain)
        
        # Log scale
        magnitude = torch.log(magnitude + 1)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_accuracy': self.best_accuracy,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with accuracy: {metrics['accuracy']:.2f}%")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        num_epochs = self.config['num_epochs']
        
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            logging.info(f'Epoch {epoch}/{num_epochs} completed in {epoch_time:.2f}s')
            logging.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                        f'Accuracy: {train_metrics["accuracy"]:.2f}%')
            logging.info(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                        f'Accuracy: {val_metrics["accuracy"]:.2f}%, '
                        f'F1: {val_metrics["f1"]:.4f}, '
                        f'AUC: {val_metrics["auc"]:.4f}')
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            self.writer.add_scalar('AUC/Val', val_metrics['auc'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
                self.best_epoch = epoch
            
            if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
        
        logging.info(f"Training completed. Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Screenshot Verification Models')
    parser.add_argument('--config', type=str, required=True, help='Path to training config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='efficientnet', 
                       choices=['efficientnet', 'vision_transformer', 'dual_stream', 'ensemble'],
                       help='Model type to train')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update config with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'model_type': args.model_type,
        'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints'),
        'log_dir': os.path.join(args.output_dir, 'logs'),
        'tensorboard_dir': os.path.join(args.output_dir, 'tensorboard')
    })
    
    # Create output directories
    for dir_path in [config['checkpoint_dir'], config['log_dir'], config['tensorboard_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Setup data transforms
    train_transform = get_transforms(img_size=config.get('img_size', 224), is_training=True)
    val_transform = get_transforms(img_size=config.get('img_size', 224), is_training=False)
    
    # Create datasets
    train_dataset = ScreenshotDataset(args.data_dir, transform=train_transform, is_training=True)
    val_dataset = ScreenshotDataset(args.data_dir, transform=val_transform, is_training=False)
    
    # Split dataset if validation set doesn't exist
    if len(val_dataset) == 0:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()