#!/usr/bin/env python3
"""
Model Evaluation and Analysis Script
Comprehensive evaluation with detailed metrics, visualizations, and analysis
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
import pandas as pd
import cv2
from PIL import Image
import grad_cam
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model_enhancements import (
    create_vision_transformer, create_dual_stream_network,
    create_adversarial_detector, create_model_ensemble
)
from scripts.train_models import ScreenshotDataset, get_transforms
from config.settings import settings


class ModelEvaluator:
    """Comprehensive Model Evaluation Class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Setup data
        self.test_loader = self._setup_data()
        
        # Results storage
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.features = []
        
        logging.info(f"Model evaluator initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_model(self) -> nn.Module:
        """Load trained model"""
        checkpoint_path = self.config['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model_config = checkpoint['config']
        model_type = model_config['model_type']
        
        if model_type == 'vision_transformer':
            model = create_vision_transformer(num_classes=model_config.get('num_classes', 2))
        elif model_type == 'dual_stream':
            model = create_dual_stream_network(num_classes=model_config.get('num_classes', 2))
        elif model_type == 'efficientnet':
            from torchvision import models
            model = models.efficientnet_b0(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, model_config.get('num_classes', 2))
        elif model_type == 'ensemble':
            model_configs = model_config['ensemble_configs']
            model = create_model_ensemble(model_configs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logging.info(f"Model loaded from {checkpoint_path}")
        logging.info(f"Best accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
        
        return model
    
    def _setup_data(self) -> DataLoader:
        """Setup test data loader"""
        data_dir = self.config['data_dir']
        transform = get_transforms(img_size=self.config.get('img_size', 224), is_training=False)
        
        # Create test dataset
        test_dataset = ScreenshotDataset(data_dir, transform=transform, is_training=False)
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        logging.info(f"Test dataset loaded: {len(test_dataset)} samples")
        return test_loader
    
    def evaluate(self) -> Dict[str, float]:
        """Run comprehensive evaluation"""
        logging.info("Starting model evaluation...")
        
        # Run inference
        self._run_inference()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Model interpretability
        if self.config.get('interpretability', {}).get('enabled', False):
            self._model_interpretability()
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _run_inference(self):
        """Run inference on test set"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.features = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'dual_stream' in str(type(self.model)):
                    # For dual-stream network
                    freq_data = self._to_frequency_domain(data)
                    output = self.model(data, freq_data)
                else:
                    output = self.model(data)
                
                # Get predictions
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                # Store results
                self.predictions.extend(preds.cpu().numpy())
                self.targets.extend(target.cpu().numpy())
                self.probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of fake
                
                # Store features for analysis
                if hasattr(self.model, 'features'):
                    self.features.extend(self.model.features.cpu().numpy())
                
                # Log progress
                if batch_idx % 50 == 0:
                    logging.info(f"Processed {batch_idx * data.size(0)}/{len(self.test_loader.dataset)} samples")
        
        logging.info("Inference completed")
    
    def _to_frequency_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to frequency domain"""
        freq_domain = torch.fft.fft2(x, dim=(-2, -1))
        magnitude = torch.abs(freq_domain)
        magnitude = torch.log(magnitude + 1)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return magnitude
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        # Basic metrics
        accuracy = accuracy_score(self.targets, self.predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='binary'
        )
        auc = roc_auc_score(self.targets, self.probabilities)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average=None
        )
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(self.targets, self.predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'precision_authentic': precision_per_class[1],
            'precision_fake': precision_per_class[0],
            'recall_authentic': recall_per_class[1],
            'recall_fake': recall_per_class[0],
            'f1_authentic': f1_per_class[1],
            'f1_fake': f1_per_class[0],
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        logging.info("Metrics calculated:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(output_dir)
        
        # 2. ROC Curve
        self._plot_roc_curve(output_dir)
        
        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve(output_dir)
        
        # 4. Prediction Distribution
        self._plot_prediction_distribution(output_dir)
        
        # 5. Feature Analysis (if available)
        if len(self.features) > 0:
            self._plot_feature_analysis(output_dir)
        
        # 6. Performance by Confidence
        self._plot_confidence_analysis(output_dir)
        
        logging.info(f"Visualizations saved to {output_dir}")
    
    def _plot_confusion_matrix(self, output_dir: Path):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.targets, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Authentic'],
                   yticklabels=['Fake', 'Authentic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, output_dir: Path):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.targets, self.probabilities)
        auc_score = roc_auc_score(self.targets, self.probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, output_dir: Path):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.targets, self.probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, output_dir: Path):
        """Plot prediction probability distribution"""
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Probability distribution
        plt.subplot(1, 2, 1)
        plt.hist([p for p, t in zip(self.probabilities, self.targets) if t == 0], 
                alpha=0.7, label='Fake', bins=30, density=True)
        plt.hist([p for p, t in zip(self.probabilities, self.targets) if t == 1], 
                alpha=0.7, label='Authentic', bins=30, density=True)
        plt.xlabel('Prediction Probability (Fake)')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Confidence vs Accuracy
        plt.subplot(1, 2, 2)
        confidence_bins = np.linspace(0, 1, 11)
        accuracies = []
        confidences = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (np.array(self.probabilities) >= confidence_bins[i]) & \
                   (np.array(self.probabilities) < confidence_bins[i + 1])
            if mask.sum() > 0:
                acc = accuracy_score(np.array(self.targets)[mask], np.array(self.predictions)[mask])
                accuracies.append(acc)
                confidences.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
        
        plt.plot(confidences, accuracies, 'o-', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, output_dir: Path):
        """Plot feature analysis using t-SNE"""
        if len(self.features) == 0:
            return
        
        # Reduce dimensionality
        features_array = np.array(self.features)
        if features_array.shape[1] > 50:
            # Use PCA first for high-dimensional features
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            features_array = pca.fit_transform(features_array)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_array)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=self.targets, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='True Label')
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_analysis(self, output_dir: Path):
        """Plot confidence analysis"""
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Confidence distribution by class
        plt.subplot(2, 2, 1)
        fake_confidences = [p for p, t in zip(self.probabilities, self.targets) if t == 0]
        authentic_confidences = [p for p, t in zip(self.probabilities, self.targets) if t == 1]
        
        plt.hist(fake_confidences, alpha=0.7, label='Fake', bins=20, density=True)
        plt.hist(authentic_confidences, alpha=0.7, label='Authentic', bins=20, density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution by Class')
        plt.legend()
        
        # Subplot 2: Error analysis
        plt.subplot(2, 2, 2)
        errors = np.array(self.predictions) != np.array(self.targets)
        error_confidences = np.array(self.probabilities)[errors]
        correct_confidences = np.array(self.probabilities)[~errors]
        
        plt.hist(error_confidences, alpha=0.7, label='Errors', bins=20, density=True)
        plt.hist(correct_confidences, alpha=0.7, label='Correct', bins=20, density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution by Prediction')
        plt.legend()
        
        # Subplot 3: Calibration plot
        plt.subplot(2, 2, 3)
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.targets, self.probabilities, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        # Subplot 4: Threshold analysis
        plt.subplot(2, 2, 4)
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        
        for threshold in thresholds:
            preds = (np.array(self.probabilities) > threshold).astype(int)
            f1 = f1_score(self.targets, preds)
            f1_scores.append(f1)
        
        plt.plot(thresholds, f1_scores)
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _model_interpretability(self):
        """Generate model interpretability visualizations"""
        if not self.config.get('interpretability', {}).get('enabled', False):
            return
        
        output_dir = Path(self.config['output_dir']) / 'interpretability'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get a few sample images
        sample_data, sample_targets = next(iter(self.test_loader))
        sample_data = sample_data[:5].to(self.device)  # Take first 5 samples
        
        # Grad-CAM visualization
        if hasattr(self.model, 'features'):
            self._grad_cam_visualization(sample_data, output_dir)
        
        # Feature importance
        self._feature_importance_analysis(sample_data, output_dir)
        
        logging.info(f"Interpretability visualizations saved to {output_dir}")
    
    def _grad_cam_visualization(self, sample_data: torch.Tensor, output_dir: Path):
        """Generate Grad-CAM visualizations"""
        # This is a simplified version - you might want to use a proper Grad-CAM implementation
        sample_data.requires_grad_(True)
        
        # Forward pass
        output = self.model(sample_data)
        
        # Backward pass
        output[:, 1].backward(torch.ones_like(output[:, 1]))
        
        # Get gradients
        gradients = sample_data.grad
        
        # Generate heatmap
        heatmap = torch.mean(gradients, dim=1)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        # Save heatmaps
        for i in range(sample_data.size(0)):
            plt.figure(figsize=(8, 6))
            plt.imshow(heatmap[i].cpu().numpy(), cmap='hot')
            plt.colorbar()
            plt.title(f'Grad-CAM Heatmap - Sample {i}')
            plt.tight_layout()
            plt.savefig(output_dir / f'gradcam_sample_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _feature_importance_analysis(self, sample_data: torch.Tensor, output_dir: Path):
        """Analyze feature importance"""
        # This is a placeholder - implement based on your model architecture
        pass
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save evaluation results"""
        output_dir = Path(self.config['output_dir'])
        
        # Save metrics
        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed classification report
        report = classification_report(
            self.targets, self.predictions,
            target_names=['Fake', 'Authentic'],
            output_dict=True
        )
        
        with open(output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save predictions
        results_df = pd.DataFrame({
            'target': self.targets,
            'prediction': self.predictions,
            'probability': self.probabilities
        })
        results_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        logging.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Screenshot Verification Models')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to evaluation config file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'checkpoint_path': args.checkpoint_path,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'log_dir': os.path.join(args.output_dir, 'logs'),
            'batch_size': 32,
            'num_workers': 4,
            'img_size': 224,
            'interpretability': {
                'enabled': True,
                'grad_cam': True,
                'attention_visualization': True,
                'feature_importance': True
            }
        }
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()