"""
Deep learning models for screenshot authenticity detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ScreenshotDataset(Dataset):
    """Dataset class for screenshot authenticity training"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class AttentionModule(nn.Module):
    """Attention mechanism for feature enhancement"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionModule, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        channel_weight = self.channel_attention(x)
        x = x * channel_weight
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        x = x * spatial_weight
        
        return x


class EfficientNetDetector(nn.Module):
    """EfficientNet-based screenshot authenticity detector"""
    
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetDetector, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone.forward_features(dummy_input)
            feature_dim = features.shape[1]
        
        # Add attention mechanism
        self.attention = AttentionModule(feature_dim)
        
        # Replace classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Additional feature extractors for different scales
        self.multi_scale_features = nn.ModuleList([
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.AdaptiveAvgPool2d((28, 28))
        ])
        
        self.scale_fusion = nn.Conv2d(feature_dim * 3, feature_dim, 1)
    
    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Multi-scale feature extraction
        scale_features = []
        for pool in self.multi_scale_features:
            scale_feat = pool(attended_features)
            scale_features.append(F.interpolate(scale_feat, size=features.shape[2:], mode='bilinear'))
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_features, dim=1)
        fused_features = self.scale_fusion(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output, attended_features


class VisionTransformerDetector(nn.Module):
    """Vision Transformer for screenshot authenticity detection"""
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', num_classes: int = 2, pretrained: bool = True):
        super(VisionTransformerDetector, self).__init__()
        
        # Load pre-trained Vision Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        # Patch-level attention for localization
        self.patch_attention = nn.MultiheadAttention(feature_dim, num_heads=8, dropout=0.1)
        
    def forward(self, x):
        # Extract patch features
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # Global classification
        output = self.classifier(features)
        
        return output, features


class DualStreamNetwork(nn.Module):
    """Dual-stream network for RGB and frequency domain analysis"""
    
    def __init__(self, num_classes: int = 2):
        super(DualStreamNetwork, self).__init__()
        
        # RGB stream (spatial domain)
        self.rgb_stream = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        rgb_feature_dim = self.rgb_stream.num_features
        
        # Frequency stream (frequency domain)
        self.freq_stream = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        freq_feature_dim = self.freq_stream.num_features
        
        # Feature fusion
        combined_dim = rgb_feature_dim + freq_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Cross-attention between streams
        self.cross_attention = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)
        
        self.rgb_proj = nn.Linear(rgb_feature_dim, 256)
        self.freq_proj = nn.Linear(freq_feature_dim, 256)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # RGB stream
        rgb_features = self.rgb_stream(x)
        
        # Convert to frequency domain
        freq_input = self._rgb_to_frequency(x)
        freq_features = self.freq_stream(freq_input)
        
        # Project features
        rgb_proj = self.rgb_proj(rgb_features)
        freq_proj = self.freq_proj(freq_features)
        
        # Cross-attention
        rgb_attended, _ = self.cross_attention(
            rgb_proj.unsqueeze(1), freq_proj.unsqueeze(1), freq_proj.unsqueeze(1)
        )
        freq_attended, _ = self.cross_attention(
            freq_proj.unsqueeze(1), rgb_proj.unsqueeze(1), rgb_proj.unsqueeze(1)
        )
        
        # Combine features
        combined_features = torch.cat([
            rgb_attended.squeeze(1), 
            freq_attended.squeeze(1)
        ], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        
        return output, (rgb_features, freq_features)
    
    def _rgb_to_frequency(self, x):
        """Convert RGB image to frequency domain representation"""
        batch_size, channels, height, width = x.shape
        
        # Convert to numpy for FFT processing
        x_np = x.detach().cpu().numpy()
        freq_images = []
        
        for i in range(batch_size):
            freq_channels = []
            for c in range(channels):
                # Apply 2D FFT
                f_transform = np.fft.fft2(x_np[i, c])
                f_shift = np.fft.fftshift(f_transform)
                
                # Convert to magnitude and phase
                magnitude = np.log(np.abs(f_shift) + 1)
                phase = np.angle(f_shift)
                
                # Normalize
                magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
                phase = (phase + np.pi) / (2 * np.pi)
                
                freq_channels.append(magnitude)
            
            freq_images.append(np.stack(freq_channels))
        
        freq_tensor = torch.FloatTensor(freq_images).to(x.device)
        return freq_tensor


class MultiScaleNetwork(nn.Module):
    """Multi-scale network for different granularity feature extraction"""
    
    def __init__(self, num_classes: int = 2):
        super(MultiScaleNetwork, self).__init__()
        
        # Different scale backbones
        self.scales = [224, 112, 56]
        self.backbones = nn.ModuleList([
            timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            for _ in self.scales
        ])
        
        # Feature dimensions
        feature_dims = [backbone.num_features for backbone in self.backbones]
        total_dim = sum(feature_dims)
        
        # Scale-specific attention
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid()
            ) for dim in feature_dims
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        scale_features = []
        
        for i, (scale, backbone, attention) in enumerate(zip(self.scales, self.backbones, self.scale_attentions)):
            # Resize input to different scales
            if scale != x.shape[-1]:
                scale_input = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                scale_input = x
            
            # Extract features
            features = backbone(scale_input)
            
            # Apply attention
            att_weights = attention(features)
            attended_features = features * att_weights
            
            scale_features.append(attended_features)
        
        # Concatenate features
        combined_features = torch.cat(scale_features, dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        
        return output, scale_features


class AdversarialTrainingMixin:
    """Mixin class for adversarial training capabilities"""
    
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.001, num_steps: int = 7):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate_adversarial_examples(self, x, y, model, loss_fn):
        """Generate adversarial examples using PGD"""
        x_adv = x.clone().detach()
        
        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs, _ = model(x_adv)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.alpha * grad_sign
                
                # Project to epsilon ball
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
                
                x_adv.grad = None
        
        return x_adv


class EnsembleDetector(nn.Module):
    """Ensemble of multiple models for robust detection"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleDetector, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        # Ensure weights sum to 1
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def forward(self, x):
        outputs = []
        all_features = []
        
        for model, weight in zip(self.models, self.weights):
            output, features = model(x)
            outputs.append(output * weight)
            all_features.append(features)
        
        # Weighted average of outputs
        ensemble_output = torch.stack(outputs).sum(dim=0)
        
        return ensemble_output, all_features


class DeepLearningDetector:
    """Main deep learning detector class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', True) else "cpu")
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize all model architectures"""
        models = {}
        
        # EfficientNet detector
        models['efficientnet'] = EfficientNetDetector(
            model_name=self.config.get('efficientnet_variant', 'efficientnet_b0'),
            num_classes=2
        ).to(self.device)
        
        # Vision Transformer detector
        models['vit'] = VisionTransformerDetector(
            model_name=self.config.get('vision_transformer_model', 'vit_base_patch16_224'),
            num_classes=2
        ).to(self.device)
        
        # Dual-stream network
        models['dual_stream'] = DualStreamNetwork(num_classes=2).to(self.device)
        
        # Multi-scale network
        models['multi_scale'] = MultiScaleNetwork(num_classes=2).to(self.device)
        
        # Ensemble of all models
        model_list = list(models.values())
        models['ensemble'] = EnsembleDetector(model_list).to(self.device)
        
        return models
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            if image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image: np.ndarray, model_name: str = 'ensemble') -> Dict:
        """
        Predict screenshot authenticity
        
        Args:
            image: Input image array
            model_name: Name of model to use for prediction
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            model = self.models[model_name]
            model.eval()
            
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Prediction
            with torch.no_grad():
                outputs, features = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get prediction
                _, predicted = torch.max(outputs, 1)
                confidence = torch.max(probabilities, 1)[0]
                
                fake_probability = probabilities[0, 1].item()  # Probability of being fake
                authentic_probability = probabilities[0, 0].item()  # Probability of being authentic
            
            result = {
                'authentic': predicted.item() == 0,
                'confidence': confidence.item(),
                'fake_probability': fake_probability,
                'authentic_probability': authentic_probability,
                'model_used': model_name,
                'raw_outputs': outputs.cpu().numpy().tolist(),
                'features_extracted': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in deep learning prediction: {e}")
            return {
                'error': str(e),
                'authentic': True,  # Default to authentic on error
                'confidence': 0.0,
                'model_used': model_name
            }
    
    def load_model(self, model_path: str, model_name: str):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.models[model_name].load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model {model_name} from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    def save_model(self, model_path: str, model_name: str, optimizer=None, epoch=None):
        """Save model weights"""
        try:
            checkpoint = {
                'model_state_dict': self.models[model_name].state_dict(),
                'model_config': self.config,
            }
            
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if epoch:
                checkpoint['epoch'] = epoch
            
            torch.save(checkpoint, model_path)
            logger.info(f"Saved model {model_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def get_model_summary(self, model_name: str) -> Dict:
        """Get model architecture summary"""
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_name': model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': str(model.__class__.__name__)
        }