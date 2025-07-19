"""
Enhanced AI Models for Screenshot Verification
Includes Vision Transformer, Dual-Stream Networks, and Adversarial Training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VisionTransformerDetector(nn.Module):
    """Vision Transformer for Screenshot Verification"""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for block in self.blocks:
            block._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits


class TransformerBlock(nn.Module):
    """Transformer Block with Multi-Head Self-Attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
    def _init_weights(self):
        """Initialize block weights"""
        self.attn._init_weights()
        self.mlp._init_weights()


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x
    
    def _init_weights(self):
        """Initialize attention weights"""
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, in_features: int, hidden_features: int, dropout: float):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def _init_weights(self):
        """Initialize MLP weights"""
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)


class DualStreamNetwork(nn.Module):
    """Dual-Stream Network (RGB + Frequency Domain)"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # RGB stream (EfficientNet)
        self.rgb_stream = models.efficientnet_b0(pretrained=True)
        rgb_features = self.rgb_stream.classifier.in_features
        self.rgb_stream.classifier = nn.Identity()
        
        # Frequency stream
        self.freq_stream = FrequencyStream()
        freq_features = 512
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(rgb_features + freq_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=rgb_features + freq_features, 
                                             num_heads=8, dropout=0.1)
    
    def forward(self, rgb_image: torch.Tensor, freq_image: torch.Tensor) -> torch.Tensor:
        # RGB stream
        rgb_features = self.rgb_stream(rgb_image)
        
        # Frequency stream
        freq_features = self.freq_stream(freq_image)
        
        # Concatenate features
        combined_features = torch.cat([rgb_features, freq_features], dim=1)
        
        # Apply attention
        combined_features = combined_features.unsqueeze(0)  # Add sequence dimension
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        attended_features = attended_features.squeeze(0)
        
        # Classification
        logits = self.fusion(attended_features)
        
        return logits


class FrequencyStream(nn.Module):
    """Frequency Domain Analysis Stream"""
    
    def __init__(self):
        super().__init__()
        
        # FFT-based feature extraction
        self.fft_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to frequency domain
        freq_x = self._to_frequency_domain(x)
        
        # Extract features
        features = self.fft_conv(freq_x)
        features = features.flatten(1)
        features = self.projection(features)
        
        return features
    
    def _to_frequency_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to frequency domain"""
        # FFT for each channel
        freq_domain = torch.fft.fft2(x, dim=(-2, -1))
        
        # Get magnitude spectrum
        magnitude = torch.abs(freq_domain)
        
        # Log scale for better visualization
        magnitude = torch.log(magnitude + 1)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude


class AdversarialTrainingDetector(nn.Module):
    """Adversarial Training Enhanced Detector"""
    
    def __init__(self, base_model: nn.Module, epsilon: float = 0.03):
        super().__init__()
        
        self.base_model = base_model
        self.epsilon = epsilon
        
        # Adversarial defense layers
        self.defense_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            # Add adversarial noise during training
            x = self._add_adversarial_noise(x)
        
        # Apply defense layers
        x = x + 0.1 * self.defense_layers(x)
        
        # Base model prediction
        return self.base_model(x)
    
    def _add_adversarial_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add adversarial noise for training"""
        x.requires_grad_(True)
        
        # Forward pass
        output = self.base_model(x)
        
        # Create adversarial target (flip labels)
        target = 1 - torch.argmax(output, dim=1)
        
        # Calculate loss
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial noise
        noise = self.epsilon * torch.sign(x.grad)
        
        # Add noise
        x_adv = x + noise
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()


class ModelEnsemble:
    """Ensemble of Multiple Models"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        assert len(self.models) == len(self.weights), "Models and weights must have same length"
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble prediction with confidence scores"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Stack predictions
        pred_stack = torch.stack(predictions, dim=0)
        
        # Calculate mean and variance
        mean_pred = torch.mean(pred_stack, dim=0)
        var_pred = torch.var(pred_stack, dim=0)
        
        # Confidence is inverse of variance
        confidence = 1.0 / (1.0 + var_pred.sum(dim=1, keepdim=True))
        
        return mean_pred, confidence


# Model factory functions
def create_vision_transformer(num_classes: int = 2) -> VisionTransformerDetector:
    """Create Vision Transformer model"""
    return VisionTransformerDetector(num_classes=num_classes)


def create_dual_stream_network(num_classes: int = 2) -> DualStreamNetwork:
    """Create Dual-Stream Network"""
    return DualStreamNetwork(num_classes=num_classes)


def create_adversarial_detector(base_model: nn.Module, epsilon: float = 0.03) -> AdversarialTrainingDetector:
    """Create Adversarial Training Detector"""
    return AdversarialTrainingDetector(base_model, epsilon)


def create_model_ensemble(model_configs: List[Dict]) -> ModelEnsemble:
    """Create model ensemble from configurations"""
    models = []
    
    for config in model_configs:
        model_type = config['type']
        
        if model_type == 'vision_transformer':
            model = create_vision_transformer(config.get('num_classes', 2))
        elif model_type == 'dual_stream':
            model = create_dual_stream_network(config.get('num_classes', 2))
        elif model_type == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, config.get('num_classes', 2))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models.append(model)
    
    weights = [config.get('weight', 1.0) for config in model_configs]
    
    return ModelEnsemble(models, weights)