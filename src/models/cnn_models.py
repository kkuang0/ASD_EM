import torch
import torch.nn as nn
import torch.nn.init as init
import timm
from typing import Dict, Optional

class MultiTaskCNN(nn.Module):
    """Multi-task CNN supporting both TIMM and torchvision backbones"""
    
    def __init__(self, backbone_name: str, num_classes_dict: Dict[str, int], 
                 dropout_rate: float = 0.3, feature_dim: int = 512,
                 use_torchvision: bool = False):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes_dict = num_classes_dict
        self.task_names = list(num_classes_dict.keys())
        self.use_torchvision = use_torchvision
        
        # Create backbone
        if use_torchvision:
            self.backbone = self._create_torchvision_backbone(backbone_name)
            backbone_features = self._get_torchvision_features(backbone_name)
        else:
            # Original TIMM approach
            self.backbone = timm.create_model(
                backbone_name, 
                pretrained=True, 
                num_classes=0,  # Remove classification head
                global_pool='avg'
            )
            backbone_features = self.backbone.num_features
        
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific classification heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_dict.items():
            head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim // 2, num_classes)
            )
            # Initialize the classification head properly
            nn.init.xavier_uniform_(head[3].weight, gain=0.1)  # Small gain for stability
            nn.init.constant_(head[3].bias, 0.0)
            self.task_heads[task_name] = head
    
    def _create_torchvision_backbone(self, backbone_name: str):
        """Create backbone using torchvision models"""
        if backbone_name == 'efficientnet_b7':
            from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
            backbone = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            # Remove the classification head
            backbone.classifier = nn.Identity()
            return backbone
        elif backbone_name == 'efficientnet_b4':
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            backbone.classifier = nn.Identity()
            return backbone
        elif backbone_name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone.fc = nn.Identity()
            return backbone
        else:
            raise ValueError(f"Torchvision backbone '{backbone_name}' not supported. "
                           f"Use TIMM instead or add support for this backbone.")
    
    def _get_torchvision_features(self, backbone_name: str) -> int:
        """Get the number of features from torchvision backbones"""
        feature_dims = {
            'efficientnet_b7': 2560,
            'efficientnet_b4': 1792,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
        }
        
        if backbone_name not in feature_dims:
            raise ValueError(f"Feature dimensions for '{backbone_name}' not defined. "
                           f"Please add it to the feature_dims dictionary.")
        
        return feature_dims[backbone_name]
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Ensure features are properly flattened
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Project to common feature dimension
        projected_features = self.feature_proj(features)
        
        # Apply task-specific heads
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](projected_features)
        
        return outputs


# Also update the get_backbone_image_size function in transforms.py:

def get_backbone_image_size(backbone_name: str, use_torchvision: bool = False) -> int:
    """Get the recommended image size for different backbone architectures"""
    
    if use_torchvision:
        # Torchvision recommended sizes
        torchvision_sizes = {
            'efficientnet_b7': 600,  # As per torchvision docs
            'efficientnet_b4': 380,
            'resnet50': 224,
            'resnet101': 224,
            'resnet152': 224,
        }
        return torchvision_sizes.get(backbone_name, 224)
    else:
        # Original TIMM sizes
        timm_sizes = {
            'efficientnet_b0': 224,
            'efficientnet_b1': 240,
            'efficientnet_b2': 260,
            'efficientnet_b3': 300,
            'efficientnet_b4': 380,
            'efficientnet_b5': 456,
            'efficientnet_b6': 528,
            'efficientnet_b7': 600,
            'resnet50': 224,
            'resnet101': 224,
            'resnet152': 224,
            'vit_base_patch16_224': 224,
            'vit_large_patch16_224': 224,
        }
        return timm_sizes.get(backbone_name, 224)

class SingleTaskCNN(nn.Module):
    """Single-task CNN for comparison"""
    
    def __init__(self, 
                 backbone_name: str = 'efficientnet_b1',
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class FixedMultiTaskCNN(nn.Module):
    """Simplified and properly initialized multi-task CNN"""
    
    def __init__(self, backbone_name='efficientnet_b1'):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=True, 
            num_classes=0,
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 260, 260)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        print(f"  Backbone feature dimension: {feature_dim}")
        
        # Simple classification heads (no complex layers)
        self.heads = nn.ModuleDict({
            'pathology': nn.Linear(feature_dim, 2),
            'region': nn.Linear(feature_dim, 3), 
            'depth': nn.Linear(feature_dim, 2)
        })
        
        # Proper weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly to prevent explosions"""
        for head in self.heads.values():
            # Xavier uniform initialization
            nn.init.xavier_uniform_(head.weight, gain=1.0)
            nn.init.constant_(head.bias, 0.0)
        
        print("  ✅ Classification heads initialized with Xavier uniform")
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply classification heads
        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(features)
        
        return outputs
