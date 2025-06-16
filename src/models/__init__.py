"""
Models package for EM Axon Classification

This package contains neural network architectures for multi-task learning
on electron microscopy axon cross-section images.
"""

# Import all model classes
try:
    from .cnn_models import (
        MultiTaskCNN,
        FixedMultiTaskCNN,  # New stable model
        SingleTaskCNN
    )
    
    # Make models available at package level
    __all__ = [
        'MultiTaskCNN',
        'FixedMultiTaskCNN', 
        'SingleTaskCNN'
    ]
    
    print("✅ Models imported successfully")
    
except ImportError as e:
    print(f"⚠️ Warning: Could not import some models: {e}")
    
    # Fallback imports
    try:
        from .cnn_models import MultiTaskCNN
        __all__ = ['MultiTaskCNN']
        print("✅ Basic MultiTaskCNN imported")
    except ImportError:
        print("❌ Critical: Could not import any models")
        __all__ = []

# Version info
__version__ = "1.0.0"

# Model factory function for easy model creation
def create_model(model_type='fixed', backbone='efficientnet_b1', **kwargs):
    """
    Factory function to create models easily
    
    Args:
        model_type (str): Type of model ('fixed', 'multi', 'single')
        backbone (str): Backbone architecture name
        **kwargs: Additional arguments for model constructor
    
    Returns:
        torch.nn.Module: Instantiated model
    """
    
    if model_type.lower() in ['fixed', 'stable']:
        if 'FixedMultiTaskCNN' in __all__:
            return FixedMultiTaskCNN(backbone_name=backbone, **kwargs)
        else:
            print("⚠️ FixedMultiTaskCNN not available, falling back to MultiTaskCNN")
            return MultiTaskCNN(backbone_name=backbone, **kwargs)
    
    elif model_type.lower() in ['multi', 'multitask']:
        return MultiTaskCNN(backbone_name=backbone, **kwargs)
    
    elif model_type.lower() in ['single', 'singletask']:
        if 'SingleTaskCNN' in __all__:
            return SingleTaskCNN(backbone_name=backbone, **kwargs)
        else:
            raise ValueError("SingleTaskCNN not available")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {['fixed', 'multi', 'single']}")

# Quick model info function
def get_model_info(model_type='fixed'):
    """Get information about available models"""
    
    info = {
        'fixed': {
            'class': 'FixedMultiTaskCNN',
            'description': 'Stable multi-task CNN with proper initialization',
            'tasks': ['pathology', 'region', 'depth'],
            'recommended': True
        },
        'multi': {
            'class': 'MultiTaskCNN', 
            'description': 'Original multi-task CNN with complex heads',
            'tasks': ['pathology', 'region', 'depth'],
            'recommended': False
        },
        'single': {
            'class': 'SingleTaskCNN',
            'description': 'Single-task CNN for individual task training',
            'tasks': ['configurable'],
            'recommended': False
        }
    }
    
    if model_type.lower() in info:
        return info[model_type.lower()]
    else:
        return info

# Print package info when imported
def print_package_info():
    """Print information about the models package"""
    print("\n📦 EM Axon Models Package")
    print("=" * 30)
    print(f"Version: {__version__}")
    print(f"Available models: {__all__}")
    
    if 'FixedMultiTaskCNN' in __all__:
        print("✅ Recommended: Use FixedMultiTaskCNN for stable training")
    else:
        print("⚠️ FixedMultiTaskCNN not available - check cnn_models.py")
    
    print("\nUsage examples:")
    print("  from src.models import FixedMultiTaskCNN")
    print("  from src.models import create_model")
    print("  model = create_model('fixed', 'efficientnet_b1')")
    print("")

# Uncomment to show info on import
# print_package_info()