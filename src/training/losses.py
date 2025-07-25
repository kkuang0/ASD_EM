import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple


class MultiTaskLoss(nn.Module):
    """
    Consolidated multi-task loss function with multiple loss types and advanced features.
    
    Supports:
    - Weighted multi-task learning
    - Class balancing with weights
    - Focal loss for handling severe class imbalance
    - Label smoothing for regularization
    - Numerically stable computations
    - Flexible loss type selection per task
    """
    
    def __init__(self, 
                 task_weights: Dict[str, float],
                 loss_types: Optional[Dict[str, str]] = None,
                 class_weights: Optional[Dict[str, torch.Tensor]] = None,
                 label_smoothing: float = 0.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 normalize_weights: bool = True):
        """
        Initialize the multi-task loss function.
        
        Args:
            task_weights: Dictionary mapping task names to their weights
            loss_types: Dictionary mapping task names to loss types 
                       ('ce', 'focal', 'smoothed_ce'). Defaults to 'ce' for all.
            class_weights: Dictionary mapping task names to class weight tensors
            label_smoothing: Label smoothing factor (0.0 to 1.0)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            normalize_weights: Whether to normalize task weights to sum to 1
        """
        super().__init__()
        
        # Validate inputs
        if not task_weights:
            raise ValueError("task_weights cannot be empty")
        
        if any(w <= 0 for w in task_weights.values()):
            raise ValueError("All task weights must be positive")
            
        if not (0.0 <= label_smoothing <= 1.0):
            raise ValueError("label_smoothing must be between 0.0 and 1.0")
        
        self.task_weights = task_weights.copy()
        if normalize_weights:
            total_weight = sum(self.task_weights.values())
            self.task_weights = {k: v / total_weight for k, v in self.task_weights.items()}
        
        # Set default loss types
        self.loss_types = loss_types or {task: 'ce' for task in task_weights.keys()}
        
        # Validate loss types
        valid_types = {'ce', 'focal', 'smoothed_ce'}
        for task, loss_type in self.loss_types.items():
            if loss_type not in valid_types:
                raise ValueError(f"Invalid loss type '{loss_type}' for task '{task}'. "
                               f"Must be one of {valid_types}")
        
        self.class_weights = class_weights or {}
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Register class weights as buffers for proper device handling
        for task, weights in self.class_weights.items():
            self.register_buffer(f'class_weights_{task}', weights)
    
    def _validate_inputs(self, outputs: Dict[str, torch.Tensor], 
                        labels: Dict[str, torch.Tensor]) -> None:
        """Validate input tensors for consistency."""
        # Check if all tasks are present
        missing_outputs = set(self.task_weights.keys()) - set(outputs.keys())
        missing_labels = set(self.task_weights.keys()) - set(labels.keys())
        
        if missing_outputs:
            raise ValueError(f"Missing outputs for tasks: {missing_outputs}")
        if missing_labels:
            raise ValueError(f"Missing labels for tasks: {missing_labels}")
        
        # Check batch size consistency
        batch_sizes = [out.size(0) for out in outputs.values()]
        label_batch_sizes = [lab.size(0) for lab in labels.values()]
        
        if len(set(batch_sizes + label_batch_sizes)) > 1:
            raise ValueError("Inconsistent batch sizes across tasks")
        
        # Check device consistency
        devices = [out.device for out in outputs.values()]
        if len(set(devices)) > 1:
            raise ValueError("All output tensors must be on the same device")
    
    def _compute_cross_entropy_loss(self, outputs: torch.Tensor, 
                                   labels: torch.Tensor,
                                   class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute standard cross-entropy loss with numerical stability."""
        # Add numerical stability by clamping extreme values
        outputs = torch.clamp(outputs, min=-100, max=100)
        loss = F.cross_entropy(outputs, labels, weight=class_weights, reduction='mean')
        
        # Check for NaN/Inf and return a safe fallback
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in cross-entropy loss, using fallback")
            return torch.tensor(0.693, device=outputs.device, requires_grad=True)  # ln(2) for binary classification
        
        return loss
    
    def _compute_focal_loss(self, outputs: torch.Tensor, 
                           labels: torch.Tensor,
                           class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute focal loss for handling class imbalance with numerical stability."""
        # Get device for tensor operations
        device = outputs.device
        
        # Add numerical stability by clamping extreme values
        outputs = torch.clamp(outputs, min=-100, max=100)
        
        # Compute cross entropy with no reduction
        ce_loss = F.cross_entropy(outputs, labels, weight=class_weights, reduction='none')
        
        # Compute p_t with clamping to prevent numerical issues
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-8, max=1.0)
        
        # Compute focal loss
        alpha = torch.tensor(self.focal_alpha, device=device, dtype=outputs.dtype)
        gamma = torch.tensor(self.focal_gamma, device=device, dtype=outputs.dtype)
        
        focal_loss = alpha * torch.pow(1 - pt, gamma) * ce_loss
        loss = focal_loss.mean()
        
        # Check for NaN/Inf and return a safe fallback
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in focal loss, using fallback")
            return torch.tensor(0.693, device=outputs.device, requires_grad=True)  # ln(2) for binary classification
        
        return loss
    
    def _compute_smoothed_cross_entropy_loss(self, outputs: torch.Tensor, 
                                            labels: torch.Tensor,
                                            class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss with numerical stability."""
        device = outputs.device
        num_classes = outputs.size(1)
        
        # Add numerical stability by clamping extreme values
        outputs = torch.clamp(outputs, min=-100, max=100)
        
        # Compute log probabilities
        log_probs = F.log_softmax(outputs, dim=1)
        
        # Compute negative log-likelihood for true labels
        nll_loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Compute smooth loss (uniform distribution component)
        smooth_loss = -log_probs.mean(dim=1)
        
        # Combine with label smoothing
        smoothing = torch.tensor(self.label_smoothing, device=device, dtype=outputs.dtype)
        task_loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        
        # Apply class weights if provided
        if class_weights is not None:
            weights = class_weights[labels]
            task_loss = task_loss * weights
        
        loss = task_loss.mean()
        
        # Check for NaN/Inf and return a safe fallback
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in smoothed cross-entropy loss, using fallback")
            return torch.tensor(0.693, device=outputs.device, requires_grad=True)  # ln(2) for binary classification
        
        return loss
    
    def _compute_task_loss(self, task: str, outputs: torch.Tensor, 
                          labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for a specific task based on its loss type."""
        # Get class weights for this task (handle device properly)
        class_weights = getattr(self, f'class_weights_{task}', None) if task in self.class_weights else None
        
        loss_type = self.loss_types[task]
        
        if loss_type == 'ce':
            return self._compute_cross_entropy_loss(outputs, labels, class_weights)
        elif loss_type == 'focal':
            return self._compute_focal_loss(outputs, labels, class_weights)
        elif loss_type == 'smoothed_ce':
            return self._compute_smoothed_cross_entropy_loss(outputs, labels, class_weights)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                labels: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the multi-task loss.
        
        Args:
            outputs: Dictionary mapping task names to output tensors [batch_size, num_classes]
            labels: Dictionary mapping task names to label tensors [batch_size]
            
        Returns:
            If return_task_losses is False: total weighted loss
            If return_task_losses is True: (total_loss, dict of individual task losses)
        """
        self._validate_inputs(outputs, labels)
        
        total_loss = 0.0
        task_losses = {}
        
        # Compute loss for each task
        for task in self.task_weights.keys():
            task_loss = self._compute_task_loss(task, outputs[task], labels[task])
            task_losses[task] = task_loss
            
            # Add weighted task loss to total
            total_loss = total_loss + self.task_weights[task] * task_loss
        
        return total_loss, task_losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        return self.task_weights.copy()
    
    def update_task_weights(self, new_weights: Dict[str, float], 
                           normalize: bool = True) -> None:
        """Update task weights dynamically."""
        if set(new_weights.keys()) != set(self.task_weights.keys()):
            raise ValueError("New weights must contain the same tasks")
        
        self.task_weights = new_weights.copy()
        if normalize:
            total_weight = sum(self.task_weights.values())
            self.task_weights = {k: v / total_weight for k, v in self.task_weights.items()}


# Convenience function for backward compatibility and simple use cases
def create_multitask_loss(task_weights: Dict[str, float],
                         loss_type: str = 'ce',
                         **kwargs) -> MultiTaskLoss:
    """
    Convenience function to create a multi-task loss with the same loss type for all tasks.
    
    Args:
        task_weights: Dictionary mapping task names to weights
        loss_type: Loss type to use for all tasks ('ce', 'focal', 'smoothed_ce')
        **kwargs: Additional arguments passed to MultiTaskLoss
        
    Returns:
        Configured MultiTaskLoss instance
    """
    loss_types = {task: loss_type for task in task_weights.keys()}
    return MultiTaskLoss(task_weights=task_weights, loss_types=loss_types, **kwargs)
