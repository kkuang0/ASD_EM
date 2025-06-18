import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

from ..data import EMAxonDataset, get_em_transforms, get_backbone_image_size, create_splits
from ..models import MultiTaskCNN
from ..utils import (
    setup_logging,
    calculate_metrics,
    plot_sample_images,
    plot_roc_curves,
    plot_pr_curves,
)
from .evaluator import ModelEvaluator
from .losses import MultiTaskLoss  # Updated import


class DDPTrainer:
    """DDP-enabled trainer for multi-task CNN with consolidated improvements"""
    
    def __init__(self, config):
        self.config = config
        self.task_names = list(config.num_classes.keys())
        
        # Initialize distributed training
        if config.distributed:
            self._setup_distributed()

        # Set device (after process group init to ensure correct device assignment)
        if config.distributed:
            torch.cuda.set_device(config.local_rank)
            self.device = torch.device(f"cuda:{config.local_rank}")
        else:
            self.device = torch.device(config.device)
        
        # Get appropriate image size for backbone
        self.image_size = get_backbone_image_size(config.backbone)
        if not config.distributed or config.rank == 0:
            print(f"Using image size {self.image_size} for backbone {config.backbone}")
        
        # Create output directories (only on rank 0)
        if not config.distributed or config.rank == 0:
            os.makedirs(os.path.join(config.output_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
        
        self.class_weights = None
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(config)
        
        # Setup loss function configuration
        self._setup_loss_function()
        
        # Synchronize all processes
        if config.distributed:
            dist.barrier(device_ids=[self.device.index])
    
    def _setup_distributed(self):
        """Initialize distributed training"""
        # Ensure each process uses the correct GPU before initializing NCCL
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)

        dist.init_process_group(
            backend=self.config.dist_backend,
            init_method=self.config.dist_url,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        if self.config.rank == 0:
            print(f"Distributed training initialized:")
            print(f"  World size: {self.config.world_size}")
            print(f"  Backend: {self.config.dist_backend}")
    
    def _setup_loss_function(self):
        """Setup the consolidated multi-task loss function"""
        # Define loss types per task (can be customized)
        loss_types = getattr(self.config, 'loss_types', {
            task: 'ce' for task in self.task_names  # Default to cross-entropy
        })
        
        # If using focal loss for imbalanced tasks
        if hasattr(self.config, 'use_focal_loss') and self.config.use_focal_loss:
            loss_types = {task: 'focal' for task in self.task_names}
        
        # If using label smoothing
        if hasattr(self.config, 'use_label_smoothing') and self.config.use_label_smoothing:
            loss_types = {task: 'smoothed_ce' for task in self.task_names}
        
        self.loss_types = loss_types
    
    def create_model(self) -> nn.Module:
        """Create and initialize the model with DDP support"""
        model = MultiTaskCNN(
            backbone_name=self.config.backbone,
            num_classes_dict=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            feature_dim=self.config.feature_dim
        ).to(self.device)
        
        if self.config.distributed:
            # Convert BatchNorm to SyncBatchNorm for better DDP performance
            if getattr(self.config, 'sync_batchnorm', True):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            # Wrap with DDP
            model = DDP(
                model, 
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True  # Needed for multi-task models
            )
        
        return model
    
    def calculate_class_weights(self, train_dataset):
        """Calculate balanced class weights for each task"""
        class_weights = {}
        task_labels = {task: [] for task in self.task_names}
        
        # Collect all labels
        for _, labels in train_dataset:
            for task, label in labels.items():
                task_labels[task].append(label.item() if torch.is_tensor(label) else label)
        
        # Calculate weights for each task
        for task, labels in task_labels.items():
            counter = Counter(labels)
            total = len(labels)
            
            # Calculate inverse frequency weights
            weights = torch.zeros(len(counter))
            for class_id, count in counter.items():
                weights[class_id] = total / (len(counter) * count)
            
            class_weights[task] = weights.to(self.device)
            
            # Only print on rank 0
            if not self.config.distributed or self.config.rank == 0:
                print(f"{task.capitalize()} class distribution: {counter}")
                print(f"{task.capitalize()} weights: {weights}")
        
        return class_weights
    
    def create_dataloaders(self, train_df, val_df, test_df) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders with distributed sampling and updated transforms"""
        
        # Create datasets with updated transforms
        train_dataset = EMAxonDataset(
            train_df, 
            transform=get_em_transforms(image_size=self.image_size, is_training=True)
        )
        val_dataset = EMAxonDataset(
            val_df, 
            transform=get_em_transforms(image_size=self.image_size, is_training=False)
        )
        test_dataset = EMAxonDataset(
            test_df, 
            transform=get_em_transforms(image_size=self.image_size, is_training=False)
        )
        
        # Calculate class weights (only on rank 0, then broadcast)
        if not self.config.distributed or self.config.rank == 0:
            self.class_weights = self.calculate_class_weights(train_dataset)
        
        # Broadcast class weights to all processes
        if self.config.distributed:
            if self.config.rank == 0:
                # Serialize class weights for broadcasting
                class_weights_data = {task: weights.cpu() for task, weights in self.class_weights.items()}
            else:
                class_weights_data = None
            
            # Broadcast using pickle (simplified approach)
            class_weights_list = [class_weights_data]
            dist.broadcast_object_list(class_weights_list, src=0)
            
            if self.config.rank != 0:
                class_weights_data = class_weights_list[0]
                self.class_weights = {task: weights.to(self.device) for task, weights in class_weights_data.items()}
        
        # Create samplers for distributed training
        if self.config.distributed:
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True,
                seed=getattr(self.config, 'random_seed', 42)
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=False
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=False
            )
            shuffle = False  # Handled by sampler
        else:
            train_sampler = val_sampler = test_sampler = None
            shuffle = True
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=getattr(self.config, 'pin_memory', True),
            drop_last=True  # Important for DDP
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=getattr(self.config, 'pin_memory', True)
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=self.config.num_workers,
            pin_memory=getattr(self.config, 'pin_memory', True)
        )
        
        return train_loader, val_loader, test_loader
    
    def create_loss_function(self) -> MultiTaskLoss:
        """Create the consolidated multi-task loss function"""
        return MultiTaskLoss(
            task_weights=self.config.task_weights,
            loss_types=self.loss_types,
            class_weights=self.class_weights,
            label_smoothing=getattr(self.config, 'label_smoothing', 0.0),
            focal_alpha=getattr(self.config, 'focal_alpha', 1.0),
            focal_gamma=getattr(self.config, 'focal_gamma', 2.0),
            normalize_weights=getattr(self.config, 'normalize_task_weights', True)
        )
    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create a learning rate scheduler from configuration."""
        sched_type = getattr(self.config, "scheduler", "cosine").lower()

        if sched_type == "step":
            return StepLR(
                optimizer,
                step_size=getattr(self.config, "scheduler_step_size", 10),
                gamma=getattr(self.config, "scheduler_gamma", 0.1),
            )
        elif sched_type == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=getattr(self.config, "scheduler_factor", 0.1),
                patience=getattr(self.config, "scheduler_patience", 5),
            )
        elif sched_type == "none":
            return None
        else:
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=getattr(self.config, "scheduler_T0", 10),
                T_mult=getattr(self.config, "scheduler_T_mult", 2),
                eta_min=getattr(self.config, "scheduler_eta_min", 1e-6),
            )

    
    def calculate_loss(self, outputs: Dict[str, torch.Tensor], 
                      labels: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate loss using consolidated loss function"""
        if not hasattr(self, 'criterion'):
            self.criterion = self.create_loss_function()
        
        return self.criterion(outputs, labels)
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        writer: Optional[SummaryWriter] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with DDP support and gradient clipping"""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        correct = {t: 0 for t in self.task_names}
        total = {t: 0 for t in self.task_names}
        
        # Set epoch for distributed sampler
        if self.config.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
            
            optimizer.zero_grad()
            
            with autocast(enabled=getattr(self.config, 'mixed_precision', True)):
                outputs = model(images)
                loss, task_losses = self.calculate_loss(outputs, labels)

            if getattr(self.config, 'mixed_precision', True):
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    getattr(self.config, 'max_grad_norm', 1.0)
                )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    getattr(self.config, 'max_grad_norm', 1.0)
                )
                
                optimizer.step()

            total_loss += loss.item()
            # Accuracy accumulation
            for task in self.task_names:
                preds = torch.argmax(outputs[task], dim=1)
                correct[task] += (preds == labels[task]).sum().item()
                total[task] += labels[task].size(0)
            
            # Logging (only on rank 0)
            if (not self.config.distributed or self.config.rank == 0) and \
               batch_idx % getattr(self.config, 'log_every_n_batches', 50) == 0:
                step = epoch * num_batches + batch_idx
                if writer:
                    writer.add_scalar('Loss/Train_Batch', loss.item(), step)
                    writer.add_scalar('GradNorm/Train_Batch', grad_norm.item(), step)
                    for task, task_loss in task_losses.items():
                        writer.add_scalar(f'Loss/Train_{task}_Batch', task_loss.item(), step)
                    for task in self.task_names:
                        batch_acc = (torch.argmax(outputs[task], dim=1) == labels[task]).float().mean()
                        writer.add_scalar(f'Accuracy/Train_{task}_Batch', batch_acc.item(), step)
                
                print(f"Rank {self.config.rank if self.config.distributed else 0} - "
                      f"Epoch {epoch+1}/{self.config.epochs} - "
                      f"Batch {batch_idx}/{num_batches} - "
                      f"Loss: {loss.item():.4f} - "
                      f"GradNorm: {grad_norm.item():.4f}")
        
        train_acc = {t: (correct[t] / total[t]) if total[t] > 0 else 0.0 for t in self.task_names}

        if self.config.distributed:
            for t in self.task_names:
                tensor = torch.tensor([correct[t], total[t]], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                train_acc[t] = tensor[0].item() / tensor[1].item() if tensor[1].item() > 0 else 0.0

        return total_loss / num_batches, train_acc
    
    def _reduce_dict(self, input_dict: Dict[str, float]) -> Dict[str, float]:
        """Reduce dictionary values across all processes"""
        if not self.config.distributed:
            return input_dict
        
        reduced_dict = {}
        for key, value in input_dict.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced_dict[key] = tensor.item() / self.config.world_size
        
        return reduced_dict
    
    def _broadcast_dataframes(self, folds, test_df):
        """Broadcast DataFrame data across processes"""
        if not self.config.distributed:
            return folds, test_df
        
        if self.config.rank == 0:
            # Serialize data for broadcasting
            data_to_broadcast = {
                'folds': [(train_df.to_dict(), val_df.to_dict()) for train_df, val_df in folds],
                'test_df': test_df.to_dict()
            }
        else:
            data_to_broadcast = None
        
        # Broadcast the data
        data_list = [data_to_broadcast]
        dist.broadcast_object_list(data_list, src=0)
        
        if self.config.rank != 0:
            import pandas as pd
            data_to_broadcast = data_list[0]
            
            # Reconstruct DataFrames
            folds = [(pd.DataFrame.from_dict(train_dict), pd.DataFrame.from_dict(val_dict)) 
                    for train_dict, val_dict in data_to_broadcast['folds']]
            test_df = pd.DataFrame.from_dict(data_to_broadcast['test_df'])
        
        return folds, test_df
    
    def train_fold(self, fold_idx: int, train_df, val_df, test_df) -> Dict[str, float]:
        """Train a single fold with DDP support"""
        if not self.config.distributed or self.config.rank == 0:
            print(f"\n{'='*50}")
            print(f"Training Fold {fold_idx + 1}")
            print(f"{'='*50}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_dataloaders(train_df, val_df, test_df)
        
        # Create model
        model = self.create_model()
        
        # Improved optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.lr, 
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = self._create_scheduler(optimizer)
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=getattr(self.config, 'mixed_precision', True))
        
        # Logging (only on rank 0)
        writer = None
        if not self.config.distributed or self.config.rank == 0:
            log_dir = os.path.join(
                self.config.output_dir, 'logs',
                f"fold_{fold_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            writer = SummaryWriter(log_dir)
            
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Val samples: {len(val_loader.dataset)}")
            print(f"Test samples: {len(test_loader.dataset)}")
            print(f"Image size: {self.image_size}")
            print(f"Loss types: {self.loss_types}")
        
        # Training variables
        best_val_score = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = getattr(self.config, 'early_stopping_patience', float('inf'))
        
        # Training loop
        if not self.config.distributed or self.config.rank == 0:
            fig = plot_sample_images(train_loader.dataset, num_samples=8)
            writer.add_figure("Samples/Train", fig, 0)
            plt.close(fig)
            fig = plot_sample_images(val_loader.dataset, num_samples=8)
            writer.add_figure("Samples/Val", fig, 0)
            plt.close(fig)

        val_eval_loader = None
        if not self.config.distributed or self.config.rank == 0:
            val_eval_loader = DataLoader(
                val_loader.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=getattr(self.config, "pin_memory", True),
            )

        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_accs = self.train_epoch(
                model, train_loader, optimizer, scaler, epoch, writer
            )
            
            # Validate
            val_metrics = self.evaluator.evaluate(model, val_loader)
            val_loss = val_metrics['loss']
            val_accuracies = val_metrics['accuracies']
            
            # Reduce validation metrics across all processes
            val_accuracies = self._reduce_dict(val_accuracies)
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            if self.config.distributed:
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_loss_tensor.item() / self.config.world_size
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Calculate average validation accuracy
            avg_val_acc = np.mean(list(val_accuracies.values()))
            
            # Early stopping and best model saving (only on rank 0)
            if not self.config.distributed or self.config.rank == 0:
                if avg_val_acc > best_val_score:
                    best_val_score = avg_val_acc
                    if self.config.distributed:
                        best_model_state = model.module.state_dict().copy()
                    else:
                        best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Broadcast early stopping decision
            if self.config.distributed:
                stop_training = torch.tensor(patience_counter >= max_patience, device=self.device)
                dist.broadcast(stop_training, src=0)
                should_stop = stop_training.item()
            else:
                should_stop = patience_counter >= max_patience
            
            # Logging (only on rank 0)
            if not self.config.distributed or self.config.rank == 0:
                if writer:
                    writer.add_scalar('Loss/Train', train_loss, epoch)
                    writer.add_scalar('Loss/Val', val_loss, epoch)
                    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

                    for task, acc in train_accs.items():
                        writer.add_scalar(f'Accuracy/Train_{task}', acc, epoch)
                    writer.add_scalar('Accuracy/Train_Average', np.mean(list(train_accs.values())), epoch)

                    for task, acc in val_accuracies.items():
                        writer.add_scalar(f'Accuracy/Val_{task}', acc, epoch)
                    writer.add_scalar('Accuracy/Val_Average', avg_val_acc, epoch)

                    if val_eval_loader is not None:
                        detailed = self.evaluator.evaluate(model, val_eval_loader, detailed=True)
                        cm_fig = self.evaluator.plot_confusion_matrices(detailed)
                        writer.add_figure('ConfusionMatrix/Val', cm_fig, epoch)
                        plt.close(cm_fig)

                        roc_fig = plot_roc_curves(detailed['labels'], detailed['probabilities'], self.task_names)
                        writer.add_figure('ROC/Val', roc_fig, epoch)
                        plt.close(roc_fig)

                        pr_fig = plot_pr_curves(detailed['labels'], detailed['probabilities'], self.task_names)
                        writer.add_figure('PR/Val', pr_fig, epoch)
                        plt.close(pr_fig)
                
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                print(f"Val Accuracies: {val_accuracies}")
                print(f"Average Val Accuracy: {avg_val_acc:.4f} - Best: {best_val_score:.4f}")
            
            # Early stopping
            if should_stop:
                if not self.config.distributed or self.config.rank == 0:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Synchronize all processes
            if self.config.distributed:
                dist.barrier(device_ids=[self.device.index])
        
        # Load best model and evaluate on test set (only on rank 0)
        test_accuracies = {}
        if not self.config.distributed or self.config.rank == 0:
            if best_model_state is not None:
                if self.config.distributed:
                    model.module.load_state_dict(best_model_state)
                else:
                    model.load_state_dict(best_model_state)
            
            test_metrics = self.evaluator.evaluate(model, test_loader, detailed=True)
            test_accuracies = test_metrics['accuracies']
            
            print(f"\nFold {fold_idx+1} Final Results:")
            print(f"Test Accuracies: {test_accuracies}")
            print(f"Average Test Accuracy: {np.mean(list(test_accuracies.values())):.4f}")
            
            # Save model
            if getattr(self.config, 'save_best_model', True):
                model_save_path = os.path.join(
                    self.config.output_dir, 'models', 
                    f"fold_{fold_idx+1}_best_model.pth"
                )
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': self.config,
                    'test_metrics': test_metrics,
                    'fold_idx': fold_idx,
                    'image_size': self.image_size,
                    'loss_types': self.loss_types
                }, model_save_path)
            
            if writer:
                writer.close()
        
        # Broadcast test accuracies to all processes
        if self.config.distributed:
            # Convert dict to tensor for broadcasting
            accuracy_values = list(test_accuracies.values()) if test_accuracies else [0.0] * len(self.task_names)
            accuracy_tensor = torch.tensor(accuracy_values, device=self.device)
            dist.broadcast(accuracy_tensor, src=0)
            
            # Convert back to dict
            test_accuracies = dict(zip(self.task_names, accuracy_tensor.cpu().tolist()))
        
        return test_accuracies
    
    def train_cross_validation(self) -> Dict[str, List[float]]:
        """Train with cross-validation and DDP support"""
        if not self.config.distributed or self.config.rank == 0:
            print(f"Starting cross-validation with {self.config.n_splits} folds")
            print(f"Using device: {self.device}")
            print(f"Backbone: {self.config.backbone}")
            print(f"Image size: {self.image_size}")
            if self.config.distributed:
                print(f"Distributed training: {self.config.world_size} GPUs")
        
        # Create data splits (only on rank 0, then broadcast)
        if not self.config.distributed or self.config.rank == 0:
            # Check if using existing splits or single CSV
            if hasattr(self.config, 'train_val_csv_path') and hasattr(self.config, 'test_csv_path'):
                # Using existing train/val and test CSV files
                folds, test_df = create_splits(
                    train_val_csv_path=self.config.train_val_csv_path,
                    test_csv_path=self.config.test_csv_path,
                    n_splits=self.config.n_splits,
                    seed=getattr(self.config, 'random_seed', 42)
                )
            elif hasattr(self.config, 'csv_path'):
                # Using single CSV file (original behavior)
                folds, test_df = create_splits(
                    csv_path=self.config.csv_path, 
                    n_splits=self.config.n_splits,
                    holdout_frac=getattr(self.config, 'holdout_frac', 0.2),
                    seed=getattr(self.config, 'random_seed', 42)
                )
            else:
                raise ValueError("Config must have either 'csv_path' OR both 'train_val_csv_path' and 'test_csv_path'")
        else:
            folds, test_df = None, None
        
        # Broadcast data splits to all processes
        if self.config.distributed:
            folds, test_df = self._broadcast_dataframes(folds, test_df)
        
        # Store results from all folds
        all_fold_results = []
        
        # Train each fold
        for fold_idx, (train_df, val_df) in enumerate(folds):
            fold_results = self.train_fold(fold_idx, train_df, val_df, test_df)
            all_fold_results.append(fold_results)
        
        # Calculate cross-validation statistics (only on rank 0)
        if not self.config.distributed or self.config.rank == 0:
            self._print_cv_results(all_fold_results)
        
        return all_fold_results
    
    def _print_cv_results(self, all_fold_results: List[Dict[str, float]]):
        """Print cross-validation results summary"""
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        
        for task in self.task_names:
            task_scores = [fold_result[task] for fold_result in all_fold_results]
            mean_score = np.mean(task_scores)
            std_score = np.std(task_scores)
            print(f"{task.upper()}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Overall average
        overall_scores = []
        for fold_result in all_fold_results:
            overall_scores.append(np.mean(list(fold_result.values())))
        
        print(f"OVERALL: {np.mean(overall_scores):.4f} ± {np.std(overall_scores):.4f}")
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.config.distributed:
            dist.destroy_process_group()

