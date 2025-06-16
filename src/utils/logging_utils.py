import logging
import os
from datetime import datetime
from typing import Optional

def setup_logging(output_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def log_model_info(model, logger: logging.Logger):
    """Log model architecture information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Architecture: {model.__class__.__name__}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

def log_dataset_info(datasets: dict, logger: logging.Logger):
    """Log dataset information"""
    for split_name, dataset in datasets.items():
        logger.info(f"{split_name} Dataset Size: {len(dataset)}")
        if hasattr(dataset, 'get_class_distribution'):
            distribution = dataset.get_class_distribution()
            for task, dist in distribution.items():
                logger.info(f"  {task} distribution: {dist}")