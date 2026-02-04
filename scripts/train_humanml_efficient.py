"""
Efficient training script for MotionPatches on HumanML3D dataset.
Simplified version with inline configuration.
"""

import os
import sys
import random
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.humanML import HumanMLDataset, collate_fn
from models.clip_point import ClipModel
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
# Based on conf/config_point.yaml and conf/dataset/HumanML3D_points.yaml
CONFIG = {
    # Experiment
    'exp_name': 'HumanML3D_PointCloud',
    'dataset_name': 'HumanML3D',
    
    # Data - from HumanML3D_points.yaml
    'data_root': './data/v4.3-wall-humanML3d-2136',
    'num_frames': 32, # 196
    'num_points': 2048,
    'batch_size': 32,  # From config_point.yaml
    'num_workers': 8,  # Reduced from 24 to prevent freezes with persistent workers
    'fps': 20,
    'max_motion_length': 224,
    
    # Model - from config_point.yaml
    'motion_encoder': 'vit_base_patch16_224_in21k',
    'text_encoder': 'distilbert-base-uncased',
    'motion_encoder_pretrained': True,
    'motion_embedding_dims': 768,
    'text_embedding_dims': 768,
    'projection_dims': 256,
    'patch_size': 16,  # From config_point.yaml
    'num_groups': 64,
    'dropout': 0.5,
    
    # Training - from config_point.yaml
    'train_motion_encoder': True,
    'train_text_encoder': True,
    'num_epochs': 50,
    'motion_lr': 1.0e-05,  # Base motion_lr from config_point.yaml
    'text_lr': 1.0e-05,    # Base text_lr from config_point.yaml
    'head_lr': 1.0e-05,    # Base head_lr from config_point.yaml
    'motion_lr_factor': 10.0,  # From HumanML3D_points.yaml
    'text_lr_factor': 1.0,     # From HumanML3D_points.yaml
    'head_lr_factor': 10.0,    # From HumanML3D_points.yaml
    'seed': 42,
    
    # Checkpoints
    'checkpoints_dir': './checkpoints/HumanML3D_PointCloud/HumanML3D',
    'save_every': 5,
    'log_every': 50,
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    logger.info("Creating datasets...")
    
    train_dataset = HumanMLDataset(
        data_root=config['data_root'],
        seed=config['seed'],
        split="train",
        num_frames=config['num_frames'],
        num_points=config['num_points']
    )
    
    val_dataset = HumanMLDataset(
        data_root=config['data_root'],
        seed=config['seed'],
        split="test",
        num_frames=config['num_frames'],
        num_points=config['num_points']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


def create_model(config, device):
    """Create and initialize the model."""
    logger.info("Creating model...")
    
    model = ClipModel(
        motion_encoder_alias=config['motion_encoder'],
        text_encoder_alias=config['text_encoder'],
        motion_encoder_pretrained=config['motion_encoder_pretrained'],
        motion_encoder_trainable=config['train_motion_encoder'],
        text_encoder_trainable=config['train_text_encoder'],
        motion_embedding_dims=config['motion_embedding_dims'],
        text_embedding_dims=config['text_embedding_dims'],
        projection_dims=config['projection_dims'],
        patch_size=config['patch_size'],
        dropout=0.5 if config['dataset_name'] == 'HumanML3D' else 0.0,
        num_frames=config['num_frames'],
        num_groups=config['num_groups'],
    )
    
    model.to(device)
    
    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer(config, model):
    """Create optimizer with separate learning rates."""
    logger.info("Creating optimizer...")
    
    # Apply learning rate factors from dataset config
    motion_lr = config['motion_lr'] * config['motion_lr_factor']
    text_lr = config['text_lr'] * config['text_lr_factor']
    head_lr = config['head_lr'] * config['head_lr_factor']
    
    parameters = [
        {
            "params": model.point_encoder.parameters(),
            "lr": motion_lr,
        },
        {
            "params": model.motion_encoder.parameters(),
            "lr": motion_lr,
        },
        {
            "params": model.text_encoder.parameters(),
            "lr": text_lr,
        },
        {
            "params": list(model.motion_projection.parameters()) + 
                     list(model.text_projection.parameters()),
            "lr": head_lr,
        },
    ]
    
    optimizer = Adam(parameters)
    logger.info(f"Motion LR: {motion_lr:.2e} (base: {config['motion_lr']:.2e} × {config['motion_lr_factor']})")
    logger.info(f"Text LR: {text_lr:.2e} (base: {config['text_lr']:.2e} × {config['text_lr_factor']})")
    logger.info(f"Head LR: {head_lr:.2e} (base: {config['head_lr']:.2e} × {config['head_lr_factor']})")
    
    return optimizer


def train_epoch(model, dataloader, optimizer, scheduler, tokenizer, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for batch_idx, batch in enumerate(pbar):
        motions, texts = batch
        motions = motions.to(device)
        
        # Tokenize texts
        texts = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        # Forward pass
        loss = model(motions, texts, return_loss=True)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        # Periodic logging
        if batch_idx % config['log_every'] == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, tokenizer, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        motions, texts = batch
        motions = motions.to(device)
        
        texts = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        loss = model(motions, texts, return_loss=True)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def compute_retrieval_metrics(model, dataloader, tokenizer, device):
    """Compute text-to-motion and motion-to-text retrieval metrics."""
    model.eval()
    
    all_motion_embeds = []
    all_text_embeds = []
    
    logger.info("Computing embeddings for retrieval...")
    
    for batch in tqdm(dataloader, desc="Computing embeddings", leave=False):
        motions, texts = batch
        motions = motions.to(device)
        
        texts = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        # Get embeddings
        motion_embeds = model.encode_motion(motions)
        text_embeds = model.encode_text(texts)
        
        all_motion_embeds.append(motion_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
    
    # Concatenate all embeddings
    all_motion_embeds = torch.cat(all_motion_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Normalize embeddings
    all_motion_embeds = all_motion_embeds / all_motion_embeds.norm(dim=-1, keepdim=True)
    all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    similarity = torch.matmul(all_text_embeds, all_motion_embeds.T)
    
    # Text-to-Motion retrieval
    t2m_ranks = []
    for i in range(similarity.shape[0]):
        rank = torch.where(torch.argsort(similarity[i], descending=True) == i)[0].item()
        t2m_ranks.append(rank)
    
    # Motion-to-Text retrieval
    m2t_ranks = []
    for i in range(similarity.shape[1]):
        rank = torch.where(torch.argsort(similarity[:, i], descending=True) == i)[0].item()
        m2t_ranks.append(rank)
    
    # Compute recall@k
    t2m_r1 = np.mean([r < 1 for r in t2m_ranks])
    t2m_r5 = np.mean([r < 5 for r in t2m_ranks])
    t2m_r10 = np.mean([r < 10 for r in t2m_ranks])
    
    m2t_r1 = np.mean([r < 1 for r in m2t_ranks])
    m2t_r5 = np.mean([r < 5 for r in m2t_ranks])
    m2t_r10 = np.mean([r < 10 for r in m2t_ranks])
    
    metrics = {
        't2m_r1': t2m_r1,
        't2m_r5': t2m_r5,
        't2m_r10': t2m_r10,
        'm2t_r1': m2t_r1,
        'm2t_r5': m2t_r5,
        'm2t_r10': m2t_r10,
    }
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, config, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }
    
    os.makedirs(config['checkpoints_dir'], exist_ok=True)
    filepath = os.path.join(config['checkpoints_dir'], filename)
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def main():
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(CONFIG)
    
    # Create model
    model = create_model(CONFIG, device)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['text_encoder'], 
        TOKENIZERS_PARALLELISM=True
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(CONFIG, model)
    # Match original scheduler: T_max = len(train_dataloader) * epoch * 2
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader) * CONFIG['num_epochs'] * 2
    )
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    best_t2m_r1 = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            tokenizer, device, CONFIG, epoch
        )
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, tokenizer, device)
        logger.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
        
        # Compute retrieval metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = compute_retrieval_metrics(model, val_loader, tokenizer, device)
            logger.info(f"Epoch {epoch+1} | Retrieval Metrics:")
            logger.info(f"  T2M: R@1={metrics['t2m_r1']:.4f}, R@5={metrics['t2m_r5']:.4f}, R@10={metrics['t2m_r10']:.4f}")
            logger.info(f"  M2T: R@1={metrics['m2t_r1']:.4f}, R@5={metrics['m2t_r5']:.4f}, R@10={metrics['m2t_r10']:.4f}")
            
            # Save best model based on T2M R@1
            if metrics['t2m_r1'] > best_t2m_r1:
                best_t2m_r1 = metrics['t2m_r1']
                save_checkpoint(model, optimizer, scheduler, epoch, CONFIG, 'best_model.pt')
                logger.info(f"New best model! T2M R@1: {best_t2m_r1:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % CONFIG['save_every'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, CONFIG, f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save last model
        save_checkpoint(model, optimizer, scheduler, epoch, CONFIG, 'last_model.pt')
        
        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        logger.info("-" * 60)
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best T2M R@1: {best_t2m_r1:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
