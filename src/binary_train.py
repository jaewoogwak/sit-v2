#!/usr/bin/env python3
"""
Binary Training Pipeline for GMMFormer + Binary Hashing Integration
Combines GMMFormer's Gaussian Mixture Model with efficient binary retrieval
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from easydict import EasyDict as edict

# Add src to path
sys.path.append(str(Path(__file__).parent))

from Models.gmmformer.hybrid_model import GMMFormerBinary
from Datasets.binary_dataset import BinaryTVRDataset, binary_collate_fn, create_binary_cache
from Datasets.data_provider import collate_train
from Validations.binary_validation import evaluate_binary_hamming, evaluate_float_similarity
from Configs.tvr import get_cfg_defaults


class BinaryTrainer:
    """Trainer for GMMFormer + Binary Hashing"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training parameters
        self.epochs = getattr(cfg, 'binary_epochs', 50)
        self.lr = getattr(cfg, 'binary_lr', 5e-3)
        self.weight_decay = getattr(cfg, 'binary_wd', 1e-2)
        self.batch_size = getattr(cfg, 'binary_batch_size', 256)
        self.binary_temp = getattr(cfg, 'binary_temp', 0.07)
        
        # Model saving
        self.save_dir = Path(cfg['model_root']) / 'binary_models'
        self.save_dir.mkdir(exist_ok=True)
        
        # Cache directory
        self.cache_dir = Path("./tvr_binary_cache")
        
        # Initialize model
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        
        # Load pretrained GMMFormer if specified
        if hasattr(cfg, 'gmmformer_checkpoint') and cfg.gmmformer_checkpoint:
            self._load_gmmformer_checkpoint(cfg.gmmformer_checkpoint)
    
    def _build_model(self):
        """Build GMMFormerBinary model"""
        model = GMMFormerBinary(self.cfg)
        model = model.to(self.device)
        
        print(f"Model built with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _build_optimizer(self):
        """Build AdamW optimizer"""
        # Separate learning rates for GMMFormer and binary projection
        gmmformer_params = []
        binary_params = []
        
        for name, param in self.model.named_parameters():
            if 'binary_proj' in name:
                binary_params.append(param)
            else:
                gmmformer_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': gmmformer_params, 'lr': self.lr * 0.1},  # Lower LR for pretrained parts
            {'params': binary_params, 'lr': self.lr}  # Higher LR for binary projection
        ], weight_decay=self.weight_decay)
        
        return optimizer
    
    def _load_gmmformer_checkpoint(self, checkpoint_path):
        """Load pretrained GMMFormer weights"""
        print(f"Loading GMMFormer checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load only GMMFormer weights (exclude binary projection layers)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() 
                          if k in model_dict and 'binary_proj' not in k}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)} pretrained parameters")
    
    def create_cache_if_needed(self):
        """Create cached features if they don't exist"""
        if not self.cache_dir.exists() or len(list(self.cache_dir.glob("*.pt"))) == 0:
            print("Creating binary cache...")
            self.cache_dir.mkdir(exist_ok=True)
            
            for split in ['train', 'val']:
                create_binary_cache(split, self.cfg, self.cache_dir)
            print("Cache creation completed!")
    
    def _get_data_loaders(self):
        """Create data loaders for training and validation"""
        # Training dataset (GMMFormer format)
        train_dataset = BinaryTVRDataset(
            'train', self.cfg, self.cache_dir, binary_mode=False
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_train, num_workers=4, pin_memory=True
        )
        
        # Validation dataset (binary format)
        val_dataset = BinaryTVRDataset(
            'val', self.cfg, self.cache_dir, binary_mode=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.get('eval_query_bsz', 50),
            shuffle=False, collate_fn=binary_collate_fn, num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        gmmformer_loss = 0
        binary_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # GMMFormer losses (from original implementation)
            gmmformer_outputs = outputs['gmmformer_output']
            clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query = gmmformer_outputs
            
            # Compute GMMFormer losses (adapt from your loss implementation)
            loss_gmm = self._compute_gmmformer_loss(
                clip_scale_scores, clip_scale_scores_, label_dict, 
                frame_scale_scores, frame_scale_scores_, batch
            )
            
            # Binary contrastive loss
            loss_binary = self.model.compute_contrastive_loss(
                outputs['text_binary_norm'], 
                outputs['video_binary_norm'],
                batch.get('text_labels')
            )
            
            # Combined loss
            alpha = getattr(self.cfg, 'binary_loss_weight', 0.5)
            loss = (1 - alpha) * loss_gmm + alpha * loss_binary
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            gmmformer_loss += loss_gmm.item()
            binary_loss += loss_binary.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{loss.item():.4f}',
                'GMM': f'{loss_gmm.item():.4f}',
                'Binary': f'{loss_binary.item():.4f}'
            })
        
        avg_total_loss = total_loss / len(train_loader)
        avg_gmm_loss = gmmformer_loss / len(train_loader)
        avg_binary_loss = binary_loss / len(train_loader)
        
        print(f"Epoch {epoch} - Total: {avg_total_loss:.4f}, GMM: {avg_gmm_loss:.4f}, Binary: {avg_binary_loss:.4f}")
        
        return avg_total_loss
    
    def _compute_gmmformer_loss(self, clip_scale_scores, clip_scale_scores_, label_dict, 
                               frame_scale_scores, frame_scale_scores_, batch):
        """Compute original GMMFormer losses"""
        # This is a simplified version - adapt based on your actual loss implementation
        from Models.gmmformer.model_components import clip_nce, frame_nce
        
        # Clip-level loss
        clip_loss_func = clip_nce()
        clip_loss = clip_loss_func(
            batch['text_labels'], label_dict, clip_scale_scores_
        )
        
        # Frame-level loss
        frame_loss_func = frame_nce()
        frame_loss = frame_loss_func(frame_scale_scores_)
        
        # Combined loss with weights from config
        loss_weights = self.cfg.get('loss_factor', [0.05, 0.04, 0.001])
        total_loss = loss_weights[0] * clip_loss + loss_weights[1] * frame_loss
        
        return total_loss
    
    def evaluate(self, epoch=None):
        """Evaluate model on validation set"""
        print(f"\n{'='*50}")
        print(f"Evaluation {'(Epoch ' + str(epoch) + ')' if epoch else ''}")
        print(f"{'='*50}")
        
        # Float similarity evaluation
        print("\n--- Float Similarity Evaluation ---")
        float_results, float_sumr = evaluate_float_similarity(
            self.model, self.cfg, 'val', self.cache_dir, self.device
        )
        
        # Binary hamming evaluation (if available)
        print("\n--- Binary Hamming Evaluation ---")
        try:
            binary_results, binary_sumr = evaluate_binary_hamming(
                self.model, self.cfg, 'val', self.cache_dir, self.device
            )
        except ImportError:
            print("Binary index not available, skipping binary evaluation")
            binary_results, binary_sumr = {}, 0
        
        return {
            'float': float_results,
            'binary': binary_results,
            'float_sumr': float_sumr,
            'binary_sumr': binary_sumr
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.cfg
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        print("Starting Binary Training...")
        
        # Create cache if needed
        self.create_cache_if_needed()
        
        # Get data loaders
        train_loader, val_loader = self._get_data_loaders()
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Initial evaluation
        print("\nInitial evaluation:")
        best_sumr = 0
        initial_metrics = self.evaluate(0)
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Train one epoch
            avg_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate every few epochs
            if epoch % 5 == 0 or epoch == self.epochs:
                metrics = self.evaluate(epoch)
                
                # Check if best model
                current_sumr = metrics['float_sumr'] + metrics['binary_sumr']
                is_best = current_sumr > best_sumr
                if is_best:
                    best_sumr = current_sumr
                
                # Save checkpoint
                self.save_checkpoint(epoch, metrics, is_best)
        
        print("\nTraining completed!")
        print(f"Best SumR: {best_sumr:.3f}")


def main():
    parser = argparse.ArgumentParser(description="GMMFormer Binary Training")
    parser.add_argument('--config', type=str, default='tvr', choices=['tvr', 'act', 'cha'],
                       help='Configuration to use')
    parser.add_argument('--gmmformer_checkpoint', type=str, default=None,
                       help='Path to pretrained GMMFormer checkpoint')
    parser.add_argument('--binary_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--binary_lr', type=float, default=5e-3,
                       help='Learning rate for binary projection')
    parser.add_argument('--binary_batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--binary_temp', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    parser.add_argument('--binary_loss_weight', type=float, default=0.5,
                       help='Weight for binary loss (vs GMMFormer loss)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--create_cache', action='store_true',
                       help='Only create cache files')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'tvr':
        from Configs.tvr import get_cfg_defaults
    elif args.config == 'act':
        from Configs.act import get_cfg_defaults
    elif args.config == 'cha':
        from Configs.cha import get_cfg_defaults
    
    cfg = get_cfg_defaults()
    
    # Add binary-specific configs
    cfg['binary_dim'] = 3008
    cfg['binary_act'] = 'tanh'
    cfg['binary_temp'] = args.binary_temp
    cfg['binary_epochs'] = args.binary_epochs
    cfg['binary_lr'] = args.binary_lr
    cfg['binary_batch_size'] = args.binary_batch_size
    cfg['binary_loss_weight'] = args.binary_loss_weight
    if args.gmmformer_checkpoint:
        cfg['gmmformer_checkpoint'] = args.gmmformer_checkpoint
    
    # Create trainer
    trainer = BinaryTrainer(cfg)
    
    if args.create_cache:
        print("Creating cache files...")
        trainer.create_cache_if_needed()
        print("Cache creation completed!")
        return
    
    if args.eval_only:
        print("Running evaluation only...")
        trainer.evaluate()
        return
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()