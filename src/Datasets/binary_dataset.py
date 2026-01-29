import torch
import torch.utils.data as data
import numpy as np
import h5py
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import pickle
import json

from Datasets.data_provider import l2_normalize_np_array, average_to_fixed_length, uniform_feature_sampling


class BinaryTVRDataset(data.Dataset):
    """
    Dataset that bridges GMMFormer data format with binary hashing capabilities
    Compatible with both GMMFormer training and binary evaluation
    """
    
    def __init__(self, split: str, cfg: Dict, cache_dir: Path = None, 
                 binary_mode: bool = False, moment_split: bool = False):
        """
        Args:
            split: 'train', 'val', 'test'
            cfg: Configuration dictionary from GMMFormer
            cache_dir: Path to cached features (like new_tvr.py)
            binary_mode: If True, prepare data for binary evaluation
            moment_split: If True, split videos into segments like TVR moments
        """
        self.split = split
        self.cfg = cfg
        self.binary_mode = binary_mode
        self.moment_split = moment_split
        
        # GMMFormer configuration
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']
        
        # Paths
        self.data_root = Path(cfg['data_root'])
        self.cache_dir = cache_dir or Path("./tvr_binary_cache")
        
        # Load data based on mode
        if cache_dir and self.cache_dir.exists():
            self._load_cached_data()
        else:
            self._load_original_data()
            
        # Create moment splits if needed
        if moment_split:
            self._create_moment_splits()
    
    def _load_cached_data(self):
        """Load cached data similar to new_tvr.py"""
        try:
            self.text_features = torch.load(self.cache_dir / f"{self.split}_text.pt")
            self.video_features = torch.load(self.cache_dir / f"{self.split}_video.pt")
            
            if self.moment_split:
                self.index = torch.load(self.cache_dir / f"{self.split}_moment.pt")
            else:
                # Create video-level index
                self.video_ids = list(self.video_features.keys())
                self.cap_ids = list(self.text_features.keys())
                self._create_video_text_pairs()
                
            print(f"Loaded cached data for {self.split}")
        except FileNotFoundError:
            print(f"Cache not found, loading original data...")
            self._load_original_data()
    
    def _load_original_data(self):
        """Load data from original GMMFormer format"""
        # Load caption file
        cap_file = self.data_root / f"TextData/roberta/{self.split}.cap"
        text_feat_path = self.data_root / f"TextData/roberta/{self.split}_text_roberta.hdf5"
        
        # Load video features
        visual_feat_path = self.data_root / f"FeatureData/{self.cfg['visual_feature']}"
        video2frames_path = self.data_root / f"{self.cfg['collection']}_videos2frames.pkl"
        
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        
        # Load captions
        with open(cap_file, 'r') as f:
            for line in f.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = cap_id.split('#')[0]
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = [cap_id]
        
        # Load features
        self.text_feat_file = h5py.File(text_feat_path, 'r')
        
        with open(video2frames_path, 'rb') as f:
            self.video2frames = pickle.load(f)
        
        # Initialize visual feature reader (implement based on your visual feature format)
        self.visual_feat = self._init_visual_feat(visual_feat_path)
        
        # Create index
        self._create_video_text_pairs()
    
    def _init_visual_feat(self, visual_feat_path):
        """Initialize visual feature reader - adapt to your feature format"""
        # This is a placeholder - adapt based on your actual visual feature storage
        class VisualFeatReader:
            def __init__(self, feat_path):
                self.feat_path = feat_path
                # Load your visual features here
                
            def read_one(self, frame_id):
                # Implement frame reading logic
                # Return numpy array of shape (feature_dim,)
                pass
        
        return VisualFeatReader(visual_feat_path)
    
    def _create_video_text_pairs(self):
        """Create video-text pairs for training/evaluation"""
        self.pairs = []
        for video_id in self.video_ids:
            if video_id in self.vid_caps:
                for cap_id in self.vid_caps[video_id]:
                    self.pairs.append((video_id, cap_id))
    
    def _create_moment_splits(self):
        """Create moment-based splits like TVR dataset"""
        self.moment_pairs = []
        for video_id in self.video_ids:
            if video_id in self.vid_caps:
                # Split each video into 5 segments
                for moment_idx in range(5):
                    for cap_id in self.vid_caps[video_id]:
                        self.moment_pairs.append((cap_id, video_id, moment_idx))
        self.pairs = self.moment_pairs
    
    def __len__(self):
        if hasattr(self, 'pairs'):
            return len(self.pairs)
        else:
            return len(self.video_ids)
    
    def __getitem__(self, index):
        if self.binary_mode:
            return self._get_binary_item(index)
        else:
            return self._get_gmmformer_item(index)
    
    def _get_binary_item(self, index):
        """Get item in binary evaluation format"""
        if self.moment_split:
            cap_id, video_id, moment_idx = self.pairs[index]
            
            # Get video features
            if hasattr(self, 'video_features'):
                feats = self.video_features[video_id]
            else:
                feats = self._extract_video_features(video_id)
            
            # Extract segment
            T = feats.shape[0]
            seg_len = T // 5
            s = moment_idx * seg_len
            e = (moment_idx + 1) * seg_len if moment_idx < 4 else T
            seg = feats[s:e]
            
            # Sample uniform frames
            seg = self._sample_uniform(seg, 32)
            
            # Get text features
            if hasattr(self, 'text_features'):
                text_feat = self.text_features[cap_id]
            else:
                text_feat = self._extract_text_features(cap_id)
            
            return {
                'vid': seg,
                'txt': text_feat,
                'cap_id': cap_id,
                'video_id': video_id,
                'moment_idx': moment_idx
            }
        else:
            video_id, cap_id = self.pairs[index]
            
            # Get video features  
            if hasattr(self, 'video_features'):
                vid_feat = self.video_features[video_id]
            else:
                vid_feat = self._extract_video_features(video_id)
            
            # Get text features
            if hasattr(self, 'text_features'):
                text_feat = self.text_features[cap_id]
            else:
                text_feat = self._extract_text_features(cap_id)
            
            return {
                'vid': vid_feat,
                'txt': text_feat,
                'cap_id': cap_id,
                'video_id': video_id
            }
    
    def _get_gmmformer_item(self, index):
        """Get item in GMMFormer training format"""
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]
        
        # Extract video features
        clip_video_feature, frame_video_feature = self._extract_gmmformer_video_features(video_id)
        
        # Extract text features
        cap_tensors = []
        for cap_id in cap_ids:
            cap_feat = self._extract_text_features(cap_id)
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)
        
        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id
    
    def _extract_video_features(self, video_id):
        """Extract video features in original format"""
        if video_id not in self.video2frames:
            return torch.zeros(32, 512)  # fallback
            
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        
        # Convert to tensor
        frame_vecs = np.array(frame_vecs)
        return torch.from_numpy(frame_vecs).float()
    
    def _extract_gmmformer_video_features(self, video_id):
        """Extract video features in GMMFormer format"""
        if video_id not in self.video2frames:
            clip_feat = torch.zeros(1, self.map_size, 2048)  # Adjust dim based on your features
            frame_feat = torch.zeros(self.max_ctx_len, 2048)
            return clip_feat, frame_feat
            
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        
        # Clip-level features
        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        
        # Frame-level features
        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)
        
        return clip_video_feature, frame_video_feature
    
    def _extract_text_features(self, cap_id):
        """Extract text features"""
        if hasattr(self, 'text_feat_file'):
            cap_feat = self.text_feat_file[cap_id][...]
        else:
            # Fallback for cached features
            cap_feat = torch.zeros(768).numpy()  # RoBERTa dim
        
        return cap_feat
    
    def _sample_uniform(self, feat: torch.Tensor, N: int = 32) -> torch.Tensor:
        """Sample uniform frames similar to new_tvr.py"""
        T = feat.size(0)
        if T >= N:
            idx = torch.linspace(0, T - 1, steps=N).long()
            return feat[idx]
        pad = feat.new_zeros(N - T, feat.size(1))
        return torch.cat([feat, pad], 0)


def binary_collate_fn(batch):
    """Collate function for binary evaluation"""
    if isinstance(batch[0], dict):
        # Binary mode
        vids = torch.stack([item['vid'] for item in batch])
        txts = torch.stack([item['txt'] for item in batch])
        return {
            'vid': vids,
            'txt': txts,
            'cap_ids': [item['cap_id'] for item in batch],
            'video_ids': [item['video_id'] for item in batch]
        }
    else:
        # GMMFormer mode - use original collate function
        from Datasets.data_provider import collate_train
        return collate_train(batch)


def create_binary_cache(split: str, cfg: Dict, cache_dir: Path):
    """Create cache files compatible with new_tvr.py format"""
    cache_dir.mkdir(exist_ok=True)
    
    # Create dataset in original mode
    dataset = BinaryTVRDataset(split, cfg, binary_mode=False)
    
    # Extract and cache features
    text_features = {}
    video_features = {}
    
    print(f"Creating cache for {split} split...")
    for i in tqdm(range(len(dataset))):
        _, _, cap_tensors, _, cap_ids, video_id = dataset[i]
        
        # Cache text features
        for cap_tensor, cap_id in zip(cap_tensors, cap_ids):
            text_features[cap_id] = cap_tensor
        
        # Cache video features
        if video_id not in video_features:
            video_features[video_id] = dataset._extract_video_features(video_id)
    
    # Save cached features
    torch.save(text_features, cache_dir / f"{split}_text.pt")
    torch.save(video_features, cache_dir / f"{split}_video.pt")
    
    # Create moment index if needed
    moment_index = []
    for cap_id in text_features.keys():
        video_id = cap_id.split('#')[0]
        for moment_idx in range(5):
            moment_index.append((cap_id, video_id, moment_idx))
    
    torch.save(moment_index, cache_dir / f"{split}_moment.pt")
    
    # Create video index for evaluation
    video_index = []
    for cap_id in text_features.keys():
        video_id = cap_id.split('#')[0]
        video_index.append((cap_id, video_id))
    
    torch.save(video_index, cache_dir / f"{split}_video_idx.pt")
    
    print(f"Cache created: {len(text_features)} texts, {len(video_features)} videos")