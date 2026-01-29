import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from Datasets.binary_dataset import BinaryTVRDataset, binary_collate_fn


def _pack_binary(x, binary_dim=3008):
    """Pack binary representations for efficient hamming distance computation"""
    bits = (x > 0).long().view(x.size(0), binary_dim // 64, 64)
    shifts = torch.arange(64, dtype=torch.int64, device=x.device)
    return (bits << shifts).sum(-1).contiguous()


def _pack_min_max(x, dim_stats=None, binary_dim=3008):
    """Min-Max distance-based binarization from new_tvr.py"""
    if dim_stats is not None:
        dim_min, dim_max = dim_stats['min'], dim_stats['max']
    else:
        dim_min = x.min(dim=0)[0]
        dim_max = x.max(dim=0)[0]
        
    dist_to_min = torch.abs(x - dim_min)
    dist_to_max = torch.abs(x - dim_max)
    bits = (dist_to_max < dist_to_min).long()
    
    bits = bits.view(bits.size(0), binary_dim // 64, 64)
    shifts = torch.arange(64, dtype=torch.int64, device=x.device)
    return (bits << shifts).sum(-1).contiguous()


def compute_dim_stats(data_loader, model, device):
    """
    Compute dimension-wise min/max statistics for min-max binarization
    """
    all_projections = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing dimension statistics"):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get model outputs
            outputs = model.encode_for_retrieval(batch, mode='binary')
            
            # Collect projections
            if 'text_feat' in batch:
                all_projections.append(outputs['binary'])
            elif 'clip_video_features' in batch:  
                all_projections.append(outputs['binary'])
    
    # Combine all projections
    all_data = torch.cat(all_projections, dim=0)
    
    return {
        'min': all_data.min(dim=0)[0],
        'max': all_data.max(dim=0)[0]
    }


@torch.no_grad()
def evaluate_binary_hamming(model, cfg, split='val', cache_dir=None, device='cuda'):
    """
    Evaluate using binary hamming distance similar to new_tvr.py
    """
    # Import binary index
    try:
        from binary_index import bp_hamming
    except ImportError:
        print("Warning: binary_index not found. Using cosine similarity fallback.")
        return evaluate_float_similarity(model, cfg, split, cache_dir, device)
    
    binary_dim = getattr(cfg, 'binary_dim', 3008)
    
    # Create datasets
    video_dataset = BinaryTVRDataset(
        split, cfg, cache_dir, binary_mode=True, moment_split=False
    )
    text_dataset = BinaryTVRDataset(
        split, cfg, cache_dir, binary_mode=True, moment_split=False  
    )
    
    # Create data loaders
    video_loader = DataLoader(
        video_dataset, batch_size=cfg.get('eval_context_bsz', 100),
        shuffle=False, collate_fn=binary_collate_fn, num_workers=4
    )
    text_loader = DataLoader(
        text_dataset, batch_size=cfg.get('eval_query_bsz', 50),
        shuffle=False, collate_fn=binary_collate_fn, num_workers=4
    )
    
    # Load caption → video index
    if cache_dir:
        cid_vid_pairs = torch.load(cache_dir / f"{split}_video_idx.pt")
    else:
        # Create mapping from dataset
        cid_vid_pairs = [(cap_id, video_id) for video_id, cap_id in video_dataset.pairs]
    
    cid_to_vid = dict(cid_vid_pairs)
    unique_vids = sorted({vid for _, vid in cid_vid_pairs})
    vid_to_index = {vid: i for i, vid in enumerate(unique_vids)}
    
    model.eval()
    
    # Build video representations
    print("Building video representations...")
    video_projections = []
    video_hashes = {}
    
    for batch in tqdm(video_loader, desc="Processing videos"):
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Get video encodings
        outputs = model.encode_for_retrieval(batch, mode='binary')
        video_projections.append(outputs['binary'].cpu())
        
        # Track duplicates
        for vid_repr in outputs['binary']:
            h = hash(vid_repr.cpu().numpy().tobytes())
            video_hashes[h] = video_hashes.get(h, 0) + 1
    
    video_proj = torch.cat(video_projections, dim=0)  # (V, binary_dim)
    
    # Build text representations  
    print("Building text representations...")
    text_projections = []
    
    for batch in tqdm(text_loader, desc="Processing texts"):
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Get text encodings
        outputs = model.encode_for_retrieval(batch, mode='binary')
        text_projections.append(outputs['binary'].cpu())
    
    text_proj = torch.cat(text_projections, dim=0)  # (Q, binary_dim)
    
    # Sanity check
    print(f"\n─── Binary Hamming Evaluation ───")
    print(f"Total unique videos: {len(video_proj):,}")
    print(f"Total queries: {len(text_proj):,}")
    print(f"Duplicate video representations: {sum(v > 1 for v in video_hashes.values()):,} / {len(video_proj):,}")
    print("──────────────────────────────────")
    
    # Binary packing for hamming distance
    txt_hv = _pack_binary(text_proj, binary_dim).cuda()
    vid_hv = _pack_binary(video_proj, binary_dim).cuda()
    
    # Create binary index
    index = bp_hamming(binary_dim)
    index.add(vid_hv)
    
    # Evaluation
    print("\nEvaluating Binary Hamming Recall@K...")
    Ks = [1, 5, 10, 100]
    hits = {k: 0 for k in Ks}
    n = len(text_proj)
    
    for i, cid in enumerate(cid_to_vid.keys()):
        true_vid = cid_to_vid[cid]
        gt_index = vid_to_index[true_vid]
        
        # Search using hamming distance
        _, topk = index.search(txt_hv[i].unsqueeze(0), Ks[-1])
        
        for k in Ks:
            if gt_index in topk[0][:k]:
                hits[k] += 1
    
    # Print results
    sumr = 0
    for k in Ks:
        r = hits[k] / n
        print(f"Binary R@{k:<3}= {r:.3%} ({hits[k]}/{n})")
        sumr += r
    print(f"Binary SumR = {sumr * 100:.3f}")
    
    return {f'R@{k}': hits[k] / n for k in Ks}, sumr


@torch.no_grad()
def evaluate_float_similarity(model, cfg, split='val', cache_dir=None, device='cuda'):
    """
    Evaluate using float cosine similarity (fallback when binary index unavailable)
    """
    # Create datasets
    video_dataset = BinaryTVRDataset(
        split, cfg, cache_dir, binary_mode=True, moment_split=False
    )
    text_dataset = BinaryTVRDataset(
        split, cfg, cache_dir, binary_mode=True, moment_split=False
    )
    
    # Create data loaders
    video_loader = DataLoader(
        video_dataset, batch_size=cfg.get('eval_context_bsz', 100),
        shuffle=False, collate_fn=binary_collate_fn, num_workers=4
    )
    text_loader = DataLoader(
        text_dataset, batch_size=cfg.get('eval_query_bsz', 50), 
        shuffle=False, collate_fn=binary_collate_fn, num_workers=4
    )
    
    # Load caption → video index
    if cache_dir:
        cid_vid_pairs = torch.load(cache_dir / f"{split}_video_idx.pt")
    else:
        cid_vid_pairs = [(cap_id, video_id) for video_id, cap_id in video_dataset.pairs]
    
    cid_to_vid = dict(cid_vid_pairs)
    unique_vids = sorted({vid for _, vid in cid_vid_pairs})
    vid_to_index = {vid: i for i, vid in enumerate(unique_vids)}
    
    model.eval()
    
    # Build representations
    print("Building video representations (float)...")
    video_projections = []
    
    for batch in tqdm(video_loader, desc="Processing videos"):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        outputs = model.encode_for_retrieval(batch, mode='binary')
        # Normalize for cosine similarity
        vid_repr = F.normalize(outputs['binary'], dim=-1)
        video_projections.append(vid_repr.cpu())
    
    video_proj = torch.cat(video_projections, dim=0)
    
    print("Building text representations (float)...")
    text_projections = []
    
    for batch in tqdm(text_loader, desc="Processing texts"):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        outputs = model.encode_for_retrieval(batch, mode='binary')
        # Normalize for cosine similarity
        text_repr = F.normalize(outputs['binary'], dim=-1)
        text_projections.append(text_repr.cpu())
    
    text_proj = torch.cat(text_projections, dim=0)
    
    print(f"\n─── Float Similarity Evaluation ───")
    print(f"Total unique videos: {len(video_proj):,}")
    print(f"Total queries: {len(text_proj):,}")
    print("───────────────────────────────────")
    
    # Compute similarities
    print("\nEvaluating Float Cosine Recall@K...")
    similarities = text_proj @ video_proj.T  # (Q, V)
    
    Ks = [1, 5, 10, 100]
    hits = {k: 0 for k in Ks}
    n = len(text_proj)
    
    for i, cid in enumerate(cid_to_vid.keys()):
        true_vid = cid_to_vid[cid]
        gt_index = vid_to_index[true_vid]
        
        query_similarities = similarities[i]  # (V,)
        _, topk_indices = torch.topk(query_similarities, k=Ks[-1], largest=True)
        topk_indices = topk_indices.tolist()
        
        for k in Ks:
            if gt_index in topk_indices[:k]:
                hits[k] += 1
    
    # Print results
    sumr = 0
    for k in Ks:
        r = hits[k] / n
        print(f"Float R@{k:<3}= {r:.3%} ({hits[k]}/{n})")
        sumr += r
    print(f"Float SumR = {sumr * 100:.3f}")
    
    return {f'R@{k}': hits[k] / n for k in Ks}, sumr


@torch.no_grad()
def evaluate_moment_retrieval(model, cfg, split='val', cache_dir=None, device='cuda'):
    """
    Evaluate moment-level retrieval (similar to TVR moment retrieval task)
    """
    # Create moment dataset
    moment_dataset = BinaryTVRDataset(
        split, cfg, cache_dir, binary_mode=True, moment_split=True
    )
    
    moment_loader = DataLoader(
        moment_dataset, batch_size=cfg.get('eval_query_bsz', 50),
        shuffle=False, collate_fn=binary_collate_fn, num_workers=4
    )
    
    model.eval()
    
    # Process moments
    print("Processing moments...")
    moment_projections = []
    moment_info = []
    
    for batch in tqdm(moment_loader, desc="Processing moments"):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Get moment encodings
        outputs = model.encode_for_retrieval(batch, mode='binary')
        
        # Store projections and info
        moment_projections.append(outputs['binary'].cpu())
        moment_info.extend([
            (batch['cap_ids'][i], batch['video_ids'][i], batch.get('moment_idx', [0] * len(batch['cap_ids']))[i])
            for i in range(len(batch['cap_ids']))
        ])
    
    moment_proj = torch.cat(moment_projections, dim=0)
    moment_proj_norm = F.normalize(moment_proj, dim=-1)
    
    print(f"\n─── Moment Retrieval Evaluation ───")
    print(f"Total moments: {len(moment_proj):,}")
    print("──────────────────────────────────")
    
    # Group by video for evaluation
    video_moments = defaultdict(list)
    for i, (cap_id, video_id, moment_idx) in enumerate(moment_info):
        video_moments[video_id].append((i, cap_id, moment_idx))
    
    # Evaluation
    Ks = [1, 3, 5]
    hits = {k: 0 for k in Ks}
    total_queries = 0
    
    for video_id, moments in video_moments.items():
        if len(moments) < 2:  # Need at least 2 moments for evaluation
            continue
            
        for query_idx, (idx, cap_id, true_moment) in enumerate(moments):
            # Query representation
            query_repr = moment_proj_norm[idx:idx+1]  # (1, D)
            
            # Candidate representations (all moments in this video)
            candidate_indices = [m[0] for m in moments]
            candidate_repr = moment_proj_norm[candidate_indices]  # (M, D)
            
            # Compute similarities
            similarities = query_repr @ candidate_repr.T  # (1, M)
            _, topk_indices = torch.topk(similarities, k=min(len(moments), Ks[-1]), largest=True)
            topk_indices = topk_indices[0].tolist()
            
            # Check if correct moment is retrieved
            for k in Ks:
                if k > len(moments):
                    continue
                if query_idx in topk_indices[:k]:
                    hits[k] += 1
            
            total_queries += 1
    
    # Print results
    print("\nMoment Retrieval Results:")
    sumr = 0
    for k in Ks:
        r = hits[k] / total_queries if total_queries > 0 else 0
        print(f"Moment R@{k:<3}= {r:.3%} ({hits[k]}/{total_queries})")  
        sumr += r
    print(f"Moment SumR = {sumr * 100:.3f}")
    
    return {f'Moment_R@{k}': hits[k] / total_queries if total_queries > 0 else 0 for k in Ks}, sumr