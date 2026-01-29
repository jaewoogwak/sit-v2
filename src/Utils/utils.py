import os
import torch
import torch.nn as nn
import random
import numpy as np
import logging
import math

import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

import ipdb


def set_seed(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:                   # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_log(file_path, file_name):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(os.path.join(file_path, file_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_ckpt(model, optimizer, config, ckpt_file, epoch, model_val):
    torch.save({
        'config': config,
        'epoch': epoch,
        'model_val': model_val,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt_file)


def load_ckpt(ckpt_file):
    ckpt = torch.load(ckpt_file, map_location="cpu")
    config = ckpt['config']
    model = ckpt['state_dict']
    optimizer = ckpt['optimizer']
    current_epoch = ckpt['epoch']
    model_val = ckpt['model_val']
    return config, model, optimizer, current_epoch, model_val


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    return _gpu_impl(data, _top=True)


_DEBUG_GPU_SHAPES_PRINTED = False


def _debug_shapes_enabled():
    return os.environ.get('GMMFORMER_DEBUG_SHAPES', '').strip() not in ('', '0', 'false', 'False')


def _summarize_shapes(x, prefix="batch", max_items=30):
    items = []

    def _add(name, obj):
        if torch.is_tensor(obj):
            items.append(f"{name}: {tuple(obj.shape)} {str(obj.dtype).replace('torch.', '')} {obj.device}")
        elif isinstance(obj, (list, tuple)):
            items.append(f"{name}: {type(obj).__name__}[{len(obj)}]")
        elif isinstance(obj, dict):
            items.append(f"{name}: dict({len(obj)})")
        else:
            items.append(f"{name}: {type(obj).__name__}")

    def _walk(obj, name):
        if len(items) >= max_items:
            return
        if torch.is_tensor(obj):
            _add(name, obj)
            return
        if isinstance(obj, dict):
            _add(name, obj)
            for k, v in obj.items():
                _walk(v, f"{name}.{k}")
            return
        if isinstance(obj, (list, tuple)):
            _add(name, obj)
            for i, v in enumerate(obj[:10]):
                _walk(v, f"{name}[{i}]")
            return
        _add(name, obj)

    _walk(x, prefix)
    if len(items) >= max_items:
        items.append("... (truncated)")
    return " | ".join(items)


def _gpu_impl(data, _top=False):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [_gpu_impl(x, _top=False) for x in data]
    elif isinstance(data, dict):
        data = {key: _gpu_impl(_data, _top=False) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)

    global _DEBUG_GPU_SHAPES_PRINTED
    if _top and (not _DEBUG_GPU_SHAPES_PRINTED) and _debug_shapes_enabled():
        try:
            logging.getLogger().info("DEBUG[pre-model gpu(batch)] " + _summarize_shapes(data))
        except Exception:
            pass
        _DEBUG_GPU_SHAPES_PRINTED = True

    return data
