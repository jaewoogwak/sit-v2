#!/usr/bin/env python3
"""Benchmark the similarity search path (clip + frame + argsort) for multiple models.

- Core pipeline stays identical to the previous GMMFormer script.
- Add --model dispatcher and optional --model-config (yaml/json) override.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict, replace
from typing import Callable, Dict, Tuple, Optional

import torch
import torch.nn.functional as F

# ---------------------------
# Optional YAML support
# ---------------------------
def _maybe_load_yaml(path: str) -> dict:
    if not path:
        return {}
    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    # best-effort YAML without hard dependency: try PyYAML if present
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"YAML config requested ({path}) but PyYAML not installed. "
            f"Install pyyaml or use JSON."
        ) from e
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

# ---------------------------
# Config & small utilities
# ---------------------------

@dataclass
class BenchmarkConfig:
    num_queries: int = 1
    num_videos: int = 20000
    num_clips: int = 32
    embedding_dim: int = 384
    clip_scale_w: float = 1.0
    frame_scale_w: float = 1.0
    pre_normalize_context: bool = False
    dtype: str = "float32"            # {"float16","float32","float64","bfloat16"}
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup: int = 5
    repeats: int = 20
    randomize: bool = False
    seed: int = 1234
    # model name carried for bookkeeping/printing
    model: str = "gmmformer"

def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device

def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_name]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"{dtype_name} is only supported on CUDA for this pipeline due to normalization.")
    return dtype

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def summarize(values) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "min": min(values), "max": max(values)}

# ---------------------------
# Model presets
# ---------------------------

MODEL_PRESETS = {
    # Shapes (N, L, D) with D=embedding_dim
    # You can still override via CLI or config.
    "gmmformer": {"num_clips": 33},   # (N, 33, 384) = (32+1)
    "ms-sl":     {"num_clips": 529},  # (N, 529, 384)
    "dldkd":     {"num_clips": 36},   # (N, 36, 384)
}

def apply_model_preset(cfg: BenchmarkConfig, model: str) -> BenchmarkConfig:
    model = model.lower()
    if model not in MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODEL_PRESETS.keys())}.")
    preset = MODEL_PRESETS[model]
    # For now, all models keep the frame branch (N,D). If you want to disable:
    #   preset.update({"frame_scale_w": 0.0})
    merged = {**asdict(cfg), **preset, "model": model}
    # if preset contains frame_scale_w, it will override; otherwise keep cfg's
    return BenchmarkConfig(**merged)

def apply_config_override(cfg: BenchmarkConfig, override: dict) -> BenchmarkConfig:
    if not override:
        return cfg
    allowed = set(k for k in asdict(cfg).keys())
    filtered = {k: v for k, v in override.items() if k in allowed}
    return replace(cfg, **filtered)

# ---------------------------
# GMMFormer-like kernels
# ---------------------------

@torch.inference_mode()
def make_embeddings(num_queries: int,
                    num_videos: int,
                    num_clips: int,
                    dim: int,
                    device: torch.device,
                    dtype: torch.dtype,
                    pre_normalize: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_queries <= 0:
        raise ValueError("num_queries must be positive.")
    if num_videos <= 0:
        raise ValueError("num_videos must be positive.")
    if num_clips <= 0:
        raise ValueError("num_clips must be positive.")

    query = torch.randn(num_queries, dim, device=device, dtype=dtype)
    frame_feat = torch.randn(num_videos, dim, device=device, dtype=dtype)
    clip_feat = torch.randn(num_videos, num_clips, dim, device=device, dtype=dtype)

    if pre_normalize:
        frame_feat = F.normalize(frame_feat, dim=-1)
        clip_feat = F.normalize(clip_feat, dim=-1)

    return query, clip_feat, frame_feat

@torch.inference_mode()
def prepare_inputs(query: torch.Tensor,
                   clip_feat: torch.Tensor,
                   frame_feat: torch.Tensor,
                   context_already_normalized: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_query = F.normalize(query, dim=-1)
    if context_already_normalized:
        normalized_clips = clip_feat
        normalized_frames = frame_feat
    else:
        normalized_clips = F.normalize(clip_feat, dim=-1)
        normalized_frames = F.normalize(frame_feat, dim=-1)
    return normalized_query, normalized_clips, normalized_frames

def compute_clip_scale_scores(normalized_query: torch.Tensor,
                              normalized_clip_feat: torch.Tensor) -> torch.Tensor:
    # normalized_clip_feat: (N, L, D)
    # normalized_query:     (M, D)
    sim = torch.matmul(normalized_clip_feat, normalized_query.t())  # (N, L, M)
    if sim.dim() == 2:
        clip_level_scores = sim.permute(1, 0).unsqueeze(0)          # (M, 1, N)
    else:
        clip_level_scores = sim.permute(2, 1, 0)                     # (M, L, N)
    query_context_scores, _ = torch.max(clip_level_scores, dim=1)    # (M, N)
    return query_context_scores

def compute_frame_scale_scores(normalized_query: torch.Tensor,
                               normalized_frame_feat: torch.Tensor) -> torch.Tensor:
    # normalized_frame_feat: (N, D)
    # normalized_query:      (M, D)
    sim = torch.matmul(normalized_frame_feat, normalized_query.t())  # (N, M)
    if sim.dim() == 1:
        return sim.unsqueeze(0)                                      # (1, N)
    return sim.permute(1, 0)                                         # (M, N)

@torch.inference_mode()
def measure_once(normalized_query: torch.Tensor,
                 normalized_clip_feat: torch.Tensor,
                 normalized_frame_feat: torch.Tensor,
                 clip_scale_w: float,
                 frame_scale_w: float,
                 sync: Callable[[], None]) -> Tuple[float, float, float, float]:
    sync()
    t_all0 = time.perf_counter()

    sync()
    t_clip0 = time.perf_counter()
    clip_scores = compute_clip_scale_scores(normalized_query, normalized_clip_feat)
    sync()
    t_clip1 = time.perf_counter()

    sync()
    t_frame0 = time.perf_counter()
    frame_scores = compute_frame_scale_scores(normalized_query, normalized_frame_feat)
    sync()
    t_frame1 = time.perf_counter()

    clip_scores.mul_(clip_scale_w)
    frame_scores.mul_(frame_scale_w)
    clip_scores.add_(frame_scores)

    sync()
    t_sort0 = time.perf_counter()
    _ = torch.argsort(clip_scores, dim=1)
    sync()
    t_sort1 = time.perf_counter()

    total_ms = (t_sort1 - t_all0) * 1000.0
    clip_ms = (t_clip1 - t_clip0) * 1000.0
    frame_ms = (t_frame1 - t_frame0) * 1000.0
    argsort_ms = (t_sort1 - t_sort0) * 1000.0

    return total_ms, clip_ms, frame_ms, argsort_ms

# ------------------------------------
# Public entry: benchmark_gmmformer()
# (usable for all models that fit this API)
# ------------------------------------

@torch.inference_mode()
def benchmark_gmmformer(
    config: BenchmarkConfig,
    *,
    make_embeddings_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    prepare_inputs_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    measure_once_fn: Optional[Callable[..., Tuple[float, float, float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run the current similarity+ranking benchmark and return stats.
    """
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    sync = lambda: _sync(device)

    make_embeddings_impl = make_embeddings_fn or make_embeddings
    prepare_inputs_impl = prepare_inputs_fn or prepare_inputs
    measure_once_impl = measure_once_fn or measure_once

    # Create & (optionally) normalize inputs once
    video_query, clip_feat, frame_feat = make_embeddings_impl(
        config.num_queries,
        config.num_videos,
        config.num_clips,
        config.embedding_dim,
        device,
        dtype,
        config.pre_normalize_context,
    )
    normalized_query, normalized_clips, normalized_frames = prepare_inputs_impl(
        video_query, clip_feat, frame_feat, config.pre_normalize_context
    )

    # Warmup
    for _ in range(max(config.warmup, 0)):
        if config.randomize:
            video_query, clip_feat, frame_feat = make_embeddings_impl(
                config.num_queries,
                config.num_videos,
                config.num_clips,
                config.embedding_dim,
                device,
                dtype,
                config.pre_normalize_context,
            )
            normalized_query, normalized_clips, normalized_frames = prepare_inputs_impl(
                video_query, clip_feat, frame_feat, config.pre_normalize_context
            )
        measure_once_impl(
            normalized_query, normalized_clips, normalized_frames,
            config.clip_scale_w, config.frame_scale_w, sync
        )

    # Timed runs
    totals, clips, frames, sorts = [], [], [], []
    for _ in range(max(config.repeats, 0)):
        if config.randomize:
            video_query, clip_feat, frame_feat = make_embeddings_impl(
                config.num_queries,
                config.num_videos,
                config.num_clips,
                config.embedding_dim,
                device,
                dtype,
                config.pre_normalize_context,
            )
            normalized_query, normalized_clips, normalized_frames = prepare_inputs_impl(
                video_query, clip_feat, frame_feat, config.pre_normalize_context
            )
        total, clip_ms, frame_ms, argsort_ms = measure_once_impl(
            normalized_query, normalized_clips, normalized_frames,
            config.clip_scale_w, config.frame_scale_w, sync
        )
        totals.append(total)
        clips.append(clip_ms)
        frames.append(frame_ms)
        sorts.append(argsort_ms)

    return {
        "meta": {
            "device": str(device),
            "dtype": str(dtype),
            "model": config.model,
            "queries": config.num_queries,
            "videos": config.num_videos,
            "clips_per_video": config.num_clips,
            "dim": config.embedding_dim,
            "clip_scale_w": config.clip_scale_w,
            "frame_scale_w": config.frame_scale_w,
            "pre_normalize_context": config.pre_normalize_context,
            "warmup": config.warmup,
            "repeats": config.repeats,
            "randomize": config.randomize,
        },
        "total": summarize(totals),
        "clip": summarize(clips),
        "frame": summarize(frames),
        "argsort": summarize(sorts),
    }

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the time to compute clip/frame similarities "
            "and run argsort for one or more queries against N videos."
        )
    )
    parser.add_argument("--model", type=str, default="gmmformer",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Which model preset to use.")
    parser.add_argument("--model-config", type=str, default="",
                        help="Optional path to YAML/JSON to override fields (e.g., num_clips, frame_scale_w, repeats, etc.).")

    parser.add_argument("--num-queries", type=int, default=1,
                        help="Number of query descriptors to benchmark.")
    parser.add_argument("--num-videos", type=int, default=20000,
                        help="Number of videos to compare each query against.")
    parser.add_argument("--num-clips", type=int, default=None,
                        help="Override clips/video; if omitted, use model preset.")
    parser.add_argument("--embedding-dim", type=int, default=384,
                        help="Dimensionality of descriptors (hidden size).")
    parser.add_argument("--clip-scale-w", type=float, default=None,
                        help="Weight applied to clip-level similarity scores.")
    parser.add_argument("--frame-scale-w", type=float, default=None,
                        help="Weight applied to frame-level similarity scores.")
    parser.add_argument("--pre-normalize-context", action="store_true",
                        help="Assume context features arrive normalized (skip extra normalization step).")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32", "float64", "bfloat16"],
                        help="Tensor dtype for all embeddings.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device string understood by torch, e.g. cpu, cuda, cuda:1.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warm-up iterations excluded from timing.")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Number of timed iterations.")
    parser.add_argument("--randomize", action="store_true",
                        help="Regenerate random embeddings before each iteration.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base random seed used for torch.")
    return parser.parse_args()

def print_stats(label: str, stats: Dict[str, float]) -> None:
    print(f"{label:<10}-> mean: {stats['mean']:.3f} ms | stdev: {stats['stdev']:.3f} | "
          f"min: {stats['min']:.3f} | max: {stats['max']:.3f}")

@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # base config from CLI
    cfg = BenchmarkConfig(
        num_queries=args.num_queries,
        num_videos=args.num_videos,
        num_clips=args.num_clips if args.num_clips is not None else 32,  # temp, will be overridden by preset
        embedding_dim=args.embedding_dim,
        clip_scale_w=args.clip_scale_w if args.clip_scale_w is not None else 1.0,
        frame_scale_w=args.frame_scale_w if args.frame_scale_w is not None else 1.0,
        pre_normalize_context=args.pre_normalize_context,
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        randomize=args.randomize,
        seed=args.seed,
        model=args.model,
    )

    # apply model preset (sets num_clips and other model-specific defaults)
    cfg = apply_model_preset(cfg, args.model)

    # apply external config file override (if provided)
    if args.model_config:
        override = _maybe_load_yaml(args.model_config)
        cfg = apply_config_override(cfg, override)

    results = benchmark_gmmformer(cfg)

    meta = results["meta"]
    print("device:", meta["device"])
    print("dtype:", meta["dtype"])
    print("model:", meta["model"])
    print(f"queries={meta['queries']}, videos={meta['videos']}, clips/video={meta['clips_per_video']}, dim={meta['dim']}")
    print(f"clip_scale_w={meta['clip_scale_w']}, frame_scale_w={meta['frame_scale_w']}, "
          f"pre_normalize_context={meta['pre_normalize_context']}")
    print(f"warmup={meta['warmup']}, repeats={meta['repeats']}, randomize={meta['randomize']}")
    print_stats("total", results["total"])
    print_stats("clip", results["clip"])
    print_stats("frame", results["frame"])
    print_stats("argsort", results["argsort"])

if __name__ == "__main__":
    main()
