#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List


def _load_yaml(path: str):
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read YAML sweep files.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sweep(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith((".yml", ".yaml")):
        data = _load_yaml(path)
    else:
        data = _load_json(path)

    if isinstance(data, dict):
        if "experiments" in data:
            exps = data["experiments"]
        else:
            # Allow single experiment dict
            exps = [data]
    elif isinstance(data, list):
        exps = data
    else:
        raise ValueError("Sweep file must be a list or dict with 'experiments'.")

    if not isinstance(exps, list) or not exps:
        raise ValueError("No experiments found in sweep file.")
    for idx, exp in enumerate(exps):
        if not isinstance(exp, dict):
            raise ValueError(f"Experiment #{idx} is not a dict.")
    return exps


def format_env_value(val: Any) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (list, tuple)):
        return ",".join(str(v) for v in val)
    return str(val)


ENV_MAP = {
    "model_name": "GMMFORMER_MODEL_NAME",
    "model_root": "GMMFORMER_MODEL_ROOT",
    "lr": "GMMFORMER_LR",
    "wd": "GMMFORMER_WD",
    "lr_warmup_proportion": "GMMFORMER_LR_WARMUP",
    "n_epoch": "GMMFORMER_N_EPOCH",
    "max_es_cnt": "GMMFORMER_MAX_ES_CNT",
    "batchsize": "GMMFORMER_BATCHSIZE",
    "num_workers": "GMMFORMER_NUM_WORKERS",
    "prefetch_factor": "GMMFORMER_PREFETCH_FACTOR",
    "persistent_workers": "GMMFORMER_PERSISTENT_WORKERS",
    "use_hard_negative": "GMMFORMER_USE_HARD_NEG",
    "hard_negative_start_epoch": "GMMFORMER_HARD_NEG_START",
    "hard_pool_size": "GMMFORMER_HARD_POOL_SIZE",
    "loss_factor": "GMMFORMER_LOSS_FACTOR",
    "neg_factor": "GMMFORMER_NEG_FACTOR",
    "margin": "GMMFORMER_MARGIN",
    "clip_scale_w": "GMMFORMER_CLIP_SCALE_W",
    "frame_scale_w": "GMMFORMER_FRAME_SCALE_W",
    "input_drop": "GMMFORMER_INPUT_DROP",
    "drop": "GMMFORMER_DROP",
    "segment_merge_ratio": "GMMFORMER_SEGMENT_MERGE_RATIO",
    "segment_merge_target": "GMMFORMER_SEGMENT_MERGE_TARGET",
}

CLI_MAP = {
    "steps_per_epoch": "--steps_per_epoch",
    "max_epoch": "--max_epoch",
    "stop_epoch": "--max_epoch",
    "accum_steps": "--accum_steps",
    "grad_clip": "--grad_clip",
    "train_shard_size": "--train_shard_size",
}

BOOL_CLI_FLAGS = {
    "amp": "--amp",
}

EPOCH_KEYS = ["n_epoch", "epochs", "epoch", "ep"]


def build_env_and_args(exp: Dict[str, Any], base_env: Dict[str, str]) -> (Dict[str, str], List[str]):
    env = dict(base_env)
    args = []

    # exp_name is an alias for model_name
    exp_name = exp.get("exp_name")
    if exp_name is not None and exp_name != "":
        env["GMMFORMER_MODEL_NAME"] = str(exp_name)

    # Epoch aliases
    for key in EPOCH_KEYS:
        if key in exp and exp[key] is not None:
            env["GMMFORMER_N_EPOCH"] = format_env_value(exp[key])
            break

    # Env overrides by config key
    for key, env_key in ENV_MAP.items():
        if key in ("model_name", "n_epoch"):
            continue
        if key in exp and exp[key] is not None:
            env[env_key] = format_env_value(exp[key])

    # Direct env keys
    for key, val in exp.items():
        if key.startswith("GMMFORMER_") and val is not None:
            env[key] = format_env_value(val)

    # CLI flags
    for key, flag in CLI_MAP.items():
        if key in exp and exp[key] is not None:
            args.extend([flag, str(exp[key])])

    for key, flag in BOOL_CLI_FLAGS.items():
        if exp.get(key) is True:
            args.append(flag)

    # Per-experiment determinism
    if exp.get("deterministic") is True:
        args.append("--deterministic")

    # Optional resume per experiment
    if exp.get("resume"):
        args.extend(["--resume", str(exp["resume"])])

    # Optional eval mode
    if exp.get("eval") is True:
        args.append("--eval")

    return env, args


def main():
    parser = argparse.ArgumentParser(description="Run sequential hyperparameter sweep for main.py")
    parser.add_argument("--dataset", default="act_frames", help="dataset name (e.g., act_frames)")
    parser.add_argument("--sweep", required=True, help="path to sweep JSON/YAML")
    parser.add_argument("--gpu", default=None, help="GPU id string (passed to main.py --gpu)")
    parser.add_argument("--out_csv", default=None, help="CSV output path for per-epoch metrics")
    parser.add_argument("--deterministic", action="store_true", help="pass --deterministic to main.py")
    parser.add_argument("--stop_on_fail", action="store_true", help="stop sweep if any experiment fails")
    parser.add_argument("--dry_run", action="store_true", help="print commands without executing")

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_path = args.sweep

    exps = load_sweep(sweep_path)

    # Default CSV path inside repo
    if args.out_csv:
        out_csv = os.path.abspath(args.out_csv)
    else:
        out_csv = os.path.join(repo_root, "results", "sweeps", f"{args.dataset}_sweep.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    base_env = os.environ.copy()

    failures = []
    for idx, exp in enumerate(exps):
        env, extra_args = build_env_and_args(exp, base_env)
        env["GMMFORMER_SWEEP_CSV"] = out_csv

        cmd = [sys.executable, "src/main.py", "-d", args.dataset]
        if args.gpu:
            cmd.extend(["--gpu", args.gpu])
        if args.deterministic:
            cmd.append("--deterministic")
        cmd.extend(extra_args)

        exp_label = exp.get("exp_name") or exp.get("model_name") or f"exp_{idx}"
        print(f"\n[RUN {idx + 1}/{len(exps)}] {exp_label}")
        print("CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        start = time.time()
        proc = subprocess.run(cmd, cwd=repo_root, env=env)
        duration = time.time() - start
        if proc.returncode != 0:
            failures.append((idx, exp_label, proc.returncode, duration))
            print(f"[FAIL] {exp_label} (code={proc.returncode}, {duration:.1f}s)")
            if args.stop_on_fail:
                break
        else:
            print(f"[OK] {exp_label} ({duration:.1f}s)")

    if failures:
        print("\nFailed experiments:")
        for idx, label, code, dur in failures:
            print(f"- #{idx + 1} {label}: code={code}, time={dur:.1f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
