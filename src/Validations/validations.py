import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import ipdb

import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
import time

from tqdm import tqdm

from Utils.utils import gpu
import json


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def log_cuda(tag: str):
    """Log CUDA memory (allocated, reserved, peak) in GiB with a tag."""
    if not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[MEM] {tag} | alloc={alloc:.2f}G reserved={reserved:.2f}G peak_alloc={peak:.2f}G")
    except Exception:
        pass


def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:

                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):

    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    eval_q2m._per_query_end_times = []
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort(s)
        if hasattr(eval_q2m, '_timing_end2'):
            _sync()
            eval_q2m._timing_end2 = time.perf_counter()
        _sync()
        eval_q2m._per_query_end_times.append(time.perf_counter())
        rank = n_m + 1
        tmp_set = []
        keys = q2m_gts.get(i, []) if isinstance(q2m_gts, dict) else q2m_gts[i]
        for k in keys:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):

    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)


class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()

        self.cfg = cfg


    def forward(self, model, context_dataloader, query_eval_loader):

        model.eval()

        measure_timing = bool(self.cfg.get("measure_search", False))
        timing_quiet = bool(self.cfg.get("timing_quiet", False))

        context_info = self.compute_context_info(model, context_dataloader, timing_quiet=timing_quiet)

        num_videos = len(context_info['video_metas'])

        score_sum, query_metas, timing_report = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info,
                                                             timing_quiet=timing_quiet)
        video_metas = context_info['video_metas']

        if measure_timing:
            _sync()
            get_gt_start = time.perf_counter()
        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
        if measure_timing:
            _sync()
            get_gt_end = time.perf_counter()
            get_gt_time = (get_gt_end - get_gt_start) * 1000.0
        else:
            get_gt_time = 0.0

        if measure_timing:
            eval_q2m._timing_end2 = None

        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

        similarity_check_time = None
        if measure_timing and timing_report and 'similarity_start_times' in timing_report:
            starts = timing_report['similarity_start_times']
            ends = timing_report.get('similarity_end_times', None)
            if ends and len(ends) == len(starts):
                durs_ms = [(ends[i] - starts[i]) * 1000.0 for i in range(len(starts))]
                if len(durs_ms) > 0:
                    similarity_check_time = float(np.mean(durs_ms))
            else:
                if hasattr(eval_q2m, '_timing_end2') and eval_q2m._timing_end2:
                    avg_start_time = sum(starts) / len(starts)
                    similarity_check_time = (eval_q2m._timing_end2 - avg_start_time) * 1000.0 - get_gt_time

        if timing_report:
            e2e_arr = None
            try:
                starts = timing_report.get('similarity_start_times', None)
                ends = timing_report.get('similarity_end_times', None)
                if ends and starts and len(ends) == len(starts) and len(starts) > 0:
                    e2e_arr = np.array([(ends[i] - starts[i]) * 1000.0 for i in range(len(starts))], dtype=np.float64)
                else:
                    end_times = getattr(eval_q2m, '_per_query_end_times', None)
                    if end_times and starts and len(end_times) > 0 and len(starts) > 0:
                        n = min(len(end_times), len(starts))
                        ends_aligned = end_times[-n:]
                        starts_aligned = starts[-n:]
                        per_gt_ms = (get_gt_time / n) if measure_timing else 0.0
                        e2e = [(ends_aligned[i] - starts_aligned[i]) * 1000.0 - per_gt_ms for i in range(n)]
                        e2e_arr = np.array(e2e, dtype=np.float64)
            except Exception:
                e2e_arr = None

            if measure_timing and not timing_quiet and 'encoding_normalize_avg_ms' in timing_report and similarity_check_time:
                encoding_time = timing_report['encoding_normalize_avg_ms']
                total_search_time = encoding_time + similarity_check_time

                print()
                print("[Performance Timing Analysis]")
                print(f"Total search time: {total_search_time:.2f} ms")
                print(f"Encoding & normalize time: {encoding_time:.2f} ms")
                print(f"Similarity check time: {similarity_check_time:.2f} ms")
                print(f"Videos compared per query: {num_videos}")
                timing_topk = int(timing_report.get('timing_topk', 0))
                if timing_topk > 0:
                    print(f"Timing mode: top-k (k={timing_topk})")
                dev = timing_report.get('timing_device', None)
                if dev:
                    print(f"Timing device: {dev.upper()}")
                if measure_timing:
                    print(f"(GT mapping time excluded: {get_gt_time:.2f} ms)")
                print()

            if 'per_query_ms' in timing_report:
                pq_stats = timing_report['per_query_ms']
                enc_avg = timing_report.get('encoding_normalize_avg_ms', None)
                sort_stats = timing_report.get('argsort_ms', None)
                timing_topk = int(timing_report.get('timing_topk', 0))
                dev = timing_report.get('timing_device', None)

                if timing_quiet:
                    lines = ["[Per-Query Timing]", f"Videos compared per query: {num_videos}"]
                    if 'avg' in pq_stats:
                        lines.append(f"Similarity (all videos) avg: {pq_stats['avg']:.2f} ms")
                    if enc_avg is not None:
                        lines.append(f"Encoding + normalize avg: {enc_avg:.2f} ms")
                    if 'avg' in pq_stats and enc_avg is not None:
                        lines.append(f"Total per-query avg (enc + sim): {enc_avg + pq_stats['avg']:.2f} ms")
                    if all(k in pq_stats for k in ('p50', 'p90', 'p99')):
                        lines.append(f"Similarity per-query p50/p90/p99: {pq_stats['p50']:.2f} / {pq_stats['p90']:.2f} / {pq_stats['p99']:.2f} ms")
                    if sort_stats and isinstance(sort_stats, dict):
                        label = 'top-k' if timing_topk > 0 else 'argsort'
                        lines.append(f"{label.capitalize()} avg: {sort_stats.get('avg', 0.0):.2f} ms")
                        if all(k in sort_stats for k in ('p50', 'p90', 'p99')):
                            lines.append(f"{label.capitalize()} p50/p90/p99: {sort_stats['p50']:.2f} / {sort_stats['p90']:.2f} / {sort_stats['p99']:.2f} ms")
                    if dev:
                        lines.append(f"Timing device: {dev.upper()}")
                    if e2e_arr is not None and e2e_arr.size > 0:
                        lines.append(f"Similarity (end-to-end, incl argsort) avg: {e2e_arr.mean():.2f} ms")
                        lines.append(f"Similarity (end-to-end) p50/p90/p99: {np.percentile(e2e_arr,50):.2f} / {np.percentile(e2e_arr,90):.2f} / {np.percentile(e2e_arr,99):.2f} ms")
                    print("\n".join(lines))
                else:
                    print("[Per-Query Timing]")
                    print(f"Videos compared per query: {num_videos}")
                    if 'avg' in pq_stats:
                        print(f"Similarity (all videos) avg: {pq_stats['avg']:.2f} ms")
                    if enc_avg is not None:
                        print(f"Encoding + normalize avg: {enc_avg:.2f} ms")
                    if 'avg' in pq_stats and enc_avg is not None:
                        print(f"Total per-query avg (enc + sim): {enc_avg + pq_stats['avg']:.2f} ms")
                    if all(k in pq_stats for k in ('p50', 'p90', 'p99')):
                        print(f"Similarity per-query p50/p90/p99: {pq_stats['p50']:.2f} / {pq_stats['p90']:.2f} / {pq_stats['p99']:.2f} ms")
                    if sort_stats and isinstance(sort_stats, dict):
                        label = 'top-k' if timing_topk > 0 else 'argsort'
                        print(f"{label.capitalize()} avg: {sort_stats.get('avg', 0.0):.2f} ms")
                        if all(k in sort_stats for k in ('p50', 'p90', 'p99')):
                            print(f"{label.capitalize()} p50/p90/p99: {sort_stats['p50']:.2f} / {sort_stats['p90']:.2f} / {sort_stats['p99']:.2f} ms")
                    if timing_topk > 0:
                        print(f"(Per-query timing uses top-k k={timing_topk})")
                    if dev:
                        print(f"Timing device: {dev.upper()}")
                    if e2e_arr is not None and e2e_arr.size > 0:
                        print(f"Similarity (end-to-end, incl argsort) avg: {e2e_arr.mean():.2f} ms")
                        print(f"Similarity (end-to-end) p50/p90/p99: {np.percentile(e2e_arr,50):.2f} / {np.percentile(e2e_arr,90):.2f} / {np.percentile(e2e_arr,99):.2f} ms")
                    print()

            if not timing_quiet:
                print("[Detailed Search Timing]")
        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]


    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info, timing_quiet=False):
        measure = bool(self.cfg.get("measure_search", False))
        timing_topk = int(self.cfg.get("timing_topk", 0))
        timing_on_gpu = bool(self.cfg.get("timing_on_gpu", False))
        warmup_batches = int(self.cfg.get("timing_warmup_batches", 1))
        per_query_ms = []
        per_clip_ms = []
        per_frame_ms = []
        argsort_ms = []
        encoding_times = []
        similarity_start_times = []
        similarity_end_times = []

        query_metas, score_sum_list = [], []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader), disable=timing_quiet):

            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]

            if measure:
                clip_s, frame_s, timing = model.get_pred_from_raw_query(
                    query_feat, query_mask, None,
                    ctx_info["video_proposal_feat"], ctx_info["video_feat"],
                    clip_mask=ctx_info.get("clip_mask"),
                    return_query_feats=False, return_timing=True
                )
                bs = query_feat.size(0)

                collect = (idx >= warmup_batches)
                if collect:
                    per_query_ms.extend([timing["search_total_ms"] / bs] * bs)
                    per_clip_ms.append(timing["clip_ms"])
                    per_frame_ms.append(timing["frame_ms"])
                    encoding_times.extend([timing["encoding_normalize_ms"] / bs] * bs)
                    if "similarity_start_time" in timing:
                        similarity_start_times.extend([timing["similarity_start_time"]] * bs)

            else:
                clip_s, frame_s = model.get_pred_from_raw_query(
                    query_feat, query_mask, None,
                    ctx_info["video_proposal_feat"], ctx_info["video_feat"],
                    clip_mask=ctx_info.get("clip_mask"),
                    return_query_feats=False, return_timing=False
                )

            clip_s.mul_(self.cfg['clip_scale_w'])
            frame_s.mul_(self.cfg['frame_scale_w'])
            clip_s.add_(frame_s)

            if measure:
                try:
                    _sync()
                except Exception:
                    pass
                if timing_on_gpu:
                    try:
                        _sync()
                    except Exception:
                        pass
                    t_sort0 = time.perf_counter()
                    if timing_topk and timing_topk > 0 and timing_topk < clip_s.size(1):
                        _ = torch.topk(clip_s, k=timing_topk, dim=1, largest=True, sorted=True)
                    else:
                        _ = torch.argsort(clip_s, dim=1)
                    try:
                        _sync()
                    except Exception:
                        pass
                    t_sort1 = time.perf_counter()
                    t_end = t_sort1
                    if 'collect' in locals() and collect:
                        bs_local = clip_s.size(0)
                        sort_ms = (t_sort1 - t_sort0) * 1000.0
                        per_q_sort = sort_ms / max(1, bs_local)
                        argsort_ms.extend([per_q_sort] * bs_local)
                        similarity_end_times.extend([t_end] * bs_local)
                    scores_cpu = clip_s.detach().cpu()
                else:
                    scores_cpu = clip_s.detach().cpu()
                    t_sort0 = time.perf_counter()
                    if timing_topk and timing_topk > 0 and timing_topk < scores_cpu.size(1):
                        _ = torch.topk(scores_cpu, k=timing_topk, dim=1, largest=True, sorted=True)
                    else:
                        _ = torch.argsort(scores_cpu, dim=1)
                    try:
                        _sync()
                    except Exception:
                        pass
                    t_sort1 = time.perf_counter()
                    t_end = t_sort1
                    if 'collect' in locals() and collect:
                        bs_local = scores_cpu.size(0)
                        sort_ms = (t_sort1 - t_sort0) * 1000.0
                        per_q_sort = sort_ms / max(1, bs_local)
                        argsort_ms.extend([per_q_sort] * bs_local)
                        similarity_end_times.extend([t_end] * bs_local)
            else:
                scores_cpu = clip_s.detach().cpu()

            score_sum_list.append(scores_cpu)
            del clip_s, frame_s

            if idx % 64 == 0:
                torch.cuda.empty_cache()

        score_sum = torch.cat(score_sum_list, dim=0)

        report = None
        if measure and per_query_ms:
            arr = np.array(per_query_ms, dtype=np.float64)
            report = {
                "per_query_ms": {
                    "avg": float(arr.mean()),
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                    "p99": float(np.percentile(arr, 99)),
                    "count": int(arr.size),
                },
                "batch_sections_ms": {
                    "clip_avg": float(np.mean(per_clip_ms)) if per_clip_ms else 0.0,
                    "frame_avg": float(np.mean(per_frame_ms)) if per_frame_ms else 0.0,
                }
            }

            if encoding_times:
                enc_arr = np.array(encoding_times, dtype=np.float64)
                report["encoding_normalize_avg_ms"] = float(enc_arr.mean())

            if similarity_start_times:
                report["similarity_start_times"] = similarity_start_times
            if similarity_end_times:
                report["similarity_end_times"] = similarity_end_times
            report["timing_topk"] = int(timing_topk)
            report["timing_device"] = "gpu" if timing_on_gpu else "cpu"
            if argsort_ms:
                s_arr = np.array(argsort_ms, dtype=np.float64)
                report["argsort_ms"] = {
                    "avg": float(s_arr.mean()),
                    "p50": float(np.percentile(s_arr, 50)),
                    "p90": float(np.percentile(s_arr, 90)),
                    "p99": float(np.percentile(s_arr, 99)),
                    "count": int(s_arr.size),
                }

        return score_sum, query_metas, report


    def compute_context_info(self, model, context_dataloader, timing_quiet=False):
        n_total_vid = len(context_dataloader.dataset)
        metas = []
        start = 0
        vid_prop_bank = None
        vid_frame_bank = None
        clip_mask_bank = None
        video_mask_cat = None

        if not timing_quiet:
            log_cuda("context: before cat")
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                            total=len(context_dataloader), disable=timing_quiet):
            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            last_level_segments = batch[3] if len(batch) >= 6 else None

            _frame_feat, _video_proposal_feat, _clip_mask = model.encode_context(
                clip_video_feat_, frame_video_feat_, frame_mask_, last_level_segments=last_level_segments
            )

            bsz = _frame_feat.size(0)
            K = _video_proposal_feat.size(1)
            H = _video_proposal_feat.size(2)
            device = _video_proposal_feat.device
            dtype = _video_proposal_feat.dtype

            if vid_prop_bank is None:
                vid_prop_bank = torch.empty((n_total_vid, K, H), device=device, dtype=dtype)
                vid_frame_bank = torch.empty((n_total_vid, H), device=device, dtype=dtype)
                clip_mask_bank = torch.empty((n_total_vid, K), device=device, dtype=torch.float32)

            end = start + bsz
            vid_prop_bank[start:end, :, :] = _video_proposal_feat
            vid_frame_bank[start:end, :] = _frame_feat
            if _clip_mask is None:
                clip_mask_bank[start:end, :K] = 1.0
            else:
                clip_mask_bank[start:end, :K] = _clip_mask
            start = end

        if not timing_quiet:
            log_cuda("context: after cat")

        pre_norm = bool(self.cfg.get('pre_normalize_context', False))
        if pre_norm:
            eps = 1e-12
            denom = torch.linalg.norm(vid_prop_bank, dim=-1, keepdim=True).clamp_min_(eps)
            vid_prop_bank.div_(denom)
            denom2 = torch.linalg.norm(vid_frame_bank, dim=-1, keepdim=True).clamp_min_(eps)
            vid_frame_bank.div_(denom2)
            try:
                if hasattr(model, 'config'):
                    setattr(model.config, 'context_already_normalized', True)
            except Exception:
                pass
            if not timing_quiet:
                log_cuda("context: after normalize")

        torch.cuda.empty_cache()
        if not timing_quiet:
            log_cuda("context: after cleanup")

        return dict(
            video_metas=metas,
            video_proposal_feat=vid_prop_bank,
            video_feat=vid_frame_bank,
            video_mask=video_mask_cat,
            clip_mask=clip_mask_bank
        )
