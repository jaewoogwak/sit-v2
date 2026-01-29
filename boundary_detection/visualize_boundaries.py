import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize boundary indices on frame grids.")
    parser.add_argument("--boundaries_json", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--level",
        type=str,
        default="fine",
        help="Select which boundaries to render (fine, coarse, levels, levelN).",
    )
    parser.add_argument(
        "--show_coarse",
        action="store_true",
        help="If set, overlay coarse boundaries in blue in addition to the selected level.",
    )
    parser.add_argument("--stride", type=int, default=3, help="Sample every N frames.")
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--tile_width", type=int, default=160)
    parser.add_argument("--tile_height", type=int, default=90)
    parser.add_argument("--draw_index", action="store_true")
    return parser.parse_args()


def _resolve_video_dir(frames_root, video_id):
    direct = os.path.join(frames_root, video_id)
    if os.path.isdir(direct):
        return direct
    show = video_id.split("_")[0]
    folder = f"{show}_frames"
    primary = os.path.join(frames_root, folder, video_id)
    if os.path.isdir(primary):
        return primary
    for name in os.listdir(frames_root):
        if not name.endswith("_frames"):
            continue
        candidate = os.path.join(frames_root, name, video_id)
        if os.path.isdir(candidate):
            return candidate
    return primary


def _parse_level_token(token):
    token = (token or "").strip().lower()
    if not token:
        return "", None
    if token == "levels":
        return "levels", None
    if token.startswith("level"):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            return "level", int(digits)
        return "level", None
    return token, None


def _peaks_from_level_entry(level_entry):
    if not isinstance(level_entry, dict):
        return []
    edges = level_entry.get("edges") or []
    if edges:
        peaks = []
        for edge in edges[1:-1]:
            try:
                peak = int(edge) - 1
            except Exception:
                continue
            if peak >= 0:
                peaks.append(peak)
        return peaks
    return level_entry.get("peaks", []) or []


def _peaks_by_level(levels):
    by_level = {}
    if not isinstance(levels, list):
        return by_level
    for level_entry in levels:
        if not isinstance(level_entry, dict):
            continue
        lvl = level_entry.get("level")
        if lvl is None:
            continue
        peaks = _peaks_from_level_entry(level_entry)
        if peaks:
            by_level.setdefault(lvl, []).extend(peaks)
    for lvl, peaks in list(by_level.items()):
        by_level[lvl] = sorted({int(p) for p in peaks if p is not None and int(p) >= 0})
    return by_level


def _peaks_from_levels(levels, level_num=None):
    peaks = []
    if not isinstance(levels, list):
        return peaks
    for level_entry in levels:
        if not isinstance(level_entry, dict):
            continue
        if level_num is not None and level_entry.get("level") != level_num:
            continue
        peaks.extend(_peaks_from_level_entry(level_entry))
    if peaks:
        peaks = sorted({int(p) for p in peaks if p is not None and int(p) >= 0})
    return peaks


def _load_boundaries(boundaries_json, video_id, level):
    if boundaries_json.lower().endswith(".jsonl"):
        with open(boundaries_json, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("video_id") == video_id:
                    entry = obj
                    break
            else:
                entry = {}
    else:
        with open(boundaries_json, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"loaded {len(data)} video ids from {boundaries_json}")
        entry = data.get(video_id, {})

    tokens = [t.strip() for t in level.split('+') if t.strip()] if '+' in level else [level]
    token, level_num = _parse_level_token(level)
    if isinstance(entry, dict):
        fine_peaks = entry.get("fine", {}).get("peaks", []) if "fine" in entry else []
        coarse_peaks = entry.get("coarse", {}).get("peaks", []) if "coarse" in entry else []
        levels_peaks = []
        levels_by_level = {}
        if "levels" in entry:
            levels_by_level = _peaks_by_level(entry.get("levels") or [])
            levels_peaks = _peaks_from_levels(entry.get("levels") or [], level_num if token == "level" else None)
        selected = []
        for tok in tokens:
            tname, tnum = _parse_level_token(tok)
            if tname == 'fine':
                selected.extend(fine_peaks)
            elif tname == 'coarse':
                selected.extend(coarse_peaks)
            elif tname in ('levels', 'level'):
                if "levels" in entry:
                    selected.extend(_peaks_from_levels(entry.get("levels") or [], tnum))
        if not selected:
            if token in ("levels", "level") and "levels" in entry:
                selected = levels_peaks
            elif "fine" in entry or "coarse" in entry:
                selected = coarse_peaks if token == "coarse" else fine_peaks
            else:
                selected = entry.get("peaks", [])
        if selected:
            selected = sorted({int(p) for p in selected if p is not None and int(p) >= 0})
        return selected, fine_peaks, coarse_peaks, levels_peaks, levels_by_level
    return entry or [], [], [], [], {}


def _parse_frame_number(name):
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return None


def _list_frames(video_dir):
    frames = []
    for name in os.listdir(video_dir):
        if not name.lower().endswith(".jpg"):
            continue
        frame_number = _parse_frame_number(name)
        if frame_number is None:
            continue
        frames.append((frame_number, name))
    frames.sort(key=lambda x: x[0])
    return frames


def _sample_frames(video_dir, stride, boundary_frame_numbers):
    frames = _list_frames(video_dir)
    if not frames:
        return []
    if stride <= 1:
        sampled = frames
    else:
        sampled = frames[::stride]
    frame_map = {num: name for num, name in frames}
    sampled_map = {num: name for num, name in sampled}
    for b in boundary_frame_numbers:
        if b in frame_map and b not in sampled_map:
            sampled.append((b, frame_map[b]))
    sampled.sort(key=lambda x: x[0])
    return sampled


def _build_grid(
    sampled,
    video_dir,
    cols,
    tile_size,
    boundary_frame_numbers,
    coarse_boundary_frame_numbers,
    levels_boundary_frame_numbers,
    level_color_map,
    draw_index,
    stride,
):
    rows = int(math.ceil(len(sampled) / float(cols))) if sampled else 1
    tile_w, tile_h = tile_size
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)
    font = None
    if draw_index:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for i, (frame_number, name) in enumerate(sampled):
        path = os.path.join(video_dir, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        img = img.resize((tile_w, tile_h))
        r = i // cols
        c = i % cols
        x0 = c * tile_w
        y0 = r * tile_h
        grid.paste(img, (x0, y0))
        has_coarse = frame_number in coarse_boundary_frame_numbers
        has_fine = frame_number in boundary_frame_numbers
        level_hits = []
        for lvl, frames in levels_boundary_frame_numbers.items():
            if frame_number in frames:
                level_hits.append(lvl)
        if has_coarse:
            draw.rectangle(
                [x0, y0, x0 + tile_w - 1, y0 + tile_h - 1],
                outline=(0, 128, 255),
                width=4,
            )
        inset = 3 if has_coarse else 0
        if level_hits:
            for lvl in sorted(level_hits):
                color = level_color_map.get(lvl, (0, 200, 0))
                draw.rectangle(
                    [x0 + inset, y0 + inset, x0 + tile_w - 1 - inset, y0 + tile_h - 1 - inset],
                    outline=color,
                    width=3,
                )
                inset += 3
        if has_fine:
            draw.rectangle(
                [x0 + inset, y0 + inset, x0 + tile_w - 1 - inset, y0 + tile_h - 1 - inset],
                outline=(255, 0, 0),
                width=2,
            )
        if draw_index and font is not None:
            draw.text((x0 + 3, y0 + 3), str(frame_number), fill=(255, 255, 0), font=font)
    return grid


def main():
    args = parse_args()
    video_dir = _resolve_video_dir(args.frames_root, args.video_id)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"video_dir not found: {video_dir}")

    boundaries, fine_peaks, coarse_peaks, levels_peaks, levels_by_level = _load_boundaries(
        args.boundaries_json, args.video_id, args.level
    )
    print(f"level={args.level}")
    print(f"boundaries[{args.video_id}]: {boundaries}")
    print(f"fine_boundaries[{args.video_id}]: {fine_peaks}")
    print(f"levels_boundaries[{args.video_id}]: {levels_peaks}")
    if levels_by_level:
        for lvl in sorted(levels_by_level.keys()):
            print(f"level-{lvl}_boundaries[{args.video_id}]: {levels_by_level[lvl]}")
    if args.show_coarse:
        print(f"coarse_boundaries[{args.video_id}]: {coarse_peaks}")
    boundary_frame_numbers = set()
    for b in fine_peaks:
        if b < 0:
            continue
        boundary_frame_numbers.add(int(b) + 1)
    coarse_boundary_frame_numbers = set()
    if args.show_coarse and coarse_peaks:
        for b in coarse_peaks:
            if b < 0:
                continue
            coarse_boundary_frame_numbers.add(int(b) + 1)

    levels_boundary_frame_numbers = {}
    for lvl, peaks in levels_by_level.items():
        frames = set()
        for b in peaks:
            if b < 0:
                continue
            frames.add(int(b) + 1)
        if frames:
            levels_boundary_frame_numbers[lvl] = frames

    union_levels = set()
    for frames in levels_boundary_frame_numbers.values():
        union_levels |= frames

    sampled = _sample_frames(
        video_dir,
        args.stride,
        boundary_frame_numbers | coarse_boundary_frame_numbers | union_levels,
    )
    level_colors = [
        ("green", (0, 200, 0)),
        ("orange", (255, 165, 0)),
        ("purple", (128, 0, 255)),
        ("cyan", (0, 200, 200)),
        ("magenta", (200, 0, 200)),
        ("yellow", (200, 200, 0)),
    ]
    level_color_map = {}
    level_color_name_map = {}
    for idx, lvl in enumerate(sorted(levels_boundary_frame_numbers.keys())):
        name, color = level_colors[idx % len(level_colors)]
        level_color_map[lvl] = color
        level_color_name_map[lvl] = name
    if level_color_map:
        for lvl in sorted(level_color_map.keys()):
            print(f"level-{lvl} color: {level_color_name_map[lvl]}")

    grid = _build_grid(
        sampled,
        video_dir,
        args.cols,
        (args.tile_width, args.tile_height),
        boundary_frame_numbers,
        coarse_boundary_frame_numbers,
        levels_boundary_frame_numbers,
        level_color_map,
        args.draw_index,
        args.stride,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    grid.save(args.output_path)
    print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()
