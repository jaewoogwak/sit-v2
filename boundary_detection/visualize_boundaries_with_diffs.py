import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize boundary indices with frame grids and diff graph."
    )
    parser.add_argument("--boundaries_json", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--level",
        type=str,
        choices=("fine", "coarse"),
        default="fine",
        help="If boundaries file contains multiple levels, choose which to render.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Sample every N frames.")
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--tile_width", type=int, default=160)
    parser.add_argument("--tile_height", type=int, default=90)
    parser.add_argument("--draw_index", action="store_true")
    parser.add_argument("--graph_height", type=int, default=240)
    parser.add_argument("--graph_padding", type=int, default=24)
    parser.add_argument("--rows_per_page", type=int, default=5)
    return parser.parse_args()


def _resolve_video_dir(frames_root, video_id):
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


def _load_entry(boundaries_json, video_id, level):
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

    if not isinstance(entry, dict):
        return entry or [], []

    segment_edges = entry.get("segment_edges", [])
    if "fine" in entry or "coarse" in entry:
        entry = entry.get(level, {})
    peaks = entry.get("peaks", [])
    diffs = entry.get("diffs", [])

    if level == "coarse" and segment_edges and diffs and len(diffs) == max(0, len(segment_edges) - 1):
        total_len = int(segment_edges[-1]) if segment_edges else 0
        frame_diffs = [0.0] * max(0, total_len)
        for i in range(len(segment_edges) - 1):
            start = int(segment_edges[i])
            end = int(segment_edges[i + 1])
            val = float(diffs[i])
            start = max(0, start)
            end = min(total_len, end)
            if start < end:
                frame_diffs[start:end] = [val] * (end - start)
        diffs = frame_diffs
    return peaks, diffs


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


def _build_row(
    row_frames,
    video_dir,
    cols,
    tile_size,
    boundary_frame_numbers,
    draw_index,
):
    tile_w, tile_h = tile_size
    row = Image.new("RGB", (cols * tile_w, tile_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(row)
    font = None
    if draw_index:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for c, (frame_number, name) in enumerate(row_frames):
        path = os.path.join(video_dir, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        img = img.resize((tile_w, tile_h))
        x0 = c * tile_w
        row.paste(img, (x0, 0))
        if frame_number in boundary_frame_numbers:
            draw.rectangle([x0, 0, x0 + tile_w - 1, tile_h - 1], outline=(255, 0, 0), width=3)
        if draw_index and font is not None:
            draw.text((x0 + 3, 3), str(frame_number), fill=(255, 255, 0), font=font)
    return row


def _build_diff_row_graph(
    diffs,
    row_frames,
    cols,
    tile_width,
    height,
    padding,
    boundary_frame_numbers,
):
    width = cols * tile_width
    graph = Image.new("RGB", (width, height), color=(12, 12, 12))
    draw = ImageDraw.Draw(graph)
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if not diffs:
        draw.text((padding, padding), "no diffs", fill=(200, 200, 200))
        return graph

    max_val = max(diffs)
    if max_val <= 0:
        max_val = 1.0

    plot_w = max(1, width - 2 * padding)
    plot_h = max(1, height - 2 * padding)
    draw.rectangle([padding, padding, padding + plot_w, padding + plot_h], outline=(80, 80, 80), width=1)

    if font is not None:
        max_y = padding
        mid_y = padding + plot_h / 2.0
        min_y = padding + plot_h
        draw.text((2, max_y - 6), f"{max_val:.2f}", fill=(200, 200, 200), font=font)
        draw.text((2, mid_y - 6), f"{(max_val * 0.5):.2f}", fill=(200, 200, 200), font=font)
        draw.text((2, min_y - 6), "0.00", fill=(200, 200, 200), font=font)

    points = []
    for c, (frame_number, _) in enumerate(row_frames[:-1]):
        idx = frame_number - 1
        if idx < 0 or idx >= len(diffs):
            val = 0.0
        else:
            val = diffs[idx]
        x = padding + ((c + 0.5) / float(max(1, cols - 1))) * plot_w
        y = padding + (1.0 - (val / max_val)) * plot_h
        points.append((x, y))

    if len(points) >= 2:
        draw.line(points, fill=(64, 200, 255), width=2)
    elif points:
        x, y = points[0]
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(64, 200, 255))

    for c, (frame_number, _) in enumerate(row_frames[:-1]):
        if frame_number not in boundary_frame_numbers:
            continue
        x = padding + ((c + 0.5) / float(max(1, cols - 1))) * plot_w
        draw.line([(x, padding), (x, padding + plot_h)], fill=(255, 80, 80), width=2)

    return graph


def main():
    args = parse_args()
    video_dir = _resolve_video_dir(args.frames_root, args.video_id)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"video_dir not found: {video_dir}")

    peaks, diffs = _load_entry(args.boundaries_json, args.video_id, args.level)
    print(f"boundaries[{args.video_id}].peaks: {peaks}")
    print(f"boundaries[{args.video_id}].diffs: {len(diffs)} values")
    boundary_frame_numbers = set()
    for b in peaks:
        if b < 0:
            continue
        boundary_frame_numbers.add(int(b) + 1)

    sampled = _sample_frames(video_dir, args.stride, boundary_frame_numbers)
    rows = int(math.ceil(len(sampled) / float(args.cols))) if sampled else 1
    tile_w, tile_h = args.tile_width, args.tile_height
    row_width = args.cols * tile_w
    rows_per_page = max(1, args.rows_per_page)
    pages = int(math.ceil(rows / float(rows_per_page))) if rows else 1

    base, ext = os.path.splitext(args.output_path)
    if not ext:
        ext = ".jpg"

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    for page_idx in range(pages):
        start_row = page_idx * rows_per_page
        end_row = min(rows, start_row + rows_per_page)
        rows_on_page = max(1, end_row - start_row)
        combined_height = rows_on_page * (tile_h + args.graph_height)
        combined = Image.new("RGB", (row_width, combined_height), color=(10, 10, 10))

        for r in range(start_row, end_row):
            start = r * args.cols
            end = start + args.cols
            row_frames = sampled[start:end]
            if not row_frames:
                continue
            row_img = _build_row(
                row_frames,
                video_dir,
                args.cols,
                (tile_w, tile_h),
                boundary_frame_numbers,
                args.draw_index,
            )
            graph = _build_diff_row_graph(
                diffs,
                row_frames,
                args.cols,
                tile_w,
                args.graph_height,
                args.graph_padding,
                boundary_frame_numbers,
            )
            y0 = (r - start_row) * (tile_h + args.graph_height)
            combined.paste(row_img, (0, y0))
            combined.paste(graph, (0, y0 + tile_h))

        if pages == 1:
            out_path = args.output_path
        else:
            out_path = f"{base}_page_{page_idx + 1}{ext}"
        combined.save(out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
