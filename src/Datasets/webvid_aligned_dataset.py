import torch
import torch.utils.data as data
import numpy as np
from typing import List

from .data_provider import (
    l2_normalize_np_array, uniform_feature_sampling, average_to_fixed_length,
)


class WebVidAlignedDataset4PRVR(data.Dataset):
    """Aligned WebVid training dataset without reading huge caption files.

    Assumes 1:1 mapping: cap index == video index
    video_id: vid_{i}
    cap_id  : vid_{i}#cap_{i}
    """

    def __init__(self, visual_reader, text_reader, cfg, total_videos: int):
        self.visual_reader = visual_reader
        self.text_reader = text_reader
        self.cfg = cfg
        self.length = total_videos
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_id = f"vid_{index}"
        cap_id = f"{video_id}#cap_{index}"

        # Read video features
        vec = self.visual_reader.read_one(video_id)
        if isinstance(vec, np.ndarray):
            if vec.ndim == 2:
                frames_2d = vec
            elif vec.ndim == 1:
                frames_2d = vec.reshape(1, -1)
            else:
                frames_2d = vec.reshape(-1, vec.shape[-1])
        else:
            frames_2d = np.zeros((1, 512), dtype=np.float32)

        clip_video_feature = average_to_fixed_length(frames_2d, self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(frames_2d, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # Read text features
        try:
            cap_feat = self.text_reader.get(cap_id)
            if cap_feat.ndim == 1:
                cap_feat = cap_feat.reshape(1, -1)
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        except Exception:
            cap_tensor = torch.zeros(1, 512)

        cap_tensors: List[torch.Tensor] = [cap_tensor]
        return clip_video_feature, frame_video_feature, cap_tensors, index, [cap_id], video_id

