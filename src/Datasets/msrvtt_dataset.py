import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
import pickle
from .data_provider import (
    l2_normalize_np_array, uniform_feature_sampling, average_to_fixed_length,
    collate_train, collate_frame_val, collate_text_val
)
from Utils.basic_utils import MultiTextH5, IndexedTextH5


def getVideoId_msrvtt(cap_id):
    """MSRVTT 데이터셋용 video ID 추출"""
    # MSRVTT caption ID format: video123#caption456
    if '#' in cap_id:
        vid_id = cap_id.split('#')[0]
    else:
        # 예외 처리: video ID가 그대로인 경우
        vid_id = cap_id
    return vid_id


def read_video_ids_msrvtt(cap_file):
    """MSRVTT 데이터셋용 video ID 목록 추출"""
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            parts = line.strip().split(' ', 1)
            if len(parts) >= 2:
                cap_id, caption = parts
                video_id = getVideoId_msrvtt(cap_id)
                if video_id not in video_ids_list:
                    video_ids_list.append(video_id)
    return video_ids_list

def read_video_ids_msrvtt_limited(cap_file, limit):
    """MSRVTT: 캡션 파일에서 최대 limit 개의 고유 비디오 ID만 빠르게 읽기"""
    if limit is None or int(limit) <= 0:
        return read_video_ids_msrvtt(cap_file)
    n = int(limit)
    video_ids_list = []
    seen = set()
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader:
            if len(video_ids_list) >= n:
                break
            parts = line.strip().split(' ', 1)
            if not parts:
                continue
            cap_id = parts[0]
            video_id = getVideoId_msrvtt(cap_id)
            if video_id not in seen:
                seen.add(video_id)
                video_ids_list.append(video_id)
    return video_ids_list


class MSRVTTDataset4PRVR(data.Dataset):
    """
    MSRVTT 데이터셋을 위한 데이터로더
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, cfg, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    cap_id, caption = parts
                    video_id = getVideoId_msrvtt(cap_id)
                    self.captions[cap_id] = caption
                    self.cap_ids.append(cap_id)
                    if video_id not in self.video_ids:
                        self.video_ids.append(video_id)
                    if video_id in self.vid_caps:
                        self.vid_caps[video_id].append(cap_id)
                    else:
                        self.vid_caps[video_id] = []
                        self.vid_caps[video_id].append(cap_id)
        
        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        self.use_text_json = isinstance(text_feat_path, str) and text_feat_path.endswith('.json')
        if self.use_text_json:
            # Decide between manifest vs meta
            if os.path.basename(self.text_feat_path) == 'text_manifest.json':
                self.text_reader = MultiTextH5(self.text_feat_path)
            else:
                # meta json
                text_dir = os.path.join(os.path.dirname(os.path.dirname(self.text_feat_path)), 'TextData')
                self.text_reader = IndexedTextH5(text_dir, self.text_feat_path)

        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']

        self.open_file = False
        self.length = len(self.vid_caps)

    def __getitem__(self, index):
        if not self.open_file and not self.use_text_json:
            self.text_feat = h5py.File(self.text_feat_path, 'r')
            self.open_file = True

        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        # video features
        if self.video2frames and video_id in self.video2frames:
            frame_list = self.video2frames[video_id]
        else:
            # MSRVTT는 video 키 하나가 이미 (T,512) 전체 시퀀스를 담음
            frame_list = [video_id]

        frames_2d = None
        for frame_id in frame_list:
            try:
                vec = self.visual_feat.read_one(frame_id)  # vec: (T,512) 또는 (512,)
            except:
                vec = np.zeros((1, 512), dtype=np.float32)

            if isinstance(vec, np.ndarray):
                if vec.ndim == 2:           # (T,512)
                    frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])
                elif vec.ndim == 1:         # (512,)
                    vec = vec.reshape(1, -1)
                    frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])
                else:
                    # 예외적인 경우 안전 가드
                    vec = vec.reshape(-1, 512)
                    frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])
            else:
                # 타입 예외 가드
                vec = np.zeros((1, 512), dtype=np.float32)
                frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])

        if frames_2d is None or frames_2d.size == 0:
            frames_2d = np.zeros((1, 512), dtype=np.float32)

        # 이제 frames_2d는 확실히 (T,512)
        clip_video_feature = average_to_fixed_length(frames_2d, self.map_size)      # (map_size, 512)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)      # (1, map_size, 512)

        frame_video_feature = uniform_feature_sampling(frames_2d, self.max_ctx_len) # (max_ctx_len, 512)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # text features
        cap_tensors = []
        for cap_id in cap_ids:
            try:
                if self.use_text_json:
                    cap_feat = self.text_reader.get(cap_id)
                else:
                    cap_feat = self.text_feat[cap_id][...]
                
                # CLIP 텍스트 피쳐가 1차원인 경우 2차원으로 변환
                if cap_feat.ndim == 1:
                    cap_feat = cap_feat.reshape(1, -1)  # (512,) -> (1, 512)

                cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
                cap_tensors.append(cap_tensor)
            except:
                # 캡션을 찾을 수 없는 경우 기본값 사용
                cap_tensor = torch.zeros(1, 512)  # CLIP 텍스트 피쳐 차원
                cap_tensors.append(cap_tensor)

        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length


class MSRVTTVisDataSet(data.Dataset):
    """MSRVTT 비디오 데이터셋"""

    def __init__(self, visual_feat, video2frames, cfg, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = list(video2frames.keys()) if video2frames else []
        self.length = len(self.video_ids)
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        
        # video features
        if self.video2frames and video_id in self.video2frames:
            frame_list = self.video2frames[video_id]
        else:
            # MSRVTT는 video 키 하나가 이미 (T,512) 전체 시퀀스를 담음
            frame_list = [video_id]

        frames_2d = None
        for frame_id in frame_list:
            try:
                vec = self.visual_feat.read_one(frame_id)  # vec: (T,512) 또는 (512,)
            except:
                vec = np.zeros((1, 512), dtype=np.float32)

            if isinstance(vec, np.ndarray):
                if vec.ndim == 2:           # (T,512)
                    frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])
                elif vec.ndim == 1:         # (512,)
                    vec = vec.reshape(1, -1)
                    frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])
                else:
                    # 예외적인 경우 안전 가드
                    vec = vec.reshape(-1, 512)
                    frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])
            else:
                # 타입 예외 가드
                vec = np.zeros((1, 512), dtype=np.float32)
                frames_2d = vec if frames_2d is None else np.vstack([frames_2d, vec])

        if frames_2d is None or frames_2d.size == 0:
            frames_2d = np.zeros((1, 512), dtype=np.float32)

        # 이제 frames_2d는 확실히 (T,512)
        clip_video_feature = average_to_fixed_length(frames_2d, self.map_size)      # (map_size, 512)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)      # (1, map_size, 512)

        frame_video_feature = uniform_feature_sampling(frames_2d, self.max_ctx_len) # (max_ctx_len, 512)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)


        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class MSRVTTTxtDataSet(data.Dataset):
    """MSRVTT 텍스트 데이터셋"""

    def __init__(self, cap_file, text_feat_path, cfg):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    cap_id, caption = parts
                    self.captions[cap_id] = caption
                    self.cap_ids.append(cap_id)
        
        self.text_feat_path = text_feat_path
        self.use_text_json = isinstance(text_feat_path, str) and text_feat_path.endswith('.json')
        if self.use_text_json:
            if os.path.basename(self.text_feat_path) == 'text_manifest.json':
                self.text_reader = MultiTextH5(self.text_feat_path)
            else:
                text_dir = os.path.join(os.path.dirname(os.path.dirname(self.text_feat_path)), 'TextData')
                self.text_reader = IndexedTextH5(text_dir, self.text_feat_path)
        self.max_desc_len = cfg['max_desc_l']
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if not self.open_file and not self.use_text_json:
            self.text_feat = h5py.File(self.text_feat_path, 'r')
            self.open_file = True

        try:
            if self.use_text_json:
                cap_feat = self.text_reader.get(cap_id)
            else:
                cap_feat = self.text_feat[cap_id][...]
            
            # CLIP 텍스트 피쳐가 1차원인 경우 2차원으로 변환
            if cap_feat.ndim == 1:
                cap_feat = cap_feat.reshape(1, -1)  # (512,) -> (1, 512)

            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        except:
            # 캡션을 찾을 수 없는 경우 기본값 사용
            cap_tensor = torch.zeros(1, 512)  # CLIP 텍스트 피쳐 차원

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length
