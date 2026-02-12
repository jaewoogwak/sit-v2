def get_configs(dataset_name):
    if dataset_name in ['tvr']:
        import Configs.tvr as tvr
        return tvr.get_cfg_defaults()
    elif dataset_name in ['tvr_hd']:
        import Configs.tvr_hd as tvr_hd
        return tvr_hd.get_cfg_defaults()
    elif dataset_name in ['tvr_clip']:
        import Configs.tvr_clip as tvr_clip
        return tvr_clip.get_cfg_defaults()
    elif dataset_name in ['tvr_frames']:
        import Configs.tvr_frames as tvr_frames
        return tvr_frames.get_cfg_defaults()
    elif dataset_name in ['act_frames']:
        import Configs.act_frames as act_frames
        return act_frames.get_cfg_defaults()
    elif dataset_name in ['cha_frames']:
        import Configs.cha_frames as cha_frames
        return cha_frames.get_cfg_defaults()
    elif dataset_name in ['tvr_internvideo']:
        import Configs.tvr_internvideo as tvr_internvideo
        return tvr_internvideo.get_cfg_defaults()
    elif dataset_name in ['act']:
        import Configs.act as act
        return act.get_cfg_defaults()
    elif dataset_name in ['act_clip']:
        import Configs.act_clip as act_clip
        return act_clip.get_cfg_defaults()
    elif dataset_name in ['msrvtt']:
        import Configs.msrvtt as msrvtt
        return msrvtt.get_cfg_defaults()
    elif dataset_name in ['webvid_dummy']:
        import Configs.webvid_dummy as webvid_dummy
        return webvid_dummy.get_cfg_defaults()
    elif dataset_name in ['webvid_dummy_18']:
        import Configs.webvid_dummy_18 as webvid_dummy_18
        return webvid_dummy_18.get_cfg_defaults()
    elif dataset_name in ['webvid']:
        import Configs.webvid as webvid
        return webvid.get_cfg_defaults()
    elif dataset_name in ['webvid-10m', 'webvid_10m', 'webvid10m']:
        # Note: module name uses underscore; accept multiple CLI aliases
        import Configs.webvid_10m as webvid_10m
        return webvid_10m.get_cfg_defaults()
    if dataset_name in ['cha']:
        import Configs.cha as cha
        return cha.get_cfg_defaults()
