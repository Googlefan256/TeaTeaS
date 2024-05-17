from dataclasses import dataclass
from typing import Optional


@dataclass
class FastHparam:
    n_fft: int = 1024
    num_mels: int = 80
    sample_rate: int = 48000
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 8000
    train_file: str = "train_files.txt"
    audio_root: str = "audio"
    train_set: str = "trainset"
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_decay: float = 0.9999
    ckpt: Optional[str] = None
    save_steps: int = 2500
    batch_size: int = 16
    threads: int = 8
    shuffle: bool = True
    compile_mod: bool = False
    n_speakers: int = 0
    prenet: bool = True
    n_feats: int = 80
    spk_emb_dim: int = 0
    prior_loss = True
    encoder_embedding_dim: int = 192
    encoder_n_channels: int = 192
    sigma_min: float = 1e-4
    solver: str = "euler"
    duration_predictor_filter_channels_dp: int = 256
    duration_predictor_kernel_size: int = 3
    duration_predictor_p_dropout: float = 0.1
    encoder_filter_channels: int = 768
    encoder_n_heads: int = 2
    encoder_n_layers: int = 6
    encoder_kernel_size: int = 3
    encoder_p_dropout: float = 0.1
    decoder_channels: tuple = (256, 256)
    decoder_dropout: int = 0.05
    decoder_attention_head_dim: int = 64
    decoder_n_blocks: int = 1
    decoder_num_mid_blocks: int = 2
    decoder_num_heads: int = 2
    decoder_act_fn: str = "snakebeta"
    mel_mean: float = -7.070861339569092
    mel_std: float = 2.773611545562744
