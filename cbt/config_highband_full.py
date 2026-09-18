from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    # 是否强制要求 CUDA；完整 FlowHigh + vocoder 训练建议保持 True。
    require_cuda: bool = True
    # 全局随机种子；用于模型初始化、数据划分等可复现实验。
    random_seed: int = 104
    # 启动训练时是否打印模型结构摘要；模型大时可以改成 False 让启动更清爽。
    show_model_summary: bool = True


@dataclass
class DataConfig:
    # VCTK 音频根目录；AudioDataset 会递归读取 speaker 子文件夹里的音频。
    data_path: str = ""
    # 读取的音频后缀；当前 VCTK silence trimmed 数据一般是 .flac。
    audio_extension: str = ".flac"

    # 完整数据集中用于训练的比例。
    train_split: float = 0.70
    # 完整数据集中用于测试/验证的比例。
    test_split: float = 0.30
    # 70/30 数据划分的随机种子；固定后每次划分一致。
    random_split_seed: int = 53

    # 目标高分辨率音频采样率，也是 mel/vocoder 的目标采样率。
    samplingrate: int = 48000
    # waveform 保存/归一化时使用的 int16 最大值。
    max_wav_value: float = 32767.0
    # STFT FFT 点数；影响 mel 频率分辨率。
    n_fft: int = 2048
    # STFT hop size；48k 下 480 表示 10ms 帧移。
    hop_length: int = 480
    # STFT window length；通常与 n_fft 一致。
    win_length: int = 2048
    # mel 频带数；需要和 BigVGAN vocoder 配置一致。
    n_mel_channels: int = 256
    # mel 最低频率。
    mel_fmin: int = 20
    # mel 最高频率；48k 目标采样率下 Nyquist 为 24k。
    mel_fmax: int = 24000

    # 训练时随机模拟 LR 输入采样率的下限；cutoff_hz = random_sr / 2。
    downsample_min: int = 4000
    # 训练时随机模拟 LR 输入采样率的上限；cutoff_hz = random_sr / 2。
    downsample_max: int = 32000
    # 随机采样率候选列表的步长，例如 4000, 5000, ..., 32000。
    downsample_step: int = 1000
    # 降采样/升采样实现方式；当前保持 scipy 路线。
    downsampling_method: str = "scipy"


@dataclass
class ModelConfig:
    # 模型名字；用于日志、checkpoint 标识。
    modelname: str = "FLowHigh"
    # 主干结构类型；当前使用 transformer。
    architecture: str = "transformer"
    # Transformer hidden dimension。
    dim: int = 1024
    # Transformer 层数。
    n_layers: int = 2
    # 多头注意力 head 数量。
    n_heads: int = 16
    # 每个 attention head 的维度。
    dim_head: int = 64
    # 模型输入通道数；高频残差实验为 z_t、M_up、mask_hf 三路输入。
    input_channels: int = 3

    # CFM 路径/方法名；保持和实验 wrapper 支持的方法一致。
    cfm_path: str = "independent_cfm_adaptive"
    # CFM 中最小噪声/数值稳定项。
    sigma: float = 1e-4
    # 是否启用 high-band residual flow；False 会回退原始 full mel 预测逻辑。
    use_highband_residual_flow: bool = True
    # 高频 soft mask 的 sigmoid 平滑宽度，单位 Hz。
    highband_mask_softness_hz: float = 200.0
    # High-band residual CFM initial noise scale; does not change sigma.
    residual_noise_scale: float = 0.1
    # Optional inference/validation mel seam smoothing around cutoff only.
    seam_smoothing_enabled: bool = True
    seam_smoothing_kernel_size: int = 3
    seam_smoothing_bins: int = 4
    # 是否把 high-band mask 作为额外 condition channel 拼到模型输入。
    condition_with_hf_mask: bool = True
    # 是否把 cutoff ratio 加入 time embedding 条件。
    condition_with_cutoff_embedding: bool = True
    # Optional AdaLN-Zero Transformer conditioning. Defaults off for old experiments.
    use_adaln_zero: bool = False
    adaln_zero_init: bool = True
    adaln_zero_gate_log: bool = True
    # 训练目标类型；当前为 mel 高频残差。
    target_type: str = "mel_highband_residual"

    # vocoder 类型；当前使用 BigVGAN。
    vocoder: str = "bigvgan"
    # BigVGAN checkpoint 路径。
    vocoderpath: str = "vocoder/BIGVGAN/checkpoint/g_48_00850000"
    # BigVGAN config json 路径。
    vocoderconfigpath: str = "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"


@dataclass
class LoggingConfig:
    # TensorBoard 日志目录；相对路径会解析到当前实验目录下。
    log_dir: str = "log/full"
    # 训练阶段 TensorBoard/accelerator 每多少个 optimizer step 记录一次。
    log_every: int = 1000
    # tqdm / trainer 显示用的 run 名称。
    run_name: str = "FLowHigh_highband_full"
    # accelerate tracker project 名称。
    tracker_project_name: str = "flowhigh_highband_full"
    # 文本日志文件名；每个 epoch 写一行关键指标。
    metrics_filename: str = "metrics.txt"
    # TensorBoard 训练曲线白名单；只记录这里列出的训练 tag。
    tensorboard_train_keys: list[str] = field(default_factory=lambda: [
        # mask 后的高频 flow loss。
        "train/loss_flow_masked",
        "train/loss_edge_continuity",
        "train/loss_freq_pred",
        "train/loss_freq_grad",
        "train/adaln_gate_msa_mean",
        "train/adaln_gate_msa_abs_mean",
        "train/adaln_gate_ffn_mean",
        "train/adaln_gate_ffn_abs_mean",
        "train/loss_total_hfboost",
        # CFM 总 loss；当前 high-band 模式下与主训练 loss 对齐。
        "training/cfm_loss",
        # 梯度裁剪前/裁剪时返回的梯度范数，用于观察训练稳定性。
        "training/grad_norm",
        # 当前学习率。
        "training/lr",
    ])
    # TensorBoard 验证曲线白名单；每个 epoch 后写一次。
    tensorboard_valid_keys: list[str] = field(default_factory=lambda: [
        # 8k/12k/16k 三个固定 cutoff 的平均 masked flow loss。
        "valid_avg/loss_flow_masked",
        "valid_avg/loss_edge_continuity",
        "valid_avg/loss_freq_pred",
        "valid_avg/loss_freq_grad",
        "valid_avg/adaln_gate_msa_mean",
        "valid_avg/adaln_gate_msa_abs_mean",
        "valid_avg/adaln_gate_ffn_mean",
        "valid_avg/adaln_gate_ffn_abs_mean",
        "valid_avg/loss_total_hfboost",
        # 固定 8k 输入采样率验证 loss。
        "valid_8k/loss_flow_masked",
        "valid_8k/loss_edge_continuity",
        "valid_8k/loss_freq_pred",
        "valid_8k/loss_freq_grad",
        # 固定 12k 输入采样率验证 loss。
        "valid_12k/loss_flow_masked",
        "valid_12k/loss_edge_continuity",
        "valid_12k/loss_freq_pred",
        "valid_12k/loss_freq_grad",
        # 固定 16k 输入采样率验证 loss。
        "valid_16k/loss_flow_masked",
        "valid_16k/loss_edge_continuity",
        "valid_16k/loss_freq_pred",
        "valid_16k/loss_freq_grad",
        # hard lowband 区域最终 mel 与 M_up 的差异；应接近 0。
        "valid_avg/hard_lowband_delta",
        # highband 区域最终 mel 与 M_up 的差异；应明显大于低频 delta。
        "valid_avg/highband_delta",
        # 高频预测幅度与目标幅度比例，用于观察预测是否塌缩或过大。
        "valid_avg/pred_target_l1_ratio",
    ])
    # metrics.txt 每个 epoch 记录的字段白名单。
    txt_epoch_keys: list[str] = field(default_factory=lambda: [
        # epoch 内训练 masked flow loss 均值。
        "train/loss_flow_masked",
        "train/loss_edge_continuity",
        "train/loss_freq_pred",
        "train/loss_freq_grad",
        "train/adaln_gate_msa_mean",
        "train/adaln_gate_msa_abs_mean",
        "train/adaln_gate_ffn_mean",
        "train/adaln_gate_ffn_abs_mean",
        "train/loss_total_hfboost",
        # epoch 内 CFM loss 均值。
        "training/cfm_loss",
        # epoch 内梯度范数均值。
        "training/grad_norm",
        # epoch 内学习率均值。
        "training/lr",
        # 固定 cutoff 平均验证 loss。
        "valid_avg/loss_flow_masked",
        "valid_avg/loss_edge_continuity",
        "valid_avg/loss_freq_pred",
        "valid_avg/loss_freq_grad",
        "valid_avg/adaln_gate_msa_mean",
        "valid_avg/adaln_gate_msa_abs_mean",
        "valid_avg/adaln_gate_ffn_mean",
        "valid_avg/adaln_gate_ffn_abs_mean",
        "valid_avg/loss_total_hfboost",
        # 8k 固定 cutoff 验证 loss。
        "valid_8k/loss_flow_masked",
        "valid_8k/loss_edge_continuity",
        "valid_8k/loss_freq_pred",
        "valid_8k/loss_freq_grad",
        # 12k 固定 cutoff 验证 loss。
        "valid_12k/loss_flow_masked",
        "valid_12k/loss_edge_continuity",
        "valid_12k/loss_freq_pred",
        "valid_12k/loss_freq_grad",
        # 16k 固定 cutoff 验证 loss。
        "valid_16k/loss_flow_masked",
        "valid_16k/loss_edge_continuity",
        "valid_16k/loss_freq_pred",
        "valid_16k/loss_freq_grad",
        # hard lowband delta。
        "valid_avg/hard_lowband_delta",
        # highband delta。
        "valid_avg/highband_delta",
        # 高频预测/目标 L1 比例。
        "valid_avg/pred_target_l1_ratio",
        # 平均 log spectral distance。
        "valid_avg/LSD",
        # 低频段 log spectral distance。
        "valid_avg/LSD_LF",
        # 高频段 log spectral distance。
        "valid_avg/LSD_HF",
        # ViSQOL 分数；没有可执行文件时为 nan。
        "valid_avg/ViSQOL",
        # real-time factor，越低推理越快。
        "valid_avg/RTF",
        # validation sampling 的 function evaluations 数。
        "valid_avg/NFEs",
        # best checkpoint 监控的指标名。
        "best_metric_name",
        # 当前历史最好指标值。
        "best_metric_value",
        # 历史最好指标所在 epoch。
        "best_epoch",
        # 历史最好指标所在 optimizer step。
        "best_step",
        # best checkpoint 保存路径。
        "checkpoint_path",
    ])


@dataclass
class CheckpointConfig:
    # checkpoint 保存目录；相对路径会解析到当前实验目录下。
    save_dir: str = "model/full"
    # 每多少个 epoch 保存一次常规 checkpoint。
    save_model_every_epochs: int = 10
    # 是否保存历史最优 checkpoint。
    save_best_checkpoint: bool = True
    # 用于判断历史最优的验证指标。
    best_metric_name: str = "valid_avg/loss_flow_masked"
    # best_metric_name 的优化方向；loss 用 min，分数类指标可用 max。
    best_metric_mode: str = "min"
    # 历史最优 checkpoint 的固定文件名。
    best_checkpoint_name: str = "best.pt"
    # 历史最优 checkpoint 的带 epoch/step 备份文件名模板。
    best_checkpoint_template: str = "best_epoch_{epoch:04d}_step_{step:06d}.pt"
    # 常规 epoch checkpoint 文件名模板。
    epoch_checkpoint_template: str = "FLowHigh.epoch{epoch}.step{step}.pt"
    # 训练结束最终 checkpoint 文件名模板。
    final_checkpoint_template: str = "FLowHigh.final.{step}.pt"


@dataclass
class ValidationConfig:
    # 是否每个 epoch 结束后执行固定 cutoff validation。
    validate_on_epoch_end: bool = True
    # 额外按 step 间隔验证；0 表示关闭，仅按 epoch 验证。
    val_step_interval: int = 0
    # 固定验证用的模拟输入采样率列表；cutoff_hz = fixed_sr / 2。
    fixed_validation_sample_rates: list[int] = field(default_factory=lambda: [8000, 12000, 16000])

    # 每次验证使用多少条测试音频；<=0 表示使用完整 30% test split。
    validation_num_samples: int = 128
    # validation dataloader batch size；音频指标逐条更稳，默认 1。
    validation_batch_size: int = 1
    # validation 采样步数；one-step Euler 时为 1。
    validation_time_steps: int = 1
    # 是否缓存固定 validation batch；full test 建议 False。
    cache_validation_batch: bool = False

    # 是否保存少量 validation 音频样例。
    validation_save_audio: bool = True
    # 每个 epoch 保存多少条 validation 音频样例。
    validation_save_audio_num: int = 4
    # validation 音频样例保存的子目录名。
    validation_audio_dirname: str = "highband_validation"

    # 是否计算 LSD / LSD_LF / LSD_HF。
    compute_lsd_metrics: bool = True
    # 是否尝试计算 ViSQOL；找不到 visqol_bin 时记录 nan。
    compute_visqol: bool = False
    # ViSQOL 可执行文件名或绝对路径；例如 "visqol" 或 "/path/to/visqol"。
    visqol_bin: str = "visqol"


@dataclass
class TrainConfig:
    # 总训练 epoch 数。
    num_epochs: int = 100
    # 如果不为 None，则按总 optimizer step 数训练；full 训练建议保持 None，以 epoch 为主。
    num_train_steps: int | None = None
    # 单卡/单进程 batch size。
    batchsize: int = 8
    # 梯度累计步数；有效 batch = batchsize * grad_accum_every * GPU 数。
    grad_accum_every: int = 2

    # Adam/优化器主学习率。
    lr: float = 1e-4
    # warmup 起始学习率。
    initial_lr: float = 1e-5
    # warmup optimizer step 数；0 表示不 warmup。
    n_warmup_steps: int = 1000
    # weight decay。
    weight_decay: float = 0.0
    # 梯度裁剪阈值；None 可关闭裁剪。
    max_grad_norm: float = 0.5
    # 是否使用旧的 weighted loss；high-band residual 第一版保持 False。
    weighted_loss: bool = False
    # 高频权重策略；full 训练默认也可用，旧 checkpoint 构建不受影响。
    use_hf_frequency_weight: bool = True
    hf_weight_min: float = 1.0
    hf_weight_max: float = 1.5
    hf_weight_mode: str = "mel_capped"
    hf_weight_cap_hz_above_cutoff: float = 4000.0
    loss_hf_l1_weight: float = 0.05
    # cutoff 边界能量连续性 loss。
    edge_continuity_loss_weight: float = 0.05
    edge_continuity_bins: int = 4
    # Optional final high-band mel auxiliary losses. Defaults off for old experiments.
    use_freq_prediction_loss: bool = False
    freq_prediction_loss_weight: float = 0.0
    use_freq_gradient_loss: bool = False
    freq_gradient_loss_weight: float = 0.0

    # accelerate 是否 split batches；通常保持 False。
    split_batches: bool = False
    # train/valid dataloader 是否丢弃最后不足 batch 的样本。
    drop_last: bool = False
    # DDP 是否查找未使用参数；当前模型没有 unused parameters，保持 False 更高效。
    find_unused_parameters: bool = False
    # 启动时是否清空已有结果目录；False 防止误删历史 checkpoint。
    force_clear_prev_results: bool | None = False
    # 旧结果保存间隔；当前 full 路线不使用，保持 0。
    save_results_every: int = 0


@dataclass
class InferenceConfig:
    # 推理输入路径；可指向文件或目录。
    input_path: str = ""
    # 推理输出目录。
    output_path: str = "inference_output/full"
    # 默认推理 checkpoint 路径。
    model_path: str = "model/full/best.pt"
    # 推理时 LR -> target_sr 的上采样方法。
    up_sampling_method: str = "scipy"
    # 推理采样步数。
    time_step: int = 4
    # ODE/采样方法名。
    ode_method: str = "midpoint"


@dataclass
class FullHighbandConfig:
    # 运行环境相关配置。
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # 数据路径、mel 参数、采样率模拟配置。
    data: DataConfig = field(default_factory=DataConfig)
    # 模型结构、CFM、高频残差实验配置。
    model: ModelConfig = field(default_factory=ModelConfig)
    # 训练循环、优化器、DDP/dataloader 配置。
    train: TrainConfig = field(default_factory=TrainConfig)
    # TensorBoard、accelerator tracker、metrics.txt 配置。
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # checkpoint 保存策略配置。
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    # 固定 cutoff validation 和测试指标配置。
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    # 推理脚本默认配置。
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# 训练脚本统一从这个对象读取配置；不要在命令行或脚本里散落超参数。
config = FullHighbandConfig()
