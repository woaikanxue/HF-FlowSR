from .base import FullHighbandConfig


config = FullHighbandConfig()

config.data.data_path = ""
config.data.train_path = None
config.data.valid_path = None
config.data.test_path = None
config.data.train_split = 0.98
config.data.test_split = 0.02

config.model.modelname = "CBTBridge-Full"
config.model.use_highband_residual_flow = True
config.model.target_type = "mel_highband_residual"
config.model.input_channels = 3
config.model.condition_with_hf_mask = True
config.model.condition_with_cutoff_embedding = True
config.model.use_interleaved_melconv = True
config.model.use_melconv_bridge = True
config.model.cbt_bridge_enabled = True
config.model.use_output_mel_adapter = True
config.model.use_adaln_zero = False
config.model.keep_hidden_dim = True

config.model.transformer0_depth = 1
config.model.transformer1_depth = 1
config.model.first_transformer_depth = 1
config.model.first_transformer_heads = 16
config.model.first_transformer_dim_head = 64
config.model.first_transformer_ff_mult = 4
config.model.second_transformer_depth = 1
config.model.second_transformer_heads = 8
config.model.second_transformer_dim_head = 64
config.model.second_transformer_ff_mult = 2

config.model.cbt_hidden_dim = 32
config.model.cbt_low_groups_hz = [
    [0, 1000],
    [1000, 2000],
    [2000, 4000],
    [4000, 6000],
    [6000, 8000],
    [8000, 12000],
]
config.model.cbt_high_groups_hz = [
    [4000, 6000],
    [6000, 8000],
    [8000, 12000],
    [12000, 16000],
    [16000, 24000],
]
config.model.cbt_use_event_gate = True
config.model.cbt_use_temporal_derivative = True
config.model.cbt_use_depthwise_temporal_conv = True
config.model.cbt_temporal_kernel = 5
config.model.cbt_zero_init = False
config.model.cbt_init_scale = 1e-3
config.model.cbt_dropout = 0.0

config.model.output_mel_adapter_channels = 64
config.model.output_mel_adapter_blocks = 2
config.model.output_mel_adapter_kernel_time = 3
config.model.output_mel_adapter_kernel_freq = 9
config.model.output_mel_adapter_zero_init = False
config.model.output_mel_adapter_final_init_std = 1e-4
config.model.output_mel_adapter_scale_ramp_steps = 5000
config.model.output_mel_adapter_use_v_base = True
config.model.output_mel_adapter_use_zt = True
config.model.output_mel_adapter_use_cond = True
config.model.output_mel_adapter_use_mask = True
config.model.output_mel_adapter_use_freq_pos = True
config.model.output_mel_adapter_use_cutoff_dist = True
config.model.output_mel_adapter_use_band_gates = True
config.model.output_mel_adapter_near_hz = 4000.0
config.model.output_mel_adapter_mid_hz = 10000.0
config.model.output_mel_adapter_band_gate_init = 1.0

config.train.train_from_scratch = True
config.train.resume_from = None
config.train.load_checkpoint = None
config.train.input_sample_rates = [8000, 12000, 16000, 24000]
config.train.sr_sampling_probs = [0.35, 0.25, 0.20, 0.20]
config.validation.fixed_validation_sample_rates = [8000, 12000, 16000, 24000]

config.train.use_hf_frequency_weight = True
config.train.hf_weight_min = 1.0
config.train.hf_weight_max = 1.5
config.train.hf_weight_mode = "mel_capped"
config.train.hf_weight_cap_hz_above_cutoff = 4000.0
config.train.loss_hf_l1_weight = 0.03
config.train.edge_continuity_loss_weight = 0.03
config.train.edge_continuity_bins = 4
config.train.melconv_bridge_l1_weight = 0.0
config.train.output_mel_adapter_l1_weight = 0.0

config.train.num_epochs = 40
config.train.batchsize = 8
config.train.grad_accum_every = 2
config.train.num_train_steps = None
config.train.lr = 1e-4
config.train.bridge_lr = 1e-4
config.train.lite_transformer_lr = 1e-4
config.train.output_adapter_lr = 1e-4
config.train.initial_lr = 1e-6
config.train.n_warmup_steps = 5000
config.train.scheduler_type = "cosine"
config.train.mixed_precision = "no"
config.train.max_grad_norm = 1.0

config.checkpoint.save_dir = "model/scratch_interleaved_melconv_headadapter_cbtbridge_official100"
config.logging.log_dir = "log/scratch_interleaved_melconv_headadapter_cbtbridge_official100"
config.logging.run_name = "ScratchInterleavedMelConv_CBTBridge_official100"
config.logging.tracker_project_name = "scratch_interleaved_melconv_cbtbridge_official100"
config.inference.model_path = "model/scratch_interleaved_melconv_headadapter_cbtbridge_official100/best.pt"
config.inference.output_path = "inference_output/scratch_interleaved_melconv_headadapter_cbtbridge_official100"

config.checkpoint.save_best_checkpoint = True
config.checkpoint.best_metric_name = "valid_avg/LSD_HF"
config.checkpoint.best_metric_mode = "min"
config.checkpoint.best_checkpoint_name = "best.pt"

cbt_train_keys = [
    "train/cbt_alpha",
    "train/cbt_event_gate_mean",
    "train/cbt_delta_l1",
    "train/cbt_injected_delta_l1",
    "train/cbt_transport_entropy",
    "train/cbt_active_high_4_6k",
    "train/cbt_active_high_6_8k",
    "train/cbt_active_high_8_12k",
    "train/cbt_active_high_12_16k",
    "train/cbt_active_high_16_24k",
]
cbt_valid_keys = [
    "valid_avg/cbt_alpha",
    "valid_avg/cbt_event_gate_mean",
    "valid_avg/cbt_delta_l1",
    "valid_avg/cbt_injected_delta_l1",
    "valid_avg/cbt_transport_entropy",
    "valid_avg/cbt_active_high_4_6k",
    "valid_avg/cbt_active_high_6_8k",
    "valid_avg/cbt_active_high_8_12k",
    "valid_avg/cbt_active_high_12_16k",
    "valid_avg/cbt_active_high_16_24k",
]

config.logging.tensorboard_train_keys = [
    "train/loss_total",
    "train/loss_flow",
    "train/loss_flow_masked",
    "train/loss_flow_hf_weighted",
    "train/loss_hf_l1",
    "train/loss_edge_continuity",
    "train/loss_output_mel_adapter_l1",
    "train/effective_batch_size",
    "train/segment_mask_valid_ratio",
    *cbt_train_keys,
    "train/output_adapter_delta_l1",
    "train/output_adapter_v_base_l1",
    "train/output_adapter_delta_base_ratio",
    "train/output_adapter_lowband_delta_l1",
    "train/output_adapter_near_delta_l1",
    "train/output_adapter_mid_delta_l1",
    "train/output_adapter_far_delta_l1",
    "train/output_adapter_band_gate_near",
    "train/output_adapter_band_gate_mid",
    "train/output_adapter_band_gate_far",
    "train/output_adapter_band_weight_mean",
    "train/output_adapter_band_weight_max",
    "train/output_adapter_ramp",
    "train/grad_norm_total",
    "train/grad_norm_base",
    "train/grad_norm_bridge",
    "train/grad_norm_lite_transformer",
    "train/grad_norm_output_adapter",
    "train/pred_target_l1_ratio",
    "train/highband_delta",
    "train/hard_lowband_delta",
    "training/cfm_loss",
    "training/grad_norm",
    "training/lr",
]

config.logging.tensorboard_valid_keys = [
    "valid_avg/LSD",
    "valid_avg/LSD_LF",
    "valid_avg/LSD_HF",
    "valid_8k/LSD_HF",
    "valid_12k/LSD_HF",
    "valid_16k/LSD_HF",
    "valid_24k/LSD_HF",
    "valid_avg/pred_target_l1_ratio",
    "valid_avg/highband_delta",
    "valid_avg/hard_lowband_delta",
    *cbt_valid_keys,
    "valid_avg/output_adapter_delta_base_ratio",
    "valid_avg/output_adapter_lowband_delta_l1",
    "valid_avg/output_adapter_near_delta_l1",
    "valid_avg/output_adapter_mid_delta_l1",
    "valid_avg/output_adapter_far_delta_l1",
    "valid_avg/output_adapter_ramp",
]

config.logging.txt_epoch_keys = [
    *config.logging.tensorboard_train_keys,
    *config.logging.tensorboard_valid_keys,
    "best_metric_name",
    "best_metric_value",
    "best_epoch",
    "best_step",
    "checkpoint_path",
]

config.training = config.train
