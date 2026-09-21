import os
print(os.getcwd())

import math
import logging
from random import random
from functools import partial
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
import torchode as to
from torchdiffeq import odeint
from beartype import beartype
from beartype.typing import Optional
from einops import rearrange, repeat, reduce, pack, unpack
import torchaudio.transforms as T
from torchaudio.functional import resample
from librosa.filters import mel as librosa_mel_fn
from .utils import sequence_mask
import numpy
import matplotlib.pyplot as plt
from .modules import LearnedSinusoidalPosEmb, ConvPositionEmbed, Transformer, ConvNeXtBlock, Attention, FeedForward
from .modules import RotaryEmbedding, RMSNorm, AdaptiveRMSNorm, GateLoop
from .postprocessing import PostProcessing
from .vocoder import init_bigvgan

LOGGER = logging.getLogger(__file__)
logging.basicConfig(filename='model_debug.log', level=logging.INFO)

DETERMINISTIC_CUTOFF_DEADPATH_PATCH = True
CUTOFF_BIN_POLICY = "compute_only_when_nonresidual_or_mel_pp_or_independent_cfm_mix"


def needs_cutoff_bins_for_sample(use_highband_residual_flow, mel_pp, cfm_method):
    return (
        (not bool(use_highband_residual_flow))
        or bool(mel_pp)
        or str(cfm_method) == "independent_cfm_mix"
    )


def randn_like_with_optional_generator(reference, generator=None):
    if generator is None:
        return torch.randn_like(reference)
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )

# helper functions
def exists(val):
    return val is not None

def identity(t):
    return t

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def coin_flip():
    return random() < 0.5

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# mel helpers
mel_basis = {}
hann_window = {}

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def build_highband_mel_mask(
    n_mels,
    sample_rate,
    f_min,
    f_max,
    cutoff_hz,
    softness_hz=500.0,
    device=None,
    dtype=None
):
    """Build a soft high-frequency mel mask from mel bin center frequencies."""
    device = device if exists(device) else torch.device('cpu')
    dtype = dtype if exists(dtype) else torch.float32

    f_max = min(float(f_max), float(sample_rate) / 2)
    mel_min = 2595.0 * numpy.log10(1.0 + float(f_min) / 700.0)
    mel_max = 2595.0 * numpy.log10(1.0 + f_max / 700.0)
    centers = 700.0 * (10.0 ** (numpy.linspace(mel_min, mel_max, int(n_mels)) / 2595.0) - 1.0)
    centers = torch.as_tensor(centers, device=device, dtype=dtype)
    cutoff_hz = torch.as_tensor(cutoff_hz, device=device, dtype=dtype)
    softness_hz = max(float(softness_hz), 1e-6)

    if cutoff_hz.ndim == 0:
        mask = torch.sigmoid((centers - cutoff_hz) / softness_hz)
        return rearrange(mask, 'd -> 1 1 d')

    mask = torch.sigmoid((rearrange(centers, 'd -> 1 d') - rearrange(cutoff_hz, 'b -> b 1')) / softness_hz)
    return rearrange(mask, 'b d -> b 1 d')


def mel_center_frequencies(n_mels, sample_rate, f_min, f_max, device=None, dtype=None):
    device = device if exists(device) else torch.device('cpu')
    dtype = dtype if exists(dtype) else torch.float32
    f_max = min(float(f_max), float(sample_rate) / 2)
    mel_min = 2595.0 * numpy.log10(1.0 + float(f_min) / 700.0)
    mel_max = 2595.0 * numpy.log10(1.0 + f_max / 700.0)
    centers = 700.0 * (10.0 ** (numpy.linspace(mel_min, mel_max, int(n_mels)) / 2595.0) - 1.0)
    return torch.as_tensor(centers, device=device, dtype=dtype)

def orient_highband_mask(mask_hf, mel_tensor):
    """Orient a [B|1,1,n_mels] mask for [B,T,n_mels] or [B,n_mels,T] mel tensors."""
    if mask_hf.shape == mel_tensor.shape:
        return mask_hf

    n_mels = mask_hf.shape[-1]
    if mel_tensor.ndim != 3:
        raise ValueError(f"expected 3D mel tensor, got shape {tuple(mel_tensor.shape)}")

    if mel_tensor.shape[-1] == n_mels:
        return mask_hf

    if mel_tensor.shape[1] == n_mels:
        return rearrange(mask_hf, 'b 1 d -> b d 1')

    raise ValueError(
        f"could not orient high-band mask with n_mels={n_mels} for mel shape {tuple(mel_tensor.shape)}"
    )


def build_hf_frequency_weight(
    hf_mask,
    hf_weight_min=1.0,
    hf_weight_max=1.5,
    *,
    mode="linear",
    cutoff_hz=None,
    mel_freqs=None,
    cap_hz_above_cutoff=4000.0,
    **legacy_kwargs,
):
    """Build a high-band frequency ramp for [B, T, n_mels] tensors.

    mode="linear" preserves the old bin-index ramp. mode="mel_capped" ramps
    from the cutoff to cutoff + cap_hz_above_cutoff and then stays capped.
    """
    if hf_mask.ndim != 3:
        raise ValueError(f"build_hf_frequency_weight expects [B, T, n_mels], got {tuple(hf_mask.shape)}")

    batch, _, n_mels = hf_mask.shape
    device, dtype = hf_mask.device, hf_mask.dtype
    min_weight = float(legacy_kwargs.get("min_weight", hf_weight_min))
    max_weight = float(legacy_kwargs.get("max_weight", hf_weight_max))
    positions = torch.linspace(0, 1, n_mels, device=device, dtype=dtype).view(1, 1, n_mels)
    mask_freq = hf_mask.detach().amax(dim=1)
    active = mask_freq > 0.01
    indices = torch.arange(n_mels, device=device).view(1, n_mels).expand(batch, n_mels)
    first_active = torch.where(active, indices, torch.full_like(indices, n_mels - 1)).amin(dim=1)
    cutoff_pos = (first_active.to(dtype) / max(n_mels - 1, 1)).view(batch, 1, 1)
    if mode != "mel_capped":
        ramp = ((positions - cutoff_pos) / (1 - cutoff_pos).clamp_min(1e-6)).clamp(0, 1)
        return min_weight + (max_weight - min_weight) * ramp

    if exists(mel_freqs) and exists(cutoff_hz):
        mel_freqs = torch.as_tensor(mel_freqs, device=device, dtype=dtype).view(1, 1, n_mels)
        cutoff_hz = torch.as_tensor(cutoff_hz, device=device, dtype=dtype)
        if cutoff_hz.ndim == 0:
            cutoff_hz = repeat(cutoff_hz, '-> b', b=batch)
        cutoff_hz = cutoff_hz.view(batch, 1, 1)
        cap = max(float(cap_hz_above_cutoff), 1e-6)
        ramp = ((mel_freqs - cutoff_hz) / cap).clamp(0, 1)
    else:
        cap_bins = max(1.0, float(n_mels) * float(cap_hz_above_cutoff) / 24000.0)
        bin_pos = torch.arange(n_mels, device=device, dtype=dtype).view(1, 1, n_mels)
        cutoff_bin = first_active.to(dtype).view(batch, 1, 1)
        ramp = ((bin_pos - cutoff_bin) / cap_bins).clamp(0, 1)
    weight = min_weight + (max_weight - min_weight) * ramp
    return torch.where(hf_mask.detach().amax(dim=1, keepdim=True) > 0.01, weight, torch.ones_like(weight))


def compute_edge_continuity_loss(pred_residual, target_residual, hf_mask, edge_bins=4, loss_mask=None):
    if edge_bins <= 0:
        return pred_residual.new_tensor(0.0)
    hf_mask = orient_highband_mask(hf_mask, pred_residual).expand_as(pred_residual)
    batch, _, n_mels = pred_residual.shape
    active = hf_mask.detach().amax(dim=1) >= 0.5
    indices = torch.arange(n_mels, device=pred_residual.device).view(1, n_mels).expand(batch, n_mels)
    first_active = torch.where(active, indices, torch.full_like(indices, n_mels)).amin(dim=1)
    edge_mask = torch.zeros((batch, n_mels), device=pred_residual.device, dtype=pred_residual.dtype)
    for i, cutoff_bin in enumerate(first_active.tolist()):
        if cutoff_bin >= n_mels:
            continue
        lo = max(int(cutoff_bin) - int(edge_bins), 0)
        hi = min(int(cutoff_bin) + int(edge_bins), n_mels)
        edge_mask[i, lo:hi] = 1.0
    edge_mask = edge_mask.view(batch, 1, n_mels).expand_as(pred_residual)
    if exists(loss_mask):
        edge_mask = edge_mask * rearrange(loss_mask, 'b n -> b n 1').to(edge_mask.dtype)
    den = edge_mask.sum().clamp_min(1.0)
    return (torch.abs(pred_residual - target_residual) * edge_mask).sum() / den


def smooth_mel_seam(M_final_raw, hf_mask, kernel_size=3, seam_bins=4):
    if kernel_size <= 1 or seam_bins <= 0:
        return M_final_raw
    if kernel_size % 2 == 0:
        raise ValueError("seam_smoothing_kernel_size must be odd")
    hf_mask = orient_highband_mask(hf_mask, M_final_raw).expand_as(M_final_raw)
    batch, time, n_mels = M_final_raw.shape
    active = hf_mask.detach().amax(dim=1) >= 0.5
    indices = torch.arange(n_mels, device=M_final_raw.device).view(1, n_mels).expand(batch, n_mels)
    first_active = torch.where(active, indices, torch.full_like(indices, n_mels)).amin(dim=1)
    seam_mask = torch.zeros((batch, n_mels), device=M_final_raw.device, dtype=M_final_raw.dtype)
    for i, cutoff_bin in enumerate(first_active.tolist()):
        if cutoff_bin >= n_mels:
            continue
        lo = max(int(cutoff_bin) - int(seam_bins), 0)
        hi = min(int(cutoff_bin) + int(seam_bins), n_mels)
        seam_mask[i, lo:hi] = 1.0
    seam_mask = seam_mask.view(batch, 1, n_mels).expand_as(M_final_raw)
    x = rearrange(M_final_raw, 'b t f -> (b t) 1 f')
    smooth = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    smooth = rearrange(smooth, '(b t) 1 f -> b t f', b=batch, t=time)
    return M_final_raw * (1.0 - seam_mask) + smooth * seam_mask


class AdaLNZeroTransformerBlock(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.,
        ff_dropout=0.,
        attn_flash=False,
        attn_qk_norm=False,
        cond_dim=None,
        zero_init=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash, qk_norm=attn_qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(default(cond_dim, dim), dim * 6)
        )
        if zero_init:
            nn.init.zeros_(self.adaln_mlp[-1].weight)
            nn.init.zeros_(self.adaln_mlp[-1].bias)
        self.last_gate_msa = None
        self.last_gate_ffn = None

    def forward(self, x, *, mask=None, rotary_emb=None, cond=None):
        if not exists(cond):
            raise ValueError("AdaLN-Zero transformer requires adaptive_rmsnorm_cond")
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = self.adaln_mlp(cond).chunk(6, dim=-1)
        self.last_gate_msa = gate_msa.detach()
        self.last_gate_ffn = gate_ffn.detach()

        h = self.norm1(x)
        h = h * (1.0 + rearrange(scale_msa, 'b d -> b 1 d')) + rearrange(shift_msa, 'b d -> b 1 d')
        x = x + rearrange(gate_msa, 'b d -> b 1 d') * self.attn(h, mask=mask, rotary_emb=rotary_emb)

        h = self.norm2(x)
        h = h * (1.0 + rearrange(scale_ffn, 'b d -> b 1 d')) + rearrange(shift_ffn, 'b d -> b 1 d')
        x = x + rearrange(gate_ffn, 'b d -> b 1 d') * self.ff(h)
        return x


class AdaLNZeroTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.,
        ff_dropout=0.,
        attn_flash=False,
        attn_qk_norm=False,
        adaln_zero_cond_dim=None,
        adaln_zero_init=True,
        **unused_kwargs,
    ):
        super().__init__()
        self.rotary_emb = None
        self.layers = nn.ModuleList([
            AdaLNZeroTransformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                attn_flash=attn_flash,
                attn_qk_norm=attn_qk_norm,
                cond_dim=adaln_zero_cond_dim,
                zero_init=adaln_zero_init,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, adaptive_rmsnorm_cond=None):
        for block in self.layers:
            x = block(x, mask=mask, rotary_emb=None, cond=adaptive_rmsnorm_cond)
        return self.final_norm(x)

    def get_adaln_gate_stats(self):
        gate_msa = [layer.last_gate_msa for layer in self.layers if exists(layer.last_gate_msa)]
        gate_ffn = [layer.last_gate_ffn for layer in self.layers if exists(layer.last_gate_ffn)]
        if not gate_msa or not gate_ffn:
            return {
                "adaln_gate_msa_mean": 0.0,
                "adaln_gate_msa_abs_mean": 0.0,
                "adaln_gate_ffn_mean": 0.0,
                "adaln_gate_ffn_abs_mean": 0.0,
            }
        gate_msa = torch.stack([g.float().mean() for g in gate_msa])
        gate_msa_abs = torch.stack([g.float().abs().mean() for g in gate_msa])
        gate_ffn = torch.stack([g.float().mean() for g in gate_ffn])
        gate_ffn_abs = torch.stack([g.float().abs().mean() for g in gate_ffn])
        return {
            "adaln_gate_msa_mean": float(gate_msa.mean().detach().cpu()),
            "adaln_gate_msa_abs_mean": float(gate_msa_abs.mean().detach().cpu()),
            "adaln_gate_ffn_mean": float(gate_ffn.mean().detach().cpu()),
            "adaln_gate_ffn_abs_mean": float(gate_ffn_abs.mean().detach().cpu()),
        }


class DepthFlexibleTransformer(Module):
    """Same local Transformer path without the repository's even-depth assertion."""

    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.,
        ff_dropout=0.,
        num_register_tokens=0.,
        attn_flash=False,
        adaptive_rmsnorm=False,
        adaptive_rmsnorm_cond_dim_in=None,
        use_unet_skip_connection=False,
        skip_connect_scale=None,
        attn_qk_norm=False,
        use_gateloop_layers=False,
        gateloop_use_jax=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.rotary_emb = RotaryEmbedding(dim=dim_head)
        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0
        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        if adaptive_rmsnorm:
            rmsnorm_klass = partial(AdaptiveRMSNorm, cond_dim=adaptive_rmsnorm_cond_dim_in)
        else:
            rmsnorm_klass = RMSNorm

        self.skip_connect_scale = default(skip_connect_scale, 2 ** -0.5)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)
            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if has_skip else None,
                GateLoop(dim=dim, use_jax_associative_scan=gateloop_use_jax, post_ln=True) if use_gateloop_layers else None,
                rmsnorm_klass(dim=dim),
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash, qk_norm=attn_qk_norm),
                rmsnorm_klass(dim=dim),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.final_norm = RMSNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, mask=None, adaptive_rmsnorm_cond=None):
        batch, seq_len, *_ = x.shape

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=batch)
            x, ps = pack([register_tokens, x], 'b * d')
            if exists(mask):
                mask = F.pad(mask, (self.num_register_tokens, 0), value=True)

        skip_connects = []
        positions = seq_len
        if self.has_register_tokens:
            main_positions = torch.arange(seq_len, device=self.device, dtype=torch.long)
            register_positions = torch.full((self.num_register_tokens,), -10000, device=self.device, dtype=torch.long)
            positions = torch.cat((register_positions, main_positions))

        rotary_emb = self.rotary_emb(positions)
        rmsnorm_kwargs = dict()
        if exists(adaptive_rmsnorm_cond):
            rmsnorm_kwargs = dict(cond=adaptive_rmsnorm_cond)

        for skip_combiner, maybe_gateloop, attn_prenorm, attn, ff_prenorm, ff in self.layers:
            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim=-1)
                x = skip_combiner(x)

            if exists(maybe_gateloop):
                x = maybe_gateloop(x) + x

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = attn(attn_input, mask=mask, rotary_emb=rotary_emb) + x
            ff_input = ff_prenorm(x, **rmsnorm_kwargs)
            x = ff(ff_input) + x

        if self.has_register_tokens:
            _, x = unpack(x, ps, 'b * d')

        return self.final_norm(x)

# tensor helpers
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def reduce_masks_with_and(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

def interpolate_1d(t, length, mode = 'bilinear'):
    " pytorch does not offer interpolation 1d, so hack by converting to 2d "

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = rearrange(t, 'b n -> b 1 n')

    t = rearrange(t, 'b d n -> b d n 1')
    t = F.interpolate(t, (length, 1), mode = mode)
    t = rearrange(t, 'b d n 1 -> b d n')

    if implicit_one_channel:
        t = rearrange(t, 'b 1 n -> b n')

    t = t.to(dtype)
    
    return t

def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value = 0.)

    return t

# mask construction helpers
def mask_from_start_end_indices(seq_len: int, start: Tensor, end: Tensor):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long() # start 
    mask &= seq < end[..., None].long()
    
    return mask

def mask_from_frac_lengths(seq_len: int, frac_lengths: Tensor):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long() 
    max_start = seq_len - lengths 

    rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1) 
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths 
    
    return mask_from_start_end_indices(seq_len, start, end)

def mask_for_freqency(cond, batch: int, seq_len: int, mel_dim: int, device):

    for i in range(batch):
        
        import random
        mask_height = random.randint(10,20)
        rand_start = random.randint(20, mel_dim - mask_height) 
        minimum = torch.min(cond)
        cond[i, :, rand_start: rand_start + mask_height] = minimum + 1e-3

    return cond
    
    
# encoder decoders

class AudioEncoderDecoder(nn.Module):
    pass

class MelVoco(AudioEncoderDecoder):
    def __init__(
        self,
        *,
        log = True,
        n_mels = 256,
        sampling_rate = 48000,
        f_max = 24000,
        f_min = 20,
        n_fft = 2048,
        win_length = 2048,
        hop_length = 480,
        vocoder = str,
        vocoder_config = './vocoder_config.json',
        vocoder_path = None
    ):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.f_min = f_min
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        
        if vocoder == 'bigvgan':
            self.vocoder_name = vocoder
            self.vocoder = init_bigvgan(vocoder_config, vocoder_path, vocoder_freeze=True)
        else:
            raise ValueError("unsuitable vocoder name")

    @property
    def downsample_factor(self):
        raise NotImplementedError

    @property
    def latent_dim(self):
        return self.n_mels
    
    def encode(self, audio):
        if torch.min(audio) < -1.:
            print('min value is ', torch.min(audio))
        if torch.max(audio) > 1.:
            print('max value is ', torch.max(audio))

        global mel_basis, hann_window
        mel_key = f"{self.f_max}_{audio.device}"
        win_key = str(audio.device)

        if mel_key not in mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max
            )
            mel_basis[mel_key] = torch.from_numpy(mel).float().to(audio.device)

        if win_key not in hann_window:
            hann_window[win_key] = torch.hann_window(self.win_length).to(audio.device)

        audio = torch.nn.functional.pad(audio.unsqueeze(1), (int((self.n_fft-self.hop_length)/2), int((self.n_fft-self.hop_length)/2)), mode='reflect')
        audio = audio.squeeze(1)

        # complex tensor as default, then use view_as_real for future pytorch compatibility
        spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=hann_window[win_key],
                        center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(mel_basis[mel_key], spec)
        spec = spectral_normalize_torch(spec)
        spec = rearrange(spec, 'b d n -> b n d')
        return spec    

    def encode_torchaudio(self, audio):

        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        ).cuda()

        audio = audio.cuda()
        spectrogram = stft_transform(audio)

        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        ).cuda()

        spec = mel_transform(spectrogram)
        
        if self.log:
            spec = T.AmplitudeToDB()(spec)
        spec = rearrange(spec, 'b d n -> b n d')
        return spec

    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')

        # if self.log:
        #     mel = DB_to_amplitude(mel, ref = 1., power = 0.5)

        if self.vocoder_name == 'bigvgan':  
            return self.vocoder.forward(mel)


def _as_batch_vector(value, batch, device, dtype):
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = repeat(value, '-> b', b=batch)
    if value.ndim > 1:
        value = value.reshape(batch)
    return value


def hz_to_mel(hz):
    hz = torch.as_tensor(hz)
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    mel = torch.as_tensor(mel)
    return 700.0 * (torch.exp(mel * (math.log(10.0) / 2595.0)) - 1.0)


def get_mel_bin_centers_hz(n_mels, target_sr=48000, device=None, dtype=None):
    dtype = default(dtype, torch.float32)
    mel_min = hz_to_mel(torch.tensor(0.0, device=device, dtype=dtype))
    mel_max = hz_to_mel(torch.tensor(float(target_sr) / 2.0, device=device, dtype=dtype))
    edges = torch.linspace(mel_min, mel_max, int(n_mels) + 1, device=device, dtype=dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return mel_to_hz(centers)


def build_band_masks(mel_centers_hz, groups_hz):
    masks = []
    for start_hz, end_hz in groups_hz:
        masks.append(((mel_centers_hz >= float(start_hz)) & (mel_centers_hz < float(end_hz))).to(mel_centers_hz.dtype))
    return torch.stack(masks, dim=0)


def _groups_to_tensor(groups_hz, device, dtype):
    return torch.tensor(groups_hz, device=device, dtype=dtype)


def _low_group_reliability(groups_hz, cutoff_hz):
    batch = cutoff_hz.shape[0]
    starts = groups_hz[:, 0].view(1, -1)
    ends = groups_hz[:, 1].view(1, -1)
    cutoff = cutoff_hz.view(batch, 1)
    ratio = (cutoff - starts) / (ends - starts).clamp_min(1e-6)
    return ratio.clamp(0.0, 1.0)


def _high_group_active(groups_hz, cutoff_hz):
    batch = cutoff_hz.shape[0]
    starts = groups_hz[:, 0].view(1, -1)
    ends = groups_hz[:, 1].view(1, -1)
    cutoff = cutoff_hz.view(batch, 1)
    ratio = (ends - cutoff) / (ends - starts).clamp_min(1e-6)
    return ratio.clamp(0.0, 1.0)


class MelConvResidualBlock(nn.Module):
    def __init__(self, channels, kernel_time=3, kernel_freq=9):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_time, kernel_freq),
                padding=(kernel_time // 2, kernel_freq // 2),
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class InterleavedMelConvBridge(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_in,
        channels=64,
        blocks=3,
        kernel_time=3,
        kernel_freq=9,
        zero_init=False,
        final_init_std=1e-4,
        scale_init=1.0,
        scale_ramp_steps=5000,
        use_hidden_mel=True,
        use_zt=True,
        use_cond=True,
        use_mask=True,
        use_freq_pos=True,
        use_cutoff_dist=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.scale_ramp_steps = int(scale_ramp_steps)
        self.use_hidden_mel = bool(use_hidden_mel)
        self.use_zt = bool(use_zt)
        self.use_cond = bool(use_cond)
        self.use_mask = bool(use_mask)
        self.use_freq_pos = bool(use_freq_pos)
        self.use_cutoff_dist = bool(use_cutoff_dist)
        in_channels = sum([
            self.use_hidden_mel,
            self.use_zt,
            self.use_cond,
            self.use_mask,
            self.use_freq_pos,
            self.use_cutoff_dist,
        ])
        self.hidden_to_mel = nn.Linear(dim, dim_in)
        self.mel_to_hidden = nn.Linear(dim_in, dim)
        self.bridge_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.in_proj = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1), nn.GELU())
        self.blocks = nn.Sequential(*[
            MelConvResidualBlock(channels, kernel_time=kernel_time, kernel_freq=kernel_freq)
            for _ in range(int(blocks))
        ])
        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)
        else:
            nn.init.normal_(self.final_conv.weight, mean=0.0, std=float(final_init_std))
            nn.init.zeros_(self.final_conv.bias)

    def _inputs(self, hidden, z_t, cond_raw, hf_mask, cutoff_ratio):
        batch, time, n_freq = z_t.shape
        device, dtype = z_t.device, z_t.dtype
        inputs = []
        hidden_mel = self.hidden_to_mel(hidden)
        if self.use_hidden_mel:
            inputs.append(hidden_mel)
        if self.use_zt:
            inputs.append(z_t)
        if self.use_cond:
            inputs.append(cond_raw)
        if self.use_mask:
            inputs.append(hf_mask)
        freq_pos = torch.linspace(0, 1, n_freq, device=device, dtype=dtype).view(1, 1, n_freq).expand(batch, time, n_freq)
        if self.use_freq_pos:
            inputs.append(freq_pos)
        if self.use_cutoff_dist:
            cutoff_ratio = _as_batch_vector(cutoff_ratio, batch, device, dtype)
            cutoff_map = cutoff_ratio.view(batch, 1, 1).expand(batch, time, n_freq)
            inputs.append((freq_pos - cutoff_map).clamp_min(0.0))
        return torch.stack(inputs, dim=1), hidden_mel

    def forward(self, *, hidden, z_t, cond_raw, hf_mask, cutoff_ratio, global_step=None):
        hf_mask = orient_highband_mask(hf_mask, z_t).expand_as(z_t).to(z_t.dtype)
        conv_in, _ = self._inputs(hidden, z_t, cond_raw, hf_mask, cutoff_ratio)
        x = self.blocks(self.in_proj(conv_in))
        mel_delta = self.final_conv(x).squeeze(1) * hf_mask
        hidden_delta = self.mel_to_hidden(mel_delta) * self.bridge_scale
        if global_step is not None and self.scale_ramp_steps > 0:
            ramp = min(1.0, float(global_step) / float(self.scale_ramp_steps))
        else:
            ramp = 1.0
        hidden_delta = hidden_delta * ramp
        hf_den = hf_mask.sum().clamp_min(1e-6)
        mel_delta_l1 = (mel_delta.abs() * hf_mask).sum() / hf_den
        hidden_base_l1 = hidden.abs().mean().clamp_min(1e-8)
        debug = {
            "melconv_bridge_mel_delta_l1": float(mel_delta_l1.detach().cpu()),
            "melconv_bridge_hidden_delta_l1": float(hidden_delta.abs().mean().detach().cpu()),
            "melconv_bridge_hidden_base_l1": float(hidden_base_l1.detach().cpu()),
            "melconv_bridge_delta_base_ratio": float((hidden_delta.abs().mean() / hidden_base_l1).detach().cpu()),
            "melconv_bridge_lowband_delta_l1": float(((1.0 - hf_mask) * mel_delta.abs()).mean().detach().cpu()),
            "melconv_bridge_scale": float(self.bridge_scale.detach().cpu()),
            "melconv_bridge_ramp": float(ramp),
        }
        return hidden_delta, debug, mel_delta_l1


class LightweightCBTBridge(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_in,
        target_sr=48000,
        hidden_dim=32,
        low_groups_hz=((0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000)),
        high_groups_hz=((4000, 6000), (6000, 8000), (8000, 12000), (12000, 16000), (16000, 24000)),
        use_event_gate=True,
        use_temporal_derivative=True,
        use_depthwise_temporal_conv=True,
        temporal_kernel=5,
        zero_init=True,
        init_scale=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.dim_in = int(dim_in)
        self.target_sr = int(target_sr)
        self.hidden_dim = min(int(hidden_dim), 64)
        self.low_groups_hz = [tuple(map(float, group)) for group in low_groups_hz]
        self.high_groups_hz = [tuple(map(float, group)) for group in high_groups_hz]
        self.use_event_gate = bool(use_event_gate)
        self.use_temporal_derivative = bool(use_temporal_derivative)
        self.use_depthwise_temporal_conv = bool(use_depthwise_temporal_conv)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.g_low = len(self.low_groups_hz)
        self.g_high = len(self.high_groups_hz)
        evidence_dim = self.g_low * (2 if self.use_temporal_derivative else 1)
        self.evidence_dim = evidence_dim

        self.transport_logits = nn.Parameter(torch.zeros(self.g_high, self.g_low))
        self.low_proj = nn.Linear(evidence_dim, self.g_low * self.hidden_dim)
        if self.use_event_gate:
            self.event_mlp = nn.Sequential(
                nn.Linear(evidence_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.event_mlp = None

        if self.use_depthwise_temporal_conv:
            kernel = int(temporal_kernel)
            if kernel % 2 == 0:
                kernel += 1
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel, padding=kernel // 2, groups=self.hidden_dim),
                nn.SiLU(),
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            )
        else:
            self.temporal_conv = None

        self.output_proj = nn.Linear(self.hidden_dim, self.dim)
        if zero_init:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.output_proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

    @staticmethod
    def _mask_to_btc(mask, reference):
        if mask is None:
            return None
        if mask.ndim == 3 and mask.shape[1] == 1:
            mask = rearrange(mask, "b 1 t -> b t 1")
        elif mask.ndim == 2:
            mask = rearrange(mask, "b t -> b t 1")
        elif mask.ndim == 3 and mask.shape[-1] != 1:
            mask = mask[..., :1]
        return mask.to(device=reference.device, dtype=reference.dtype)

    def active_high_gates_for_cutoff(self, cutoff_hz, device=None, dtype=None):
        dtype = default(dtype, torch.float32)
        cutoff = torch.as_tensor(cutoff_hz, device=device, dtype=dtype)
        if cutoff.ndim == 0:
            cutoff = cutoff.view(1)
        groups = _groups_to_tensor(self.high_groups_hz, cutoff.device, cutoff.dtype)
        return _high_group_active(groups, cutoff)

    def forward(self, *, h, m_up, cutoff_hz, segment_mask=None, cutoff_emb=None):
        batch, time, _ = h.shape
        device, dtype = h.device, h.dtype
        m_up = m_up.to(device=device, dtype=dtype)
        cutoff_hz = _as_batch_vector(cutoff_hz, batch, device, dtype)

        mel_centers = get_mel_bin_centers_hz(m_up.shape[-1], self.target_sr, device=device, dtype=dtype)
        low_masks = build_band_masks(mel_centers, self.low_groups_hz).to(dtype)
        low_den = low_masks.sum(dim=-1).clamp_min(1.0)
        low_mean = torch.einsum("btf,gf->btg", m_up, low_masks) / low_den.view(1, 1, self.g_low)

        low_group_tensor = _groups_to_tensor(self.low_groups_hz, device, dtype)
        reliable_low = _low_group_reliability(low_group_tensor, cutoff_hz)
        low_mean = low_mean * reliable_low.view(batch, 1, self.g_low)

        if self.use_temporal_derivative:
            low_diff = torch.zeros_like(low_mean)
            low_diff[:, 1:] = (low_mean[:, 1:] - low_mean[:, :-1]).abs()
            low_diff = low_diff * reliable_low.view(batch, 1, self.g_low)
            evidence = torch.cat([low_mean, low_diff], dim=-1)
        else:
            evidence = low_mean

        transport = torch.softmax(self.transport_logits, dim=-1).view(1, self.g_high, self.g_low)
        transport = transport * reliable_low.view(batch, 1, self.g_low)
        transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        low_proj = self.low_proj(evidence).view(batch, time, self.g_low, self.hidden_dim)
        transported = torch.einsum("bhl,btld->bthd", transport, low_proj)

        high_group_tensor = _groups_to_tensor(self.high_groups_hz, device, dtype)
        active_high = _high_group_active(high_group_tensor, cutoff_hz)
        transported = transported * active_high.view(batch, 1, self.g_high, 1)

        if self.use_event_gate:
            event_gate = torch.sigmoid(self.event_mlp(evidence))
        else:
            event_gate = torch.ones(batch, time, 1, device=device, dtype=dtype)
        transported = transported * event_gate.view(batch, time, 1, 1)

        transported_sum = transported.sum(dim=2)
        if exists(self.temporal_conv):
            transported_sum = self.temporal_conv(rearrange(transported_sum, "b t d -> b d t"))
            transported_sum = rearrange(transported_sum, "b d t -> b t d")
        transported_sum = self.dropout(transported_sum)

        delta_h = self.output_proj(transported_sum)
        valid_mask = self._mask_to_btc(segment_mask, delta_h)
        if exists(valid_mask):
            delta_h = delta_h * valid_mask

        injected_delta = self.alpha * delta_h
        h_out = h + injected_delta
        entropy = -(transport * torch.log(transport + 1e-8)).sum(dim=-1).mean()
        debug = {
            "cbt_alpha": float(self.alpha.detach().cpu()),
            "cbt_event_gate_mean": float(event_gate.detach().mean().cpu()),
            "cbt_delta_l1": float(delta_h.detach().abs().mean().cpu()),
            "cbt_injected_delta_l1": float(injected_delta.detach().abs().mean().cpu()),
            "cbt_transport_entropy": float(entropy.detach().cpu()),
        }
        labels = ("4_6k", "6_8k", "8_12k", "12_16k", "16_24k")
        active_mean = active_high.detach().mean(dim=0).cpu()
        for idx, label in enumerate(labels[:self.g_high]):
            debug[f"cbt_active_high_{label}"] = float(active_mean[idx])
        return h_out, debug


class OutputMelAdapter(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        channels=64,
        blocks=2,
        kernel_time=3,
        kernel_freq=9,
        zero_init=False,
        final_init_std=1e-4,
        scale_ramp_steps=5000,
        use_v_base=True,
        use_zt=True,
        use_cond=True,
        use_mask=True,
        use_freq_pos=True,
        use_cutoff_dist=True,
        use_band_gates=True,
        near_hz=4000.0,
        mid_hz=10000.0,
        band_gate_init=1.0,
    ):
        super().__init__()
        self.scale_ramp_steps = int(scale_ramp_steps)
        self.use_v_base = bool(use_v_base)
        self.use_zt = bool(use_zt)
        self.use_cond = bool(use_cond)
        self.use_mask = bool(use_mask)
        self.use_freq_pos = bool(use_freq_pos)
        self.use_cutoff_dist = bool(use_cutoff_dist)
        self.use_band_gates = bool(use_band_gates)
        self.near_width = float(near_hz) / 24000.0
        self.mid_width = float(mid_hz) / 24000.0
        in_channels = sum([
            self.use_v_base,
            self.use_zt,
            self.use_cond,
            self.use_mask,
            self.use_freq_pos,
            self.use_cutoff_dist,
        ])
        self.in_proj = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=1), nn.GELU())
        self.blocks = nn.Sequential(*[
            MelConvResidualBlock(channels, kernel_time=kernel_time, kernel_freq=kernel_freq)
            for _ in range(int(blocks))
        ])
        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.output_band_gates = nn.Parameter(torch.ones(3) * float(band_gate_init))
        if zero_init:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)
        else:
            nn.init.normal_(self.final_conv.weight, mean=0.0, std=float(final_init_std))
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, *, v_base, z_t, cond_raw, hf_mask, cutoff_ratio, global_step=None):
        batch, time, n_freq = v_base.shape
        device, dtype = v_base.device, v_base.dtype
        hf_mask = orient_highband_mask(hf_mask, v_base).expand_as(v_base).to(dtype)
        freq_pos = torch.linspace(0, 1, n_freq, device=device, dtype=dtype).view(1, 1, n_freq).expand(batch, time, n_freq)
        inputs = []
        if self.use_v_base:
            inputs.append(v_base)
        if self.use_zt:
            inputs.append(z_t)
        if self.use_cond:
            inputs.append(cond_raw)
        if self.use_mask:
            inputs.append(hf_mask)
        if self.use_freq_pos:
            inputs.append(freq_pos)
        cutoff_ratio = _as_batch_vector(cutoff_ratio, batch, device, dtype)
        cutoff_map = cutoff_ratio.view(batch, 1, 1).expand(batch, time, n_freq)
        if self.use_cutoff_dist:
            inputs.append((freq_pos - cutoff_map).clamp_min(0.0))
        x = torch.stack(inputs, dim=1)
        x = self.blocks(self.in_proj(x))
        delta_raw = self.final_conv(x).squeeze(1)

        band_weight = torch.ones_like(delta_raw)
        gates = F.softplus(self.output_band_gates)
        if self.use_band_gates:
            near_mask = ((freq_pos >= cutoff_map) & (freq_pos < cutoff_map + self.near_width)).to(dtype)
            mid_mask = ((freq_pos >= cutoff_map + self.near_width) & (freq_pos < cutoff_map + self.mid_width)).to(dtype)
            far_mask = (freq_pos >= cutoff_map + self.mid_width).to(dtype)
            band_weight = gates[0] * near_mask + gates[1] * mid_mask + gates[2] * far_mask
            hf_den = hf_mask.sum().clamp_min(1e-6)
            band_mean_hf = (band_weight * hf_mask).sum() / hf_den
            band_weight = band_weight / band_mean_hf.detach().clamp_min(1e-6)
            delta_raw = delta_raw * band_weight
        if global_step is not None and self.scale_ramp_steps > 0:
            ramp = min(1.0, float(global_step) / float(self.scale_ramp_steps))
        else:
            ramp = 1.0
        delta_v = delta_raw * ramp * hf_mask

        hf_den = hf_mask.sum().clamp_min(1e-6)
        delta_v_l1 = (delta_v.abs() * hf_mask).sum() / hf_den
        v_base_l1 = (v_base.abs() * hf_mask).sum().clamp_min(1e-8) / hf_den
        def band_l1(mask):
            den = (mask * hf_mask).sum().clamp_min(1e-6)
            return ((delta_v.abs() * mask * hf_mask).sum() / den).detach()
        near_mask = ((freq_pos >= cutoff_map) & (freq_pos < cutoff_map + self.near_width)).to(dtype)
        mid_mask = ((freq_pos >= cutoff_map + self.near_width) & (freq_pos < cutoff_map + self.mid_width)).to(dtype)
        far_mask = (freq_pos >= cutoff_map + self.mid_width).to(dtype)
        debug = {
            "output_adapter_delta_l1": float(delta_v_l1.detach().cpu()),
            "output_adapter_v_base_l1": float(v_base_l1.detach().cpu()),
            "output_adapter_delta_base_ratio": float((((delta_v.abs() * hf_mask).sum() / hf_den) / v_base_l1.clamp_min(1e-8)).detach().cpu()),
            "output_adapter_lowband_delta_l1": float(((1.0 - hf_mask) * delta_v.abs()).mean().detach().cpu()),
            "output_adapter_near_delta_l1": float(band_l1(near_mask).cpu()),
            "output_adapter_mid_delta_l1": float(band_l1(mid_mask).cpu()),
            "output_adapter_far_delta_l1": float(band_l1(far_mask).cpu()),
            "output_adapter_band_gate_near": float(gates[0].detach().cpu()),
            "output_adapter_band_gate_mid": float(gates[1].detach().cpu()),
            "output_adapter_band_gate_far": float(gates[2].detach().cpu()),
            "output_adapter_band_weight_mean": float(band_weight.detach().mean().cpu()),
            "output_adapter_band_weight_max": float(band_weight.detach().max().cpu()),
            "output_adapter_ramp": float(ramp),
        }
        return delta_v, debug, delta_v_l1


class FLowHigh(Module):
    def __init__(
        self,
        *,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        dim_in = None, # 256
        dim_cond_emb = 0,
        dim = 1024,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        ff_dropout = 0.,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0.,
        attn_flash = False,
        attn_qk_norm = True,
        use_gateloop_layers = False,
        architecture = None,
        input_channels = 2,
        condition_with_hf_mask = False,
        condition_with_cutoff_embedding = False,
        use_adaln_zero = False,
        adaln_zero_init = True,
        adaln_zero_gate_log = True,
        use_interleaved_melconv = False,
        use_melconv_bridge = False,
        cbt_bridge_enabled = False,
        cbt_hidden_dim = 32,
        cbt_low_groups_hz = None,
        cbt_high_groups_hz = None,
        cbt_use_event_gate = True,
        cbt_use_temporal_derivative = True,
        cbt_use_depthwise_temporal_conv = True,
        cbt_temporal_kernel = 5,
        cbt_zero_init = True,
        cbt_init_scale = 0.0,
        cbt_dropout = 0.0,
        use_output_mel_adapter = False,
        first_transformer_depth = 1,
        first_transformer_heads = 16,
        first_transformer_dim_head = 64,
        first_transformer_ff_mult = 4,
        second_transformer_depth = 1,
        second_transformer_heads = 8,
        second_transformer_dim_head = 64,
        second_transformer_ff_mult = 2,
        melconv_bridge_channels = 64,
        melconv_bridge_blocks = 3,
        melconv_bridge_kernel_time = 3,
        melconv_bridge_kernel_freq = 9,
        melconv_bridge_zero_init = False,
        melconv_bridge_final_init_std = 1e-4,
        melconv_bridge_scale_init = 1.0,
        melconv_bridge_scale_ramp_steps = 5000,
        melconv_bridge_use_hidden_mel = True,
        melconv_bridge_use_zt = True,
        melconv_bridge_use_cond = True,
        melconv_bridge_use_mask = True,
        melconv_bridge_use_freq_pos = True,
        melconv_bridge_use_cutoff_dist = True,
        output_mel_adapter_channels = 64,
        output_mel_adapter_blocks = 2,
        output_mel_adapter_kernel_time = 3,
        output_mel_adapter_kernel_freq = 9,
        output_mel_adapter_zero_init = False,
        output_mel_adapter_final_init_std = 1e-4,
        output_mel_adapter_scale_ramp_steps = 5000,
        output_mel_adapter_use_v_base = True,
        output_mel_adapter_use_zt = True,
        output_mel_adapter_use_cond = True,
        output_mel_adapter_use_mask = True,
        output_mel_adapter_use_freq_pos = True,
        output_mel_adapter_use_cutoff_dist = True,
        output_mel_adapter_use_band_gates = True,
        output_mel_adapter_near_hz = 4000.0,
        output_mel_adapter_mid_hz = 10000.0,
        output_mel_adapter_band_gate_init = 1.0,
    ):
        super().__init__()
        dim_in = default(dim_in, dim)
        
        self.architecture = architecture
        self.dim_in = dim_in
        self.input_channels = input_channels
        self.condition_with_hf_mask = condition_with_hf_mask
        self.condition_with_cutoff_embedding = condition_with_cutoff_embedding
        self.use_adaln_zero = use_adaln_zero
        self.adaln_zero_gate_log = adaln_zero_gate_log
        self.use_interleaved_melconv = bool(use_interleaved_melconv)
        self.use_melconv_bridge = bool(use_melconv_bridge)
        self.cbt_bridge_enabled = bool(cbt_bridge_enabled)
        self.use_output_mel_adapter = bool(use_output_mel_adapter)
        self.last_melconv_debug = {}
        self.last_adaln_debug = {
            "adaln_gate_msa_mean": 0.0,
            "adaln_gate_msa_abs_mean": 0.0,
            "adaln_gate_ffn_mean": 0.0,
            "adaln_gate_ffn_abs_mean": 0.0,
        }

        if self.architecture=='transformer':
            time_hidden_dim = default(time_hidden_dim, dim)
        elif self.architecture=='convnext':
            time_hidden_dim = default(time_hidden_dim, dim)
        else:
            raise ValueError("Choose approriate architecture")
        
        self.audio_enc_dec = audio_enc_dec 

        self.proj_in = nn.Identity()    
        
        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )

        self.cutoff_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        ) if condition_with_cutoff_embedding else None

        self.dim_cond_emb = dim_cond_emb
        self.to_embed = nn.Linear(dim_in * input_channels + dim_cond_emb, dim)    
        self.null_cond = nn.Parameter(torch.zeros(dim_in), requires_grad = False)
        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        if self.architecture =='transformer':
            if self.use_interleaved_melconv:
                if use_adaln_zero:
                    raise ValueError("Scratch interleaved melconv does not use AdaLN-Zero")
                self.block0_full = DepthFlexibleTransformer(
                    dim = dim,
                    depth = first_transformer_depth,
                    dim_head = first_transformer_dim_head,
                    heads = first_transformer_heads,
                    ff_mult = first_transformer_ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout= attn_dropout,
                    attn_flash = attn_flash,
                    attn_qk_norm = attn_qk_norm,
                    adaptive_rmsnorm = True,
                    adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
                    use_gateloop_layers = use_gateloop_layers
                )
                self.melconv_bridge = InterleavedMelConvBridge(
                    dim=dim,
                    dim_in=dim_in,
                    channels=melconv_bridge_channels,
                    blocks=melconv_bridge_blocks,
                    kernel_time=melconv_bridge_kernel_time,
                    kernel_freq=melconv_bridge_kernel_freq,
                    zero_init=melconv_bridge_zero_init,
                    final_init_std=melconv_bridge_final_init_std,
                    scale_init=melconv_bridge_scale_init,
                    scale_ramp_steps=melconv_bridge_scale_ramp_steps,
                    use_hidden_mel=melconv_bridge_use_hidden_mel,
                    use_zt=melconv_bridge_use_zt,
                    use_cond=melconv_bridge_use_cond,
                    use_mask=melconv_bridge_use_mask,
                    use_freq_pos=melconv_bridge_use_freq_pos,
                    use_cutoff_dist=melconv_bridge_use_cutoff_dist,
                ) if (self.use_melconv_bridge and not self.cbt_bridge_enabled) else None
                self.cbt_bridge = LightweightCBTBridge(
                    dim=dim,
                    dim_in=dim_in,
                    target_sr=self.audio_enc_dec.sampling_rate if exists(self.audio_enc_dec) else 48000,
                    hidden_dim=cbt_hidden_dim,
                    low_groups_hz=default(cbt_low_groups_hz, ((0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000))),
                    high_groups_hz=default(cbt_high_groups_hz, ((4000, 6000), (6000, 8000), (8000, 12000), (12000, 16000), (16000, 24000))),
                    use_event_gate=cbt_use_event_gate,
                    use_temporal_derivative=cbt_use_temporal_derivative,
                    use_depthwise_temporal_conv=cbt_use_depthwise_temporal_conv,
                    temporal_kernel=cbt_temporal_kernel,
                    zero_init=cbt_zero_init,
                    init_scale=cbt_init_scale,
                    dropout=cbt_dropout,
                ) if self.cbt_bridge_enabled else None
                self.block1_lite = DepthFlexibleTransformer(
                    dim = dim,
                    depth = second_transformer_depth,
                    dim_head = second_transformer_dim_head,
                    heads = second_transformer_heads,
                    ff_mult = second_transformer_ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout= attn_dropout,
                    attn_flash = attn_flash,
                    attn_qk_norm = attn_qk_norm,
                    adaptive_rmsnorm = True,
                    adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
                    use_gateloop_layers = use_gateloop_layers
                )
            else:
                if use_adaln_zero:
                    self.transformer = AdaLNZeroTransformer(
                        dim = dim,
                        depth = depth,
                        dim_head = dim_head,
                        heads = heads,
                        ff_mult = ff_mult,
                        ff_dropout = ff_dropout,
                        attn_dropout= attn_dropout,
                        attn_flash = attn_flash,
                        attn_qk_norm = attn_qk_norm,
                        adaptive_rmsnorm = False,
                        adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
                        adaln_zero_cond_dim = time_hidden_dim,
                        adaln_zero_init = adaln_zero_init,
                        use_gateloop_layers = use_gateloop_layers
                    )
                else:
                    self.transformer = Transformer(
                        dim = dim,
                        depth = depth,
                        dim_head = dim_head,
                        heads = heads,
                        ff_mult = ff_mult,
                        ff_dropout = ff_dropout,
                        attn_dropout= attn_dropout,
                        attn_flash = attn_flash,
                        attn_qk_norm = attn_qk_norm,
                        adaptive_rmsnorm = True,
                        adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
                        use_gateloop_layers = use_gateloop_layers
                    )   

        elif self.architecture=='convnext':
            intermediate_dim = dim * 3
            num_layers = 8
            layer_scale_init_value = 1
            self.convnext = nn.ModuleList(
                [
                    ConvNeXtBlock(
                        dim=dim,
                        intermediate_dim=intermediate_dim,
                        layer_scale_init_value=layer_scale_init_value,
                        hidden_dim=time_hidden_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
    
        dim_out = dim_in
        self.to_pred = nn.Linear(dim, dim_out, bias = False)
        self.output_mel_adapter = OutputMelAdapter(
            dim_in=dim_in,
            channels=output_mel_adapter_channels,
            blocks=output_mel_adapter_blocks,
            kernel_time=output_mel_adapter_kernel_time,
            kernel_freq=output_mel_adapter_kernel_freq,
            zero_init=output_mel_adapter_zero_init,
            final_init_std=output_mel_adapter_final_init_std,
            scale_ramp_steps=output_mel_adapter_scale_ramp_steps,
            use_v_base=output_mel_adapter_use_v_base,
            use_zt=output_mel_adapter_use_zt,
            use_cond=output_mel_adapter_use_cond,
            use_mask=output_mel_adapter_use_mask,
            use_freq_pos=output_mel_adapter_use_freq_pos,
            use_cutoff_dist=output_mel_adapter_use_cutoff_dist,
            use_band_gates=output_mel_adapter_use_band_gates,
            near_hz=output_mel_adapter_near_hz,
            mid_hz=output_mel_adapter_mid_hz,
            band_gate_init=output_mel_adapter_band_gate_init,
        ) if self.use_output_mel_adapter else None

    @property
    def device(self):
        return next(self.parameters()).device

    def hz_to_mel(self,f):
        if isinstance(f, (list, numpy.ndarray)): 
            f = numpy.array(f) 
        return 2595 * numpy.log10(1 + f / 700)

    def mel_bin_index(self, frequency, sample_rate, num_mel_bins):
        nyquist = sample_rate / 2
        m_min = self.hz_to_mel(0)
        m_max = self.hz_to_mel(nyquist)
        mel_value = self.hz_to_mel(frequency)
        bin_index = numpy.floor((mel_value - m_min) / (m_max - m_min) * num_mel_bins)
        if isinstance(bin_index, numpy.ndarray):
            bin_index = bin_index.astype(int)  
        else:
            bin_index = int(bin_index) 
        return bin_index

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale    
    
    def forward(
        self,
        x,
        *,
        times,
        self_attn_mask = None,
        cond_drop_prob = 0.1,
        target = None,
        cond = None,
        cond_mask = None,
        cond_freq_masking = False,
        random_sr = None,
        weighted_loss = False,
        cutoff_bins = None,
        hf_mask = None,
        cutoff_ratio = None,
        use_hf_frequency_weight = False,
        hf_weight_min = 1.0,
        hf_weight_max = 1.5,
        hf_weight_mode = "linear",
        hf_weight_cap_hz_above_cutoff = 4000.0,
        loss_hf_l1_weight = 0.0,
        edge_continuity_loss_weight = 0.0,
        edge_continuity_bins = 4,
        cutoff_hz = None,
        use_freq_prediction_loss = False,
        freq_prediction_loss_weight = 0.0,
        use_freq_gradient_loss = False,
        freq_gradient_loss_weight = 0.0,
        melconv_bridge_l1_weight = 0.0,
        output_mel_adapter_l1_weight = 0.0,
        global_step = None,
    ):

        x = self.proj_in(x) 
        flow_state = x
        cond = default(cond, target)
        
        if exists(cond):
            cond = self.proj_in(cond) 

        # shapes
        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]


        # auto manage shape of times, for odeint times
        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0]) 
        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0]) 

        # Cond frequency masking 
        if cond_freq_masking:
            if self.training:
                cond = mask_for_freqency(cond, batch, seq_len, cond_dim, device=self.device)
            else:
                cond_freq_mask = torch.ones((batch, seq_len,cond_dim), device = cond.device, dtype =torch.bool)
                cond = cond * cond_freq_mask
        else:
            pass         

        cond_raw = cond

        # Classifier free guidance 
        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device)
            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )
                      
        # x.shape : [B, Time, channel]
        # cond.shape : [B, Time, channel]
        to_concat = [x, cond]

        if self.condition_with_hf_mask:
            if not exists(hf_mask):
                raise ValueError("condition_with_hf_mask=True requires hf_mask")
            hf_mask = orient_highband_mask(hf_mask, x).expand_as(x)
            to_concat.append(hf_mask)
        
        # embed.shape : [B, Time, dim_in * input_channels]
        embed = torch.cat(to_concat, dim = -1) 
        
        x = self.to_embed(embed)
        x = self.conv_embed(x, mask = self_attn_mask) + x

        time_emb = self.sinu_pos_emb(times)
        if exists(self.cutoff_pos_emb):
            if not exists(cutoff_ratio):
                raise ValueError("condition_with_cutoff_embedding=True requires cutoff_ratio")
            if not torch.is_tensor(cutoff_ratio):
                cutoff_ratio = torch.tensor(cutoff_ratio, device=times.device, dtype=times.dtype)
            cutoff_ratio = cutoff_ratio.to(device=times.device, dtype=times.dtype)
            if cutoff_ratio.ndim == 0:
                cutoff_ratio = repeat(cutoff_ratio, '-> b', b=batch)
            time_emb = time_emb + self.cutoff_pos_emb(cutoff_ratio)
        
        if self.architecture=='transformer':
            self.last_melconv_loss_tensors = {
                "loss_melconv_bridge_l1": x.new_tensor(0.0),
                "loss_output_mel_adapter_l1": x.new_tensor(0.0),
            }
            if self.use_interleaved_melconv:
                x = self.block0_full(x, mask = self_attn_mask, adaptive_rmsnorm_cond = time_emb)
                bridge_debug = {
                    "melconv_bridge_mel_delta_l1": 0.0,
                    "melconv_bridge_hidden_delta_l1": 0.0,
                    "melconv_bridge_hidden_base_l1": 0.0,
                    "melconv_bridge_delta_base_ratio": 0.0,
                    "melconv_bridge_lowband_delta_l1": 0.0,
                    "melconv_bridge_scale": 0.0,
                    "melconv_bridge_ramp": 0.0,
                    "cbt_alpha": 0.0,
                    "cbt_event_gate_mean": 0.0,
                    "cbt_delta_l1": 0.0,
                    "cbt_injected_delta_l1": 0.0,
                    "cbt_transport_entropy": 0.0,
                    "cbt_active_high_4_6k": 0.0,
                    "cbt_active_high_6_8k": 0.0,
                    "cbt_active_high_8_12k": 0.0,
                    "cbt_active_high_12_16k": 0.0,
                    "cbt_active_high_16_24k": 0.0,
                }
                if exists(getattr(self, "cbt_bridge", None)):
                    if not exists(cutoff_hz):
                        if not exists(cutoff_ratio):
                            raise ValueError("cbt_bridge_enabled=True requires cutoff_hz or cutoff_ratio")
                        cutoff_hz = cutoff_ratio * (self.audio_enc_dec.sampling_rate / 2)
                    x, bridge_debug = self.cbt_bridge(
                        h=x,
                        m_up=cond_raw,
                        cutoff_hz=cutoff_hz,
                        segment_mask=self_attn_mask,
                        cutoff_emb=time_emb,
                    )
                elif exists(self.melconv_bridge):
                    hidden_delta, bridge_debug, bridge_l1 = self.melconv_bridge(
                        hidden=x,
                        z_t=flow_state,
                        cond_raw=cond_raw,
                        hf_mask=hf_mask,
                        cutoff_ratio=cutoff_ratio,
                        global_step=global_step,
                    )
                    self.last_melconv_loss_tensors["loss_melconv_bridge_l1"] = bridge_l1
                    x = x + hidden_delta
                x = self.block1_lite(x, mask = self_attn_mask, adaptive_rmsnorm_cond = time_emb)
                self.last_melconv_debug = bridge_debug
            else:
                x = self.transformer(x, mask = self_attn_mask, adaptive_rmsnorm_cond = time_emb)        

        elif self.architecture=='convnext':
            x = x.transpose(1,2)
            for convnext_block in self.convnext:
                x = convnext_block(x, cond=time_emb)
        
            x = x.transpose(1,2)
            x = self.final_layer_norm(x)

        # Protect NaN
        logging.info(f"After transformer: {x}")
        if torch.isnan(x).any():
            print(x)
            logging.error("NaN detected after main architecture")
            
        x = self.to_pred(x)
        v_base = x
        if exists(self.output_mel_adapter):
            if not exists(hf_mask):
                raise ValueError("use_output_mel_adapter=True requires hf_mask")
            delta_v, output_debug, output_l1 = self.output_mel_adapter(
                v_base=v_base,
                z_t=flow_state,
                cond_raw=cond_raw,
                hf_mask=hf_mask,
                cutoff_ratio=cutoff_ratio,
                global_step=global_step,
            )
            x = v_base + delta_v
            self.last_melconv_debug.update(output_debug)
            self.last_melconv_debug["loss_output_mel_adapter_l1"] = output_debug.get("output_adapter_delta_l1", 0.0)
            self.last_melconv_loss_tensors["loss_output_mel_adapter_l1"] = output_l1
        else:
            self.last_melconv_debug.update({
                "output_adapter_delta_l1": 0.0,
                "output_adapter_v_base_l1": 0.0,
                "output_adapter_delta_base_ratio": 0.0,
                "output_adapter_lowband_delta_l1": 0.0,
                "output_adapter_near_delta_l1": 0.0,
                "output_adapter_mid_delta_l1": 0.0,
                "output_adapter_far_delta_l1": 0.0,
                "output_adapter_band_gate_near": 0.0,
                "output_adapter_band_gate_mid": 0.0,
                "output_adapter_band_gate_far": 0.0,
                "output_adapter_band_weight_mean": 0.0,
                "output_adapter_band_weight_max": 0.0,
                "output_adapter_ramp": 0.0,
            })
            self.last_melconv_debug.setdefault("loss_output_mel_adapter_l1", 0.0)
        self.last_melconv_debug.setdefault("loss_melconv_bridge_l1", self.last_melconv_debug.get("melconv_bridge_mel_delta_l1", 0.0))
        v_pred = x
        self.last_v_pred_shape = tuple(x.shape)
        if (not self.use_interleaved_melconv) and self.use_adaln_zero and self.adaln_zero_gate_log and hasattr(self.transformer, "get_adaln_gate_stats"):
            self.last_adaln_debug = self.transformer.get_adaln_gate_stats()
        else:
            self.last_adaln_debug = {
                "adaln_gate_msa_mean": 0.0,
                "adaln_gate_msa_abs_mean": 0.0,
                "adaln_gate_ffn_mean": 0.0,
                "adaln_gate_ffn_abs_mean": 0.0,
            }
                
        # Protect NaN
        logging.info(f"After predict: {x}")
        if torch.isnan(x).any():
            print(x)
            logging.error("NaN detected after last projection layer")


        # if no target passed in, just return logits
        # for inference mode
        if not exists(target):

            return x

        loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)
        segment_mask_valid_ratio = float(loss_mask.float().mean().detach().cpu()) if exists(loss_mask) else 1.0
        loss_freq_pred = x.new_tensor(0.0)
        loss_freq_grad = x.new_tensor(0.0)
        freq_aux_enabled = (
            bool(use_freq_prediction_loss or use_freq_gradient_loss)
            and exists(hf_mask)
            and exists(cond_raw)
            and exists(target)
        )
        if freq_aux_enabled:
            hf_mask_for_loss = orient_highband_mask(hf_mask, v_pred).expand_as(v_pred)
            t_aux = times
            if t_aux.ndim == 0:
                t_aux = repeat(t_aux, '-> b', b=batch)
            if t_aux.ndim == 1:
                t_aux = rearrange(t_aux, 'b -> b 1 1')
            r_pred = flow_state + (1.0 - t_aux) * v_pred
            r_target = flow_state + (1.0 - t_aux) * target
            M_pred = cond_raw + hf_mask_for_loss * r_pred
            M_target = cond_raw + hf_mask_for_loss * r_target
            loss_freq_pred_map = torch.abs(M_pred - M_target)
            if exists(loss_mask):
                valid = rearrange(loss_mask, 'b t -> b t 1').to(loss_freq_pred_map.dtype)
                loss_freq_pred = (loss_freq_pred_map * valid).sum() / ((hf_mask_for_loss * valid).sum().clamp_min(1.0))
            else:
                loss_freq_pred = loss_freq_pred_map.sum() / hf_mask_for_loss.sum().clamp_min(1.0)

            d_pred = M_pred[:, :, 1:] - M_pred[:, :, :-1]
            d_target = M_target[:, :, 1:] - M_target[:, :, :-1]
            hf_pair_mask = torch.maximum(hf_mask_for_loss[:, :, 1:], hf_mask_for_loss[:, :, :-1])
            loss_freq_grad_map = torch.abs(d_pred - d_target) * hf_pair_mask
            if exists(loss_mask):
                valid = rearrange(loss_mask, 'b t -> b t 1').to(loss_freq_grad_map.dtype)
                loss_freq_grad = (loss_freq_grad_map * valid).sum() / ((hf_pair_mask * valid).sum().clamp_min(1.0))
            else:
                loss_freq_grad = loss_freq_grad_map.sum() / hf_pair_mask.sum().clamp_min(1.0)

        def add_freq_aux(loss_main):
            loss_total = loss_main
            if use_freq_prediction_loss:
                loss_total = loss_total + float(freq_prediction_loss_weight) * loss_freq_pred
            if use_freq_gradient_loss:
                loss_total = loss_total + float(freq_gradient_loss_weight) * loss_freq_grad
            return loss_total

        melconv_loss_tensors = getattr(self, "last_melconv_loss_tensors", {})
        loss_melconv_bridge_l1 = melconv_loss_tensors.get("loss_melconv_bridge_l1", x.new_tensor(0.0))
        loss_output_mel_adapter_l1 = melconv_loss_tensors.get("loss_output_mel_adapter_l1", x.new_tensor(0.0))

        def add_bridge_output_aux(loss_main):
            return (
                loss_main
                + float(melconv_bridge_l1_weight) * loss_melconv_bridge_l1
                + float(output_mel_adapter_l1_weight) * loss_output_mel_adapter_l1
            )

        def add_all_aux(loss_main):
            return add_freq_aux(add_bridge_output_aux(loss_main))

        if exists(hf_mask):
            hf_mask_for_debug = orient_highband_mask(hf_mask, x).expand_as(x)
            err = x - target
            high_err = hf_mask_for_debug * err
            low_err = (1 - hf_mask_for_debug) * err
            target_hf = hf_mask_for_debug * target
            pred_hf = hf_mask_for_debug * x
            target_hf_l1 = torch.mean(torch.abs(target_hf)).clamp(min=1e-8)
            mel_freqs = None
            if exists(getattr(self, "audio_enc_dec", None)):
                mel_freqs = mel_center_frequencies(
                    self.audio_enc_dec.n_mels,
                    self.audio_enc_dec.sampling_rate,
                    self.audio_enc_dec.f_min,
                    self.audio_enc_dec.f_max,
                    device=x.device,
                    dtype=x.dtype,
                )
            hf_weight = build_hf_frequency_weight(
                hf_mask_for_debug,
                hf_weight_min,
                hf_weight_max,
                mode=hf_weight_mode,
                cutoff_hz=cutoff_hz,
                mel_freqs=mel_freqs,
                cap_hz_above_cutoff=hf_weight_cap_hz_above_cutoff,
            )
            loss_mask_expanded_for_debug = (
                rearrange(loss_mask, 'b n -> b n 1').to(err.dtype)
                if exists(loss_mask)
                else torch.ones_like(hf_mask_for_debug)
            )
            valid_hf_mask_for_debug = hf_mask_for_debug * loss_mask_expanded_for_debug
            weighted_mask = valid_hf_mask_for_debug * hf_weight
            loss_flow_masked = ((hf_mask_for_debug * err) ** 2 * loss_mask_expanded_for_debug).sum() / valid_hf_mask_for_debug.sum().clamp_min(1e-6)
            loss_flow_hf_weighted = (err ** 2 * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
            # The model predicts the flow velocity in this forward pass.  This
            # L1 term is therefore a high-band velocity L1 proxy for residual
            # detail quality, logged as loss_hf_l1 for the HF-Boost experiment.
            loss_hf_l1 = (torch.abs(err) * valid_hf_mask_for_debug).sum() / valid_hf_mask_for_debug.sum().clamp_min(1e-6)
            # Single explicit scalar for the HF-Boost objective.
            # When use_hf_frequency_weight=True, this exact tensor is returned
            # below and therefore used by accelerator.backward(...).
            loss_edge_continuity = compute_edge_continuity_loss(
                x,
                target,
                hf_mask_for_debug,
                edge_bins=int(edge_continuity_bins),
                loss_mask=loss_mask,
            ) if float(edge_continuity_loss_weight) > 0 else x.new_tensor(0.0)
            loss_total_hfboost_main = (
                loss_flow_hf_weighted
                + float(loss_hf_l1_weight) * loss_hf_l1
                + float(edge_continuity_loss_weight) * loss_edge_continuity
            )
            loss_total_hfboost = add_bridge_output_aux(loss_total_hfboost_main)
            loss_total_after_freq_aux = add_freq_aux(loss_total_hfboost)
            loss_used_for_backward = loss_total_after_freq_aux if use_hf_frequency_weight else add_all_aux(loss_flow_masked)
            self.last_highband_loss_debug = {
                "v_pred_hf_l1": float(torch.mean(torch.abs(pred_hf)).detach().cpu()),
                "v_target_hf_l1": float(target_hf_l1.detach().cpu()),
                "pred_target_l1_ratio": float((torch.mean(torch.abs(pred_hf)) / target_hf_l1).detach().cpu()),
                "loss_lowband_debug": float(torch.mean(low_err ** 2).detach().cpu()),
                "loss_highband_debug": float(torch.mean(high_err ** 2).detach().cpu()),
                "loss_unmasked_debug": float(torch.mean(err ** 2).detach().cpu()),
                "loss_flow_masked": float(loss_flow_masked.detach().cpu()),
                "loss_flow_hf_weighted": float(loss_flow_hf_weighted.detach().cpu()),
                "loss_flow": float((loss_flow_hf_weighted if use_hf_frequency_weight else loss_flow_masked).detach().cpu()),
                "loss_hf_l1": float(loss_hf_l1.detach().cpu()),
                "loss_hf_velocity_l1": float(loss_hf_l1.detach().cpu()),
                "loss_edge_continuity": float(loss_edge_continuity.detach().cpu()),
                "loss_freq_pred": float(loss_freq_pred.detach().cpu()),
                "loss_freq_grad": float(loss_freq_grad.detach().cpu()),
                "freq_prediction_loss_weight": float(freq_prediction_loss_weight),
                "freq_gradient_loss_weight": float(freq_gradient_loss_weight),
                "use_freq_prediction_loss": float(bool(use_freq_prediction_loss)),
                "use_freq_gradient_loss": float(bool(use_freq_gradient_loss)),
                "loss_total_main": float(loss_total_hfboost_main.detach().cpu()),
                "loss_total_before_freq_aux": float(loss_total_hfboost.detach().cpu()),
                "loss_total_after_freq_aux": float(loss_total_after_freq_aux.detach().cpu()),
                "loss_total": float(loss_total_after_freq_aux.detach().cpu()),
                "loss_total_hfboost": float(loss_total_hfboost.detach().cpu()),
                "loss_melconv_bridge_l1": float(loss_melconv_bridge_l1.detach().cpu()),
                "loss_output_mel_adapter_l1": float(loss_output_mel_adapter_l1.detach().cpu()),
                "segment_mask_valid_ratio": segment_mask_valid_ratio,
                "loss_used_for_backward": float(loss_used_for_backward.detach().cpu()),
                "hfboost_enabled": float(bool(use_hf_frequency_weight)),
                "hf_weight_min": float(hf_weight_min),
                "hf_weight_max": float(hf_weight_max),
                "hf_weight_mode": str(hf_weight_mode),
                "hf_weight_cap_hz_above_cutoff": float(hf_weight_cap_hz_above_cutoff),
                "edge_bins": int(edge_continuity_bins),
            }
            self.last_highband_loss_debug.update(getattr(self, "last_adaln_debug", {}))
            self.last_highband_loss_debug.update(getattr(self, "last_melconv_debug", {}))

        if not exists(loss_mask):
            
            if exists(hf_mask):
                hf_mask = orient_highband_mask(hf_mask, x).expand_as(x)
                err = x - target
                if use_hf_frequency_weight:
                    mel_freqs = None
                    if exists(getattr(self, "audio_enc_dec", None)):
                        mel_freqs = mel_center_frequencies(
                            self.audio_enc_dec.n_mels,
                            self.audio_enc_dec.sampling_rate,
                            self.audio_enc_dec.f_min,
                            self.audio_enc_dec.f_max,
                            device=x.device,
                            dtype=x.dtype,
                        )
                    hf_weight = build_hf_frequency_weight(
                        hf_mask,
                        hf_weight_min,
                        hf_weight_max,
                        mode=hf_weight_mode,
                        cutoff_hz=cutoff_hz,
                        mel_freqs=mel_freqs,
                        cap_hz_above_cutoff=hf_weight_cap_hz_above_cutoff,
                    )
                    weighted_mask = hf_mask * hf_weight
                    loss_flow_hf_weighted = (err ** 2 * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
                    loss_hf_l1 = (torch.abs(err) * hf_mask).sum() / hf_mask.sum().clamp_min(1e-6)
                    loss_edge_continuity = compute_edge_continuity_loss(
                        x,
                        target,
                        hf_mask,
                        edge_bins=int(edge_continuity_bins),
                        loss_mask=None,
                    ) if float(edge_continuity_loss_weight) > 0 else x.new_tensor(0.0)
                    loss_total = (
                        loss_flow_hf_weighted
                        + float(loss_hf_l1_weight) * loss_hf_l1
                        + float(edge_continuity_loss_weight) * loss_edge_continuity
                    )
                    loss_return = add_all_aux(loss_total)
                    if hasattr(self, "last_highband_loss_debug"):
                        self.last_highband_loss_debug.update({
                            "loss_total_main": float(loss_total.detach().cpu()),
                            "loss_total_before_freq_aux": float(add_bridge_output_aux(loss_total).detach().cpu()),
                            "loss_total_after_freq_aux": float(loss_return.detach().cpu()),
                            "loss_total": float(loss_return.detach().cpu()),
                            "loss_total_hfboost": float(add_bridge_output_aux(loss_total).detach().cpu()),
                            "loss_used_for_backward": float(loss_return.detach().cpu()),
                        })
                    return loss_return
                masked_loss = (hf_mask * err) ** 2
                loss_main = masked_loss.sum() / hf_mask.sum().clamp_min(1e-6)
                loss_return = add_all_aux(loss_main)
                if hasattr(self, "last_highband_loss_debug"):
                    self.last_highband_loss_debug.update({
                        "loss_total_main": float(loss_main.detach().cpu()),
                        "loss_total_before_freq_aux": float(add_bridge_output_aux(loss_main).detach().cpu()),
                        "loss_total_after_freq_aux": float(loss_return.detach().cpu()),
                        "loss_total": float(loss_return.detach().cpu()),
                        "loss_used_for_backward": float(loss_return.detach().cpu()),
                    })
                return loss_return

            if weighted_loss == False:
                return add_bridge_output_aux(F.mse_loss(x, target))
            
            elif weighted_loss == True:
                low_weight = 1.0
                high_weight = 2.0
                n_mels = self.audio_enc_dec.n_mels
                weight = torch.ones(batch,n_mels) * low_weight
                if isinstance(cutoff_bins, numpy.ndarray):
                    for i, bin_idx in enumerate(cutoff_bins):
                        weight[i, bin_idx:] = high_weight
                else:
                    exit()
                    weight[cutoff_bins:] = high_weight
                        
                weight = weight.unsqueeze(1).expand(batch, seq_len, n_mels).cuda()
                mse_loss = F.mse_loss(x, target, reduction='none') 
                weighted_mse_loss = mse_loss * weight
                mean_loss = weighted_mse_loss.mean()
                return add_bridge_output_aux(mean_loss)

        if exists(hf_mask):
            hf_mask = orient_highband_mask(hf_mask, x).expand_as(x)
            err = x - target
            if use_hf_frequency_weight:
                mel_freqs = None
                if exists(getattr(self, "audio_enc_dec", None)):
                    mel_freqs = mel_center_frequencies(
                        self.audio_enc_dec.n_mels,
                        self.audio_enc_dec.sampling_rate,
                        self.audio_enc_dec.f_min,
                        self.audio_enc_dec.f_max,
                        device=x.device,
                        dtype=x.dtype,
                    )
                hf_weight = build_hf_frequency_weight(
                    hf_mask,
                    hf_weight_min,
                    hf_weight_max,
                    mode=hf_weight_mode,
                    cutoff_hz=cutoff_hz,
                    mel_freqs=mel_freqs,
                    cap_hz_above_cutoff=hf_weight_cap_hz_above_cutoff,
                )
                weighted_mask = hf_mask * hf_weight
                loss = (err ** 2) * weighted_mask
            else:
                loss = (hf_mask * err) ** 2
            loss_mask_expanded = rearrange(loss_mask, 'b n -> b n 1').to(loss.dtype)
            loss = loss * loss_mask_expanded
            den_mask = weighted_mask if use_hf_frequency_weight else hf_mask
            den = (den_mask * loss_mask_expanded).sum().clamp_min(1e-6)
            loss_value = loss.sum() / den
            if use_hf_frequency_weight:
                l1_num = (torch.abs(err) * hf_mask * loss_mask_expanded).sum()
                l1_den = (hf_mask * loss_mask_expanded).sum().clamp_min(1e-6)
                loss_value = loss_value + float(loss_hf_l1_weight) * (l1_num / l1_den)
            if float(edge_continuity_loss_weight) > 0:
                loss_value = loss_value + float(edge_continuity_loss_weight) * compute_edge_continuity_loss(
                    x,
                    target,
                    hf_mask,
                    edge_bins=int(edge_continuity_bins),
                    loss_mask=loss_mask,
                )
            loss_return = add_all_aux(loss_value)
            if hasattr(self, "last_highband_loss_debug"):
                self.last_highband_loss_debug.update({
                    "loss_total_main": float(loss_value.detach().cpu()),
                    "loss_total_before_freq_aux": float(add_bridge_output_aux(loss_value).detach().cpu()),
                    "loss_total_after_freq_aux": float(loss_return.detach().cpu()),
                    "loss_total": float(loss_return.detach().cpu()),
                    "loss_used_for_backward": float(loss_return.detach().cpu()),
                })
            return loss_return
        else:
            loss = F.mse_loss(x, target, reduction = 'none')
        loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean
        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        return add_bridge_output_aux(loss.mean())

def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))

class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        cfm_method = 'basic_cfm',
        torchdiffeq_ode_method = 'midpoint',   # [euler, midpoint]
        torchode_method_klass = to.Tsit5,      
        cond_drop_prob = 0.,
        use_highband_residual_flow = False,
        highband_mask_softness_hz = 500.0,
        condition_with_hf_mask = False,
        condition_with_cutoff_embedding = False,
        target_type = "mel",
        use_hf_frequency_weight = False,
        hf_weight_min = 1.0,
        hf_weight_max = 1.5,
        hf_weight_mode = "linear",
        hf_weight_cap_hz_above_cutoff = 4000.0,
        loss_hf_l1_weight = 0.0,
        edge_continuity_loss_weight = 0.0,
        edge_continuity_bins = 4,
        residual_noise_scale = 1.0,
        seam_smoothing_enabled = False,
        seam_smoothing_kernel_size = 3,
        seam_smoothing_bins = 4,
        use_freq_prediction_loss = False,
        freq_prediction_loss_weight = 0.0,
        use_freq_gradient_loss = False,
        freq_gradient_loss_weight = 0.0,
        melconv_bridge_l1_weight = 0.0,
        output_mel_adapter_l1_weight = 0.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.flowhigh = flowhigh
        self.cond_drop_prob = cond_drop_prob
        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass
        self.cfm_method = cfm_method
        self.use_highband_residual_flow = use_highband_residual_flow
        self.highband_mask_softness_hz = highband_mask_softness_hz
        self.condition_with_hf_mask = condition_with_hf_mask
        self.condition_with_cutoff_embedding = condition_with_cutoff_embedding
        self.target_type = target_type
        self.use_hf_frequency_weight = use_hf_frequency_weight
        self.hf_weight_min = hf_weight_min
        self.hf_weight_max = hf_weight_max
        self.hf_weight_mode = hf_weight_mode
        self.hf_weight_cap_hz_above_cutoff = hf_weight_cap_hz_above_cutoff
        self.loss_hf_l1_weight = loss_hf_l1_weight
        self.edge_continuity_loss_weight = edge_continuity_loss_weight
        self.edge_continuity_bins = edge_continuity_bins
        self.residual_noise_scale = residual_noise_scale
        self.seam_smoothing_enabled = seam_smoothing_enabled
        self.seam_smoothing_kernel_size = seam_smoothing_kernel_size
        self.seam_smoothing_bins = seam_smoothing_bins
        self.use_freq_prediction_loss = use_freq_prediction_loss
        self.freq_prediction_loss_weight = freq_prediction_loss_weight
        self.use_freq_gradient_loss = use_freq_gradient_loss
        self.freq_gradient_loss_weight = freq_gradient_loss_weight
        self.melconv_bridge_l1_weight = float(melconv_bridge_l1_weight)
        self.output_mel_adapter_l1_weight = float(output_mel_adapter_l1_weight)
        self.last_highband_debug = {}
        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path, strict = True):
        # return pkg so the trainer can access it
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_model_state_dict(pkg['model'], strict = strict)
        return pkg

    def load_model_state_dict(self, state_dict, strict=True):
        if strict:
            return self.load_state_dict(state_dict, strict=True)

        current = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        loaded_ratio = len(compatible) / max(len(current), 1)
        if loaded_ratio < 0.8:
            print(f"[load warning] compatible checkpoint keys ratio is low: {loaded_ratio:.3f}")
        return self.load_state_dict(compatible, strict=False)

    def _cutoff_hz_tensor(self, batch, dtype, device, input_sampling_rate=None, random_sr=None, cutoff_hz=None):
        if exists(cutoff_hz):
            cutoff = cutoff_hz
        elif exists(random_sr):
            cutoff = random_sr
            if isinstance(cutoff, (list, tuple)):
                cutoff = torch.tensor(cutoff, dtype=dtype, device=device)
            else:
                cutoff = torch.as_tensor(cutoff, dtype=dtype, device=device)
            cutoff = cutoff / 2
        elif exists(input_sampling_rate):
            cutoff = float(input_sampling_rate) / 2
        else:
            cutoff = self.flowhigh.audio_enc_dec.sampling_rate / 2

        cutoff = torch.as_tensor(cutoff, dtype=dtype, device=device)
        if cutoff.ndim == 0:
            cutoff = repeat(cutoff, '-> b', b=batch)
        return cutoff

    def _build_hf_mask_for(self, mel, cutoff_hz):
        audio_enc_dec = self.flowhigh.audio_enc_dec
        mask = build_highband_mel_mask(
            n_mels=audio_enc_dec.n_mels,
            sample_rate=audio_enc_dec.sampling_rate,
            f_min=audio_enc_dec.f_min,
            f_max=audio_enc_dec.f_max,
            cutoff_hz=cutoff_hz,
            softness_hz=self.highband_mask_softness_hz,
            device=mel.device,
            dtype=mel.dtype
        )
        return orient_highband_mask(mask, mel)

    # For mel repalcement
    def locate_cutoff_freq(self, mel, percentile=0.9995):
        def find_cutoff(x, percentile=0.99):
            percentile = x[-1] * percentile
            for i in range(1, x.shape[0]):
                if x[-i] < percentile:
                    return x.shape[0] - i
            return 0

        # Keep cutoff control deterministic without weakening model-wide checks.
        magnitude = torch.abs(mel).detach().to(device='cpu', dtype=torch.float32)
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        return find_cutoff(energy, percentile)

    def mel_replace_ops(self, samples, input, cutoff_melbins):
        result = torch.zeros_like(samples)
        for i in range(samples.size(0)):

            result[i][..., cutoff_melbins[i]:] = samples[i][..., cutoff_melbins[i]:]
            result[i][..., :cutoff_melbins[i]] = input[i][..., :cutoff_melbins[i]]
        return result, cutoff_melbins
    
    def mel_cutoff_bins(self, input):
        cutoff_melbins = [] 
        for i in range(input.size(0)):
            cutoff_melbin = self.locate_cutoff_freq(torch.exp(input[i]))
            cutoff_melbins.append(cutoff_melbin) 
        return cutoff_melbins


    @torch.inference_mode()
    def sample(
        self,
        *,
        cond = None,
        cond_mask = None,
        time_steps = 4,
        cond_scale = 1.,
        decode_to_audio = True,
        std_1 = None,
        std_2 = None,
        mel_pp = False,
        cfm_method = None,
        input_sampling_rate = None,
        random_sr = None,
        cutoff_hz = None,
        return_intermediates = False,
        validation_generator = None,
        initial_noise = None,
    ):
        if cfm_method not in ['basic_cfm','independent_cfm_adaptive', 'independent_cfm_constant', 'independent_cfm_mix']:
            cfm_method = self.cfm_method
            # raise ValueError("Do not define cfm_method variable for sample()")
            
        if cfm_method in ['independent_cfm_adaptive', 'independent_cfm_constant','independent_cfm_mix']:
            if std_1 is None or std_2 is None:
                std_1 = 1.0
                std_2 = self.sigma
            
        cond_is_raw_audio = is_probably_audio_from_shape(cond)

        if cond_is_raw_audio:
            assert exists(self.flowhigh.audio_enc_dec)
            
            self.flowhigh.audio_enc_dec.eval()
            cond = self.flowhigh.audio_enc_dec.encode(cond)

        self_attn_mask = None
        shape = cond.shape # [B, Time, Channel]
        batch = shape[0]
        needs_cutoff_bins = needs_cutoff_bins_for_sample(
            self.use_highband_residual_flow, mel_pp, cfm_method
        )
        cutoff_bins = self.mel_cutoff_bins(cond) if needs_cutoff_bins else None
        cutoff_audit = {
            "deterministic_cutoff_deadpath_patch": DETERMINISTIC_CUTOFF_DEADPATH_PATCH,
            "cutoff_bin_policy": CUTOFF_BIN_POLICY,
            "needs_cutoff_bins": bool(needs_cutoff_bins),
            "cutoff_bins_computed": cutoff_bins is not None,
        }
        hf_mask = None
        cutoff_ratio = None
        if self.use_highband_residual_flow:
            cutoff_hz_tensor = self._cutoff_hz_tensor(
                batch,
                cond.dtype,
                cond.device,
                input_sampling_rate=input_sampling_rate,
                random_sr=random_sr,
                cutoff_hz=cutoff_hz
            )
            hf_mask = self._build_hf_mask_for(cond, cutoff_hz_tensor)
            cutoff_ratio = cutoff_hz_tensor / (self.flowhigh.audio_enc_dec.sampling_rate / 2)
        
        # neural ode
        self.flowhigh.eval()
        model_forward_count = 0
    
        # ode function
        def ode_fn(t, x, *, packed_shape = None): 
            nonlocal model_forward_count
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')
                
            out = self.flowhigh.forward_with_cond_scale(
                x,
                times = t,
                cond = cond,
                cond_scale = cond_scale,
                cond_mask = cond_mask,
                self_attn_mask = self_attn_mask,
                hf_mask = hf_mask,
                cutoff_ratio = cutoff_ratio
            )
            model_forward_count += 1

            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')
            return out # out.shape : [1, Time, mel_channel]

        if self.use_highband_residual_flow:
            if exists(initial_noise):
                y0 = initial_noise.to(device=cond.device, dtype=cond.dtype) * float(self.residual_noise_scale)
            elif exists(validation_generator):
                y0 = torch.randn(cond.shape, dtype=cond.dtype, device=cond.device, generator=validation_generator) * float(self.residual_noise_scale)
            else:
                y0 = torch.randn_like(cond, device=cond.device) * float(self.residual_noise_scale)

        elif cfm_method == 'basic_cfm': 
            y0 = randn_like_with_optional_generator(cond, validation_generator)

        elif cfm_method == 'independent_cfm_adaptive':
            # y0 from intended prior
            epsilon = randn_like_with_optional_generator(cond, validation_generator)
            y0 = cond*std_1 + epsilon*std_2

        elif cfm_method == 'independent_cfm_constant':
            # y0 from intended prior
            epsilon = randn_like_with_optional_generator(cond, validation_generator)
            y0 = cond*std_1 + epsilon*std_2
    
        elif cfm_method == 'independent_cfm_mix':
            # y0 from intended prior
            epsilon = randn_like_with_optional_generator(cond, validation_generator)
            y0_low = cond*std_1 + epsilon*std_2
            y0_high = epsilon
            y0, _ = self.mel_replace_ops(y0_high, y0_low, cutoff_bins)
            
        t = torch.linspace(0, 1, time_steps + 1, device = self.device)
        if self.use_highband_residual_flow:
            sampled = y0
            for step_idx in range(int(time_steps)):
                v_step = ode_fn(t[step_idx], sampled)
                dt = t[step_idx + 1] - t[step_idx]
                sampled = sampled + dt * v_step
        elif not self.use_torchode:

            LOGGER.debug('sampling with torchdiffeq')
            trajectory = odeint(ode_fn, y0, t, **self.odeint_kwargs) # bottle neck
            sampled = trajectory[-1]
            
            # # trajectory plot
            # n = len(trajectory) 
            # for i in range(n):
            #     plt.figure(figsize=(12, 4)) 
            #     plt.imshow(numpy.rot90(trajectory[i].squeeze().cpu().numpy(), 1), aspect='auto', origin='upper', interpolation='none')  
            #     plt.colorbar() 
            #     plt.title(f'trajectory[{i}]') 
            #     plt.xlabel('X-axis')  
            #     plt.ylabel('Y-axis') 
                
            #     plt.savefig(f'__trajectory[{i}].png', dpi=300, bbox_inches='tight') 
            #     plt.close()  

        else:
            LOGGER.debug('sampling with torchode')
            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')
            fn = partial(ode_fn, packed_shape = packed_shape)
            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term = term)
            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)
            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)
            sol = jit_solver.solve(init_value)
            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, 'b *')
            
        if mel_pp:
            sampled, cutoff_bins = self.mel_replace_ops(sampled, cond, cutoff_bins)

        if self.use_highband_residual_flow:
            r_hat = sampled
            hf_mask_expanded = hf_mask.expand_as(cond)
            # Final mel replacement: low band comes from M_up, high band comes from the sampled residual.
            M_final_raw = (1 - hf_mask_expanded) * cond + hf_mask_expanded * (cond + r_hat)
            sampled = smooth_mel_seam(
                M_final_raw,
                hf_mask,
                kernel_size=int(self.seam_smoothing_kernel_size),
                seam_bins=int(self.seam_smoothing_bins),
            ) if self.seam_smoothing_enabled else M_final_raw
            self.last_highband_debug = {
                "M_up_shape": tuple(cond.shape),
                "R_hat_shape": tuple(r_hat.shape),
                "M_final_shape": tuple(sampled.shape),
                "mask_shape": tuple(hf_mask.shape),
                "mask_min": float(hf_mask.min().detach().cpu()),
                "mask_max": float(hf_mask.max().detach().cpu()),
                "mask_mean": float(hf_mask.mean().detach().cpu()),
                "cutoff_hz": cutoff_hz_tensor.detach().cpu().tolist(),
                "target_sr": self.flowhigh.audio_enc_dec.sampling_rate,
                "residual_noise_scale": float(self.residual_noise_scale),
                "model_forward_count": float(model_forward_count),
                "flow_nfe": float(time_steps),
                "seam_smoothing_enabled": bool(self.seam_smoothing_enabled),
                "final_mel_formula": "M_final_raw = M_up + mask_hf * R_hat"
            }
            self.last_highband_debug.update(getattr(self.flowhigh, "last_melconv_debug", {}))

            if return_intermediates:
                debug_out = {
                    "M_up": cond,
                    "R_hat": r_hat,
                    "M_final_raw": M_final_raw,
                    "M_final": sampled,
                    "mask_hf": hf_mask,
                    "model_forward_count": model_forward_count,
                    "flow_nfe": time_steps,
                    "seam_smoothing_enabled": bool(self.seam_smoothing_enabled),
                    **cutoff_audit,
                }
                debug_out.update(getattr(self.flowhigh, "last_melconv_debug", {}))
                return debug_out
            
        if not decode_to_audio or not exists(self.flowhigh.audio_enc_dec):
            return sampled 
        
        return self.flowhigh.audio_enc_dec.decode(sampled)

    def forward(
        self,
        x1,
        *,
        mask = None,
        cond = None,
        cond_mask = None,
        cond_lengths = None,
        input_sampling_rate = None, 
        cond_freq_masking = False, 
        random_sr = None,
        weighted_loss = None, 
        cfm_method = None, # not necessary
        validation_generator = None,
        fixed_times = None,
        fixed_noise = None,
        crop_mel_segments = True,
        global_step = None,
    ):
        
        if cfm_method not in ['basic_cfm','independent_cfm_adaptive' ,'independent_cfm_constant','independent_cfm_mix']:
            cfm_method = self.cfm_method
        
        batch, seq_len, dtype, sigma_min = *x1.shape[:2], x1.dtype, self.sigma
        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))
        
        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.flowhigh.audio_enc_dec), 'audio_enc_dec must be set on FLowHigh to train directly on raw audio'
            audio_enc_dec_sampling_rate = self.flowhigh.audio_enc_dec.sampling_rate
            input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)            
            
            with torch.no_grad():
                self.flowhigh.audio_enc_dec.eval()
                # Making Ground truth mel-spectrogram
                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = self.flowhigh.audio_enc_dec.encode(x1) # x1.shape : [B, Time, channel]

                # Making mel-spectrogram which are empty in high-freqeuncy information
                if exists(cond) and cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                cond = self.flowhigh.audio_enc_dec.encode(cond) # cond.shape : [B, Time, channel]
     
        if x1.size(1) != cond.size(1):
            max_timelength = max(x1.size(1), cond.size(1))
            x1 = F.pad(x1, (0, 0,max_timelength - x1.size(1), 0 ))  
            cond = F.pad(cond, (0, 0, max_timelength - cond.size(1), 0))

        # main conditional flow logic is below
        if exists(fixed_times):
            times = fixed_times.to(device=self.device, dtype=dtype)
        elif exists(validation_generator):
            times = torch.rand((batch,), dtype=dtype, device=self.device, generator=validation_generator)
        else:
            times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')
        hf_mask = None
        cutoff_ratio = None

        if self.use_highband_residual_flow:
            cutoff_hz = self._cutoff_hz_tensor(
                batch,
                x1.dtype,
                x1.device,
                input_sampling_rate=input_sampling_rate,
                random_sr=random_sr
            )
            hf_mask = self._build_hf_mask_for(x1, cutoff_hz)
            hf_mask_expanded = hf_mask.expand_as(x1)
            cutoff_ratio = cutoff_hz / (self.flowhigh.audio_enc_dec.sampling_rate / 2)

            # Target is only the missing high-band residual; low mel bins stay anchored to M_up.
            residual_full = x1 - cond
            residual_low = (1 - hf_mask_expanded) * residual_full
            r_hf = hf_mask_expanded * (x1 - cond)
            if exists(fixed_noise):
                x0 = fixed_noise.to(device=r_hf.device, dtype=r_hf.dtype) * float(self.residual_noise_scale)
            elif exists(validation_generator):
                x0 = torch.randn(r_hf.shape, dtype=r_hf.dtype, device=r_hf.device, generator=validation_generator) * float(self.residual_noise_scale)
            else:
                x0 = torch.randn_like(r_hf) * float(self.residual_noise_scale)
            w = (1 - t) * x0 + t * r_hf
            flow = r_hf - x0
            cutoff_bins = None

            self.last_highband_debug = {
                "M_hr_shape": tuple(x1.shape),
                "M_up_shape": tuple(cond.shape),
                "mask_hf_shape": tuple(hf_mask.shape),
                "R_hf_shape": tuple(r_hf.shape),
                "z_t_shape": tuple(w.shape),
                "v_target_shape": tuple(flow.shape),
                "mask_min": float(hf_mask.min().detach().cpu()),
                "mask_max": float(hf_mask.max().detach().cpu()),
                "mask_mean": float(hf_mask.mean().detach().cpu()),
                "cutoff_hz": cutoff_hz.detach().cpu().tolist(),
                "cutoff_ratio": float(torch.mean(cutoff_ratio).detach().cpu()),
                "target_sr": self.flowhigh.audio_enc_dec.sampling_rate,
                "residual_full_l1": float(torch.mean(torch.abs(residual_full)).detach().cpu()),
                "residual_low_l1": float(torch.mean(torch.abs(residual_low)).detach().cpu()),
                "residual_high_l1": float(torch.mean(torch.abs(r_hf)).detach().cpu()),
                "residual_hf_l1": float(torch.mean(torch.abs(r_hf)).detach().cpu()),
                "highband_delta": float(torch.mean(torch.abs(r_hf)).detach().cpu()),
                "hard_lowband_delta": 0.0,
                "residual_noise_scale": float(self.residual_noise_scale),
                "masked_loss": "loss_hf_norm = ((mask_hf * (v_pred - v_target)) ** 2).sum() / mask_hf.sum().clamp_min(1e-6)"
            }

        elif cfm_method == 'basic_cfm':
            """
            probability path: N(t x1, 1 - (1 - sigma) t)
            mu_t: t * x1 
            sigma_t: 1 - (1 - sigma_min)t
            sigma_min = 1e-4

            sample x_t: sigma_t * x0 + t * x1 = (1 - (1 - sigma_min) * t) * x0 + t * x1             
            target vector field: u_t = (x1 - (1 - sigma_min) x_t) / (1 - (1 - sigma_min) t) = x1 - (1 - sigma_min) * x0 
            
            if sigma_min = 0, then basic_cfm same with rectified-flow from standard normal distribution N(0,I)
            """   
            # x0 is gaussian noise
            x0 = torch.randn_like(x1)   # [B, Time, channel]
            cutoff_bins = None
            
            sigma_t = (1 - (1 - sigma_min) * t)
            
            # sample xt = noisy speech (\psi_t (x_0|x_1))                        
            w = sigma_t * x0 + t * x1  # [B, Time, channel]
            # w = (1 - (1 - sigma_min) * t) * x0 + t * x1  # [B, Time, channel]
            
            # target vector field u_t
            flow = x1 - (1 - sigma_min) * x0  # [B, Time, channel]
            # flow = (x1 - (1 - sigma_min) * w) / (1- (1 - sigma_min) *t)  # [B, Time, channel]
            
        elif cfm_method == 'independent_cfm_adaptive':
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, 1 - (1 - sigma_min) t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t:  1 - (1 - sigma_min) t

            sample x_t: mean + sigma * eps = t * x1 + (1 - t) *x0 + sigma_t * epsilon  
            target vector field: u_t = { (x1-x0) - (1-sigma_min)(xt-x0) } / { 1 - (1 - sigma_min) t } = (x1-x0) - (1-sigma_min) * epsilon 

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """   
            
            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)
            
            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()
            
            cutoff_bins = None
            
            mu_t = t * x1 + (1 - t) * x0 
            sigma_t = (1 - (1 - sigma_min)*t)
            
            # sample xt 
            w = mu_t + sigma_t * epsilon
            
            # target vector field u_t
            flow = (x1-x0) - (1-sigma_min) * epsilon # { (x1-x0) - (1-sigma_min)*(w-x0) } / {1 - (1 - sigma_min)*t} # [B, Time, channel]            
            

        elif cfm_method == 'independent_cfm_constant':
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, sigma_t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t: sigma_min (small enough)

            sample x_t: mean + sigma*eps(eps~N(0,I)) = t * x1 + (1 - t) *x0 + sigma_t * epsilon
            target vector field: u_t = x1 - x0

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """   
            
            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)
            
            cutoff_bins = None
            
            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()
            
            mu_t = t * x1 + (1 - t) * x0 
            sigma_t = sigma_min
            
            # sample xt 
            w = mu_t + sigma_t * epsilon
            
            # target vector field u_t
            flow = x1 - x0  # [B, Time, channel]
            

        elif cfm_method == 'independent_cfm_mix':
            """
            q(z) = q(x0)q(x1)
            probability path_high: N(    t * x1          , 1 - (1 - sigma) t)
            probability path_low : N(t * x1 + (1 - t) *x0,       sigma_min    )
            
            x0: x^mel_low

            sample x_t: 
            target vector field: u_t = 
            """   
            
            # # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)
            
            # get cutoff mel bins of LR mel
            cutoff_bins = self.mel_cutoff_bins(cond)
            
            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()
            
            # sample xt_high
            mu_t_high = t * x1 
            sigma_t_high = (1 - (1 - sigma_min) * t)                      
            xt_high = mu_t_high + sigma_t_high * epsilon   # [B, Time, channel]
            
            # sample xt_low
            mu_t_low = t * x1 + (1 - t) * x0 
            sigma_t_low = sigma_min
            xt_low = mu_t_low + sigma_t_low * epsilon

            w, _ = self.mel_replace_ops(xt_high, xt_low, cutoff_bins)
        
            # target vector field u_t
            flow = torch.zeros_like(x1)
            flow_high = x1 - (1 - sigma_min) * epsilon
            flow_low = x1 - x0  # [B, Time, channel]＼
            for i, cutoff_bin in enumerate(cutoff_bins):
                flow[i][..., cutoff_bin:] = flow_high[i][..., cutoff_bin:]
                flow[i][..., :cutoff_bin] = flow_low[i][..., :cutoff_bin]
            
        # x1.shape = cond.shape = x0.shape = w.shape = flow.shape = [Batch, Time, mel_bin]
        
        # Training mode!
        self.flowhigh.train(self.training)

        # Cut a small segment of mel-spectrogram
        cond_lengths = cond_lengths.to(torch.int32)
        max_cond_lengths = x1.size(1)
        x_mask = sequence_mask(cond_lengths, max_cond_lengths).unsqueeze(1) 
        out_size = 2* self.flowhigh.audio_enc_dec.sampling_rate // self.flowhigh.audio_enc_dec.hop_length

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if crop_mel_segments and not isinstance(out_size, type(None)):
            max_offset = (cond_lengths - out_size).clamp(0)
            offset_ranges = list( zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            
            import random

            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(cond_lengths)

            w_cut = torch.zeros(w.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=w.dtype, device=w.device)
            flow_cut = torch.zeros(flow.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=flow.dtype, device=flow.device)
            cond_cut = torch.zeros(cond.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=cond.dtype, device=cond.device)
            hf_mask_cut = torch.zeros_like(w_cut) if exists(hf_mask) else None
            hf_mask_expanded_for_cut = hf_mask.expand_as(w) if exists(hf_mask) else None
            
            x_cut_lengths = []

            for i, (w_, flow_, cond_, out_offset_) in enumerate(zip(w, flow, cond, out_offset)):
                
                # w_.shape = flow_.shape = cond_.shape = [Time, channel]
                x_cut_length = out_size + (cond_lengths[i] - out_size).clamp(None, 0)
                x_cut_lengths.append(x_cut_length)
                
                cut_lower, cut_upper = out_offset_, out_offset_ + x_cut_length
                w_cut[i, :x_cut_length,: ] = w_[cut_lower:cut_upper,: ]
                flow_cut[i, :x_cut_length,: ] = flow_[cut_lower:cut_upper,: ]
                cond_cut[i, :x_cut_length,: ] = cond_[cut_lower:cut_upper,: ]
                if exists(hf_mask_cut):
                    hf_mask_cut[i, :x_cut_length,: ] = hf_mask_expanded_for_cut[i, cut_lower:cut_upper,: ]

            x_cut_lengths = torch.stack([
                length.to(device=cond_lengths.device, dtype=torch.long)
                for length in x_cut_lengths
            ])
            x_cut_mask = sequence_mask(x_cut_lengths, out_size).unsqueeze(1).to(
                device=x_mask.device,
                dtype=x_mask.dtype,
            )

            w = w_cut 
            flow = flow_cut
            cond = cond_cut
            if exists(hf_mask_cut):
                hf_mask = hf_mask_cut
            x_mask = x_cut_mask

        if exists(x_mask):
            segment_mask = rearrange(x_mask, "b 1 t -> b t").bool()
            if exists(cond_mask) and tuple(cond_mask.shape) == tuple(segment_mask.shape):
                segment_mask = segment_mask & cond_mask.bool()
            if exists(mask) and tuple(mask.shape) == tuple(segment_mask.shape):
                segment_mask = segment_mask & mask.bool()
        else:
            segment_mask = None

        # forward 
        loss = self.flowhigh(
            x = w,
            cond = cond,
            cond_mask = segment_mask,
            times = times,
            target = flow,
            self_attn_mask = segment_mask,
            cond_drop_prob = self.cond_drop_prob,
            cond_freq_masking = cond_freq_masking,
            random_sr = random_sr,
            weighted_loss = weighted_loss,
            cutoff_bins = cutoff_bins,
            hf_mask = hf_mask,
            cutoff_ratio = cutoff_ratio,
            use_hf_frequency_weight = self.use_hf_frequency_weight,
            hf_weight_min = self.hf_weight_min,
            hf_weight_max = self.hf_weight_max,
            hf_weight_mode = self.hf_weight_mode,
            hf_weight_cap_hz_above_cutoff = self.hf_weight_cap_hz_above_cutoff,
            loss_hf_l1_weight = self.loss_hf_l1_weight,
            edge_continuity_loss_weight = self.edge_continuity_loss_weight,
            edge_continuity_bins = self.edge_continuity_bins,
            cutoff_hz = cutoff_hz if self.use_highband_residual_flow else None,
            use_freq_prediction_loss = self.use_freq_prediction_loss,
            freq_prediction_loss_weight = self.freq_prediction_loss_weight,
            use_freq_gradient_loss = self.use_freq_gradient_loss,
            freq_gradient_loss_weight = self.freq_gradient_loss_weight,
            melconv_bridge_l1_weight = self.melconv_bridge_l1_weight,
            output_mel_adapter_l1_weight = self.output_mel_adapter_l1_weight,
            global_step = global_step,
        )
        if self.use_highband_residual_flow:
            self.last_highband_debug["v_pred_shape"] = tuple(self.flowhigh.last_v_pred_shape)
            self.last_highband_debug.update(getattr(self.flowhigh, "last_highband_loss_debug", {}))
        return loss
