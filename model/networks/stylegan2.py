"""
stylegan2.py
────────────
StyleGAN2 generator and discriminator, for conditional MRI -> CT translation.

WHAT THIS IS FOR. Everything in the exp0-exp4 ladder varies the loss on top of one
fixed pair of networks — a U-Net judged by a 70x70 PatchGAN. So the ladder can only
ever answer questions about the objective. This module supplies a second
architecture, so `exp6_stylegan2_fitted` can be compared against `exp2_paper` with
the loss held constant and the architecture as the only variable.

THE CORE IDEA, AND HOW IT DIFFERS FROM THE U-NET. The U-Net decides each output
pixel by carrying spatial detail down through an encoder and back up through skip
connections. StyleGAN2 splits that job in two: a style vector `w` says WHAT KIND of
thing is being drawn, globally and per-layer, while per-pixel noise supplies the
stochastic fine grain. The synthesis network itself starts from a learned 4x4
constant and knows nothing about the input except through `w`.

That has a consequence worth stating plainly, because it is the main risk this
architecture carries here: there are NO encoder-to-decoder skip connections, so the
only route from the MRI to the output is a single global vector. The `to_image`
sums between resolutions are StyleGAN2's own "skip" generator — image residuals,
not feature skips — and they do not restore spatial correspondence. Anatomical
drift is the expected failure mode, and it will show up in the sample panels before
it shows up in any metric.

WHY MODULATED CONVOLUTION. StyleGAN1 used AdaIN: normalise the activations, then
re-scale them by the style. That destroys information carried in the relative
magnitudes between feature maps, and the generator learned to smuggle it past the
normaliser as a large localised spike — the notorious "water droplet" artifact.
StyleGAN2 never touches the activations: the style scales the convolution WEIGHTS
(modulation), and the weights are then divided by their own L2 norm (demodulation).
Same statistical control, no spike to hide behind.

WHY EQUALIZED LEARNING RATE. Weights are held as N(0,1) and scaled at runtime by
1/sqrt(fan_in). Adam normalises each parameter's update by its own gradient
standard deviation, so parameters with different dynamic ranges would otherwise
take different effective step sizes; equalising at runtime puts every weight on the
same scale.

    *** DO NOT PASS THESE MODULES TO networks/init.py:init_weights. ***

    That function applies N(0, 0.02) to every Conv and Linear, which is right for
    pix2pix and fatal here: equalized LR REQUIRES an N(0,1) init, so applying both
    leaves every weight ~50x too small and then runtime-scales it. The network
    trains without complaint and learns nothing. Everything below initialises
    itself, deliberately.

DEVIATIONS FROM THE PAPER, all forced by the task rather than chosen:

  * Conditioning. `w = MappingNetwork(StyleEncoder(MRI))` rather than
    `MappingNetwork(z)` — this is paired translation, not sampling. The encoder is
    also what supplies PatchNCE its taps, so it implements the same tap protocol
    UnetGenerator does.
  * The discriminator is conditional, seeing cat[MRI, CT]. StyleGAN2's is
    unconditional; left that way here, nothing whatsoever would tie the output to
    the input and the model would be a CT generator that merely accepts an
    MRI-derived style vector.
  * Up/downsampling uses nearest/average resampling paired with the [1,3,3,1] FIR
    blur, rather than the paper's fused upfirdn2d kernel. Same filter, same
    intent, far less machinery.
  * A fixed-size synthesis network cannot emit the 512x512 slices validation
    produces; see tiling.py for how that is handled.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiling import tiled_forward

logger = logging.getLogger(__name__)

# LeakyReLU(0.2) shrinks activation variance; the paper's fused activation
# multiplies by sqrt(2) to put it back. Skipping this silently changes the scale
# every equalized-LR calculation above assumes.
ACT_GAIN = math.sqrt(2.0)


def leaky(x):
    return F.leaky_relu(x, 0.2) * ACT_GAIN


def channels_at(resolution, channel_base, channel_max):
    """
    StyleGAN2's channel schedule: wide at low resolution, narrow at high.

    channel_base=32768, channel_max=512 is the paper's config-f. That was fitted to
    FFHQ's 70k images; at 1687 training slices it is heavily over-parameterised,
    which is why exp6 halves both.
    """
    return int(min(channel_base // resolution, channel_max))


# ── Primitives ────────────────────────────────────────────────────────────────

class EqualizedLinear(nn.Module):
    """
    Linear layer with runtime weight scaling.

    `lr_mul` below 1 slows a layer down relative to the rest of the network. The
    mapping network uses 0.01, because a mapping net trained at the same rate as
    the synthesis net destabilises W early in training.
    """

    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0, bias_init=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = (nn.Parameter(torch.full((out_features,), float(bias_init)))
                     if bias else None)
        self.scale = lr_mul / math.sqrt(in_features)
        self.lr_mul = lr_mul

    def forward(self, x):
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias)


class EqualizedConv2d(nn.Module):
    """Conv2d with runtime weight scaling. No normalisation layer follows it."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        stride=self.stride, padding=self.padding)


class Blur(nn.Module):
    """
    Separable [1,3,3,1] FIR low-pass, applied around every resampling step.

    Nearest-neighbour upsampling and strided convolution both alias badly. The
    filter is what stops that aliasing from being baked into the texture, and its
    absence is visible as a faint checkerboard.
    """

    def __init__(self):
        super().__init__()
        k = torch.tensor([1.0, 3.0, 3.0, 1.0])
        k = k[:, None] * k[None, :]
        self.register_buffer("kernel", (k / k.sum())[None, None])

    def forward(self, x):
        channels = x.shape[1]
        kernel = self.kernel.to(x.dtype).expand(channels, 1, 4, 4)
        # Asymmetric padding keeps the 4-tap even-length kernel size-preserving.
        return F.conv2d(F.pad(x, (1, 2, 1, 2)), kernel, groups=channels)


class PixelNorm(nn.Module):
    """Normalise each position's feature vector to unit RMS. Applied to the
    mapping network's input, where it keeps W from drifting in overall scale."""

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class NoiseInjection(nn.Module):
    """
    Per-pixel Gaussian noise with one learned scalar.

    WHAT IT BUYS. Without it the generator must synthesise stochastic detail —
    grain, texture — deterministically from `w`, which burns capacity and makes
    texture visibly repeat.

    TWO GATES, BOTH SPECIFIC TO MEDICAL SYNTHESIS.

      at_eval           Noise is resampled every call, so leaving it on at
                        evaluation would make mae_norm a random variable — two runs
                        of evaluate.py on one checkpoint would disagree, and a
                        clinician could not be shown a stable image. This is the
                        same call pix2pix_nce.py already makes for dropout.

                        The gate is read off nn.Module.training rather than from an
                        externally-set flag ON PURPOSE. A flag flipped by
                        set_eval_mode would stay flipped when the trainer went back
                        to training, silently disabling noise for the rest of the
                        run after the first validation pass.

      max_resolution    Noise at 128 and 256 px synthesises fine texture that has
                        no counterpart anywhere in the source MRI. That is
                        hallucinated detail a reader would interpret as structure.
                        Capping it at 64 keeps the mechanism for coarse stochastic
                        variation and denies it the scale at which it invents
                        findings. 0 means no cap, which is the paper's behaviour.
    """

    def __init__(self, resolution, enabled=True, at_eval=False, max_resolution=0):
        super().__init__()
        self.strength = nn.Parameter(torch.zeros(()))
        self.resolution = int(resolution)
        self.enabled = bool(enabled)
        self.at_eval = bool(at_eval)
        self.gated_off = bool(max_resolution) and resolution > int(max_resolution)

    def forward(self, x):
        if self.gated_off or not self.enabled:
            return x
        if not self.training and not self.at_eval:
            return x
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3],
                            device=x.device, dtype=x.dtype)
        return x + self.strength * noise


class MinibatchStdDev(nn.Module):
    """
    Append the batch's average per-pixel standard deviation as a feature map.

    Targets mode collapse directly: if the generator is producing near-identical
    outputs, that number is near zero and the discriminator can read it off in one
    layer rather than having to infer it.

    NOTE this couples samples within a batch — the same objection base.yaml raises
    against BatchNorm. It is confined to the discriminator and to training (D is
    never run at validation), which is why it is accepted here.
    """

    def __init__(self, group_size=4, num_channels=1):
        super().__init__()
        self.group_size = int(group_size)
        self.num_channels = int(num_channels)

    def forward(self, x):
        n, c, h, w = x.shape
        group = min(self.group_size, n)
        while group > 1 and n % group:
            group -= 1

        f = self.num_channels
        y = x.reshape(group, -1, f, c // f, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.square().mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4]).reshape(-1, f, 1, 1)
        return torch.cat([x, y.repeat(group, 1, h, w)], dim=1)


class ModulatedConv2d(nn.Module):
    """
    The heart of StyleGAN2. See the module docstring for why it exists.

    The batch is folded into the channel axis and run as a grouped convolution,
    because after modulation every sample has its OWN weight tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, w_dim,
                 demodulate=True, upsample=False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.upsample = upsample
        self.blur = Blur() if upsample else None
        # bias_init=1.0 so an untrained style is the identity modulation rather
        # than multiplying every feature map by roughly zero.
        self.affine = EqualizedLinear(w_dim, in_channels, bias_init=1.0)

    def forward(self, x, w):
        batch, channels, height, width = x.shape

        style = self.affine(w).view(batch, 1, channels, 1, 1)
        weight = self.weight * self.scale * style

        if self.demodulate:
            # In fp32 on purpose: this sums squares over an entire kernel — 512
            # input channels by 3x3 — and overflows fp16 under AMP, which shows up
            # as a loss that goes to nan a few hundred steps in.
            inv = torch.rsqrt(weight.float().pow(2).sum(dim=[2, 3, 4]) + 1e-8)
            weight = weight * inv.to(weight.dtype).view(batch, -1, 1, 1, 1)

        if self.upsample:
            x = self.blur(F.interpolate(x, scale_factor=2, mode="nearest"))
            height, width = height * 2, width * 2

        x = x.reshape(1, batch * channels, height, width)
        weight = weight.reshape(batch * self.out_channels, channels,
                                self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.reshape(batch, self.out_channels, height, width)
        return out + self.bias.view(1, -1, 1, 1)


class ResDownBlock(nn.Module):
    """
    Residual downsampling block, shared by the discriminator and the style encoder.

    The 1/sqrt(2) on the sum keeps activation variance stable through a deep
    residual stack — without it the signal grows by sqrt(2) per block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.skip = EqualizedConv2d(in_channels, out_channels, 1, bias=False)
        self.blur = Blur()

    def forward(self, x):
        residual = self.skip(F.avg_pool2d(self.blur(x), 2))
        h = leaky(self.conv0(x))
        h = leaky(self.conv1(self.blur(h)))
        return (h + residual) / math.sqrt(2.0)


# ── Generator ─────────────────────────────────────────────────────────────────

class MappingNetwork(nn.Module):
    """Eight equalized-LR layers turning the encoded MRI into the style vector w."""

    def __init__(self, w_dim, n_layers=8, lr_mul=0.01):
        super().__init__()
        self.norm = PixelNorm()
        self.layers = nn.ModuleList(
            [EqualizedLinear(w_dim, w_dim, lr_mul=lr_mul) for _ in range(n_layers)])

    def forward(self, x):
        h = self.norm(x)
        for layer in self.layers:
            h = leaky(layer(h))
        return h


class StyleEncoder(nn.Module):
    """
    Encodes the source MRI into the style vector, and supplies PatchNCE its taps.

    This carries the whole conditioning burden. StyleGAN2 samples `z` from a
    Gaussian prior; there is no prior here, so `w` has to come from the input
    image, and every spatial fact the generator will ever know about the MRI has to
    survive the trip through this bottleneck.

    The tap protocol is deliberately identical to UnetGenerator's (unet.py:120-141)
    so Pix2PixNCEModel.compute_nce needs no branch: tap 0 is the input image, tap k
    is the output of block k-1, at input_size / 2**k.
    """

    def __init__(self, in_channels, native_size, const_size, w_dim,
                 channel_base, channel_max):
        super().__init__()
        self.in_channels = int(in_channels)
        self.native_size = int(native_size)
        self.const_size = int(const_size)
        self.num_downs = int(math.log2(native_size // const_size))

        def ch(res):
            return channels_at(res, channel_base, channel_max)

        self.from_image = EqualizedConv2d(in_channels, ch(native_size), 1)

        blocks, tap_channels = [], []
        resolution = native_size
        for _ in range(self.num_downs):
            blocks.append(ResDownBlock(ch(resolution), ch(resolution // 2)))
            resolution //= 2
            tap_channels.append(ch(resolution))
        self.blocks = nn.ModuleList(blocks)
        self.block_channels = tap_channels

        self.to_w = EqualizedLinear(ch(const_size) * const_size * const_size, w_dim)

    # ── Tap bookkeeping (mirrors UnetGenerator) ──────────────────────────────

    @property
    def n_taps(self):
        return self.num_downs + 1

    def _check_tap(self, tap):
        if not 0 <= tap < self.n_taps:
            raise ValueError(
                f"Tap {tap} is out of range for a StyleEncoder with "
                f"{self.num_downs} downsamplings. Valid taps are 0..{self.num_downs} "
                f"(0 = the input image, k = the output of block k-1)."
            )

    def tap_channels(self, tap):
        self._check_tap(tap)
        return self.in_channels if tap == 0 else self.block_channels[tap - 1]

    def tap_spatial(self, tap, input_size):
        self._check_tap(tap)
        return input_size // (2 ** tap)

    def forward(self, x, tap_layers=None, encode_only=False):
        """Returns w, or the tapped features when encode_only is set."""
        taps = sorted(set(tap_layers)) if tap_layers else []
        for tap in taps:
            self._check_tap(tap)

        feats = [x] if 0 in taps else []
        deepest = max(taps) if taps else self.num_downs

        h = leaky(self.from_image(x))
        for i, block in enumerate(self.blocks):
            h = block(h)
            if (i + 1) in taps:
                feats.append(h)
            if encode_only and (i + 1) >= deepest:
                return feats

        if encode_only:
            return feats

        w = self.to_w(h.flatten(1))
        return (w, feats) if taps else w


class SynthesisBlock(nn.Module):
    """One resolution of the synthesis network, plus its contribution to the image."""

    def __init__(self, in_channels, out_channels, w_dim, resolution, image_channels,
                 is_first, noise_enabled=True, noise_at_eval=False,
                 noise_max_resolution=0):
        super().__init__()
        self.is_first = bool(is_first)
        self.resolution = int(resolution)

        def make_noise():
            return NoiseInjection(resolution, noise_enabled, noise_at_eval,
                                  noise_max_resolution)

        if not self.is_first:
            self.conv0 = ModulatedConv2d(in_channels, out_channels, 3, w_dim,
                                         upsample=True)
            self.noise0 = make_noise()

        conv1_in = in_channels if self.is_first else out_channels
        self.conv1 = ModulatedConv2d(conv1_in, out_channels, 3, w_dim)
        self.noise1 = make_noise()

        # to_image is NOT demodulated: it writes intensities, not features, and
        # renormalising them would fight the tanh at the end.
        self.to_image = ModulatedConv2d(out_channels, image_channels, 1, w_dim,
                                        demodulate=False)
        self.num_ws = 2 if self.is_first else 3

    def forward(self, x, ws, image):
        i = 0
        if not self.is_first:
            x = leaky(self.noise0(self.conv0(x, ws[:, i])))
            i += 1
        x = leaky(self.noise1(self.conv1(x, ws[:, i])))
        i += 1

        contribution = self.to_image(x, ws[:, i])
        if image is None:
            image = contribution
        else:
            image = F.interpolate(image, scale_factor=2, mode="bilinear",
                                  align_corners=False) + contribution
        return x, image


class StyleGAN2Generator(nn.Module):
    """
    Parameters
    ----------
    native_size    : the one output size the synthesis network can emit. Inputs
                     larger than this are handled by sliding-window inference; see
                     tiling.py for why that is necessary rather than a convenience.
    style_mixing_prob
                   : probability of splicing a second w in partway up. In
                     unconditional StyleGAN that second w is a free random latent.
                     Here every w encodes a real patient, so the only one available
                     comes from ANOTHER PATIENT IN THE BATCH. Harmless when the
                     objective is purely adversarial (exp5); actively wrong when
                     lambda_l1 = 100 demands per-pixel agreement with this
                     patient's CT (exp6, which sets it to 0).
    truncation_psi : pulls w toward the dataset mean at evaluation. Meaningful when
                     sampling; meaningless here, where w encodes WHICH PATIENT this
                     is — truncation would pull the output toward the average
                     patient. Left at 1.0 (off) in both configs.
    """

    def __init__(self, in_channels=1, out_channels=1, w_dim=512, n_mapping=8,
                 channel_base=32768, channel_max=512, const_size=4, native_size=256,
                 tile_stride=128, noise=True, noise_at_eval=False,
                 noise_max_resolution=0, style_mixing_prob=0.0, truncation_psi=1.0):
        super().__init__()
        if native_size % const_size or not float(math.log2(native_size // const_size)).is_integer():
            raise ValueError(
                f"native_size ({native_size}) must be const_size ({const_size}) "
                f"times a power of two.")

        self.in_channels = int(in_channels)
        self.native_size = int(native_size)
        self.tile_stride = int(tile_stride)
        self.const_size = int(const_size)
        self.style_mixing_prob = float(style_mixing_prob)
        self.truncation_psi = float(truncation_psi)
        self.noise_at_eval = bool(noise_at_eval)

        def ch(res):
            return channels_at(res, channel_base, channel_max)

        self.encoder = StyleEncoder(in_channels, native_size, const_size, w_dim,
                                    channel_base, channel_max)
        self.mapping = MappingNetwork(w_dim, n_mapping)

        self.const = nn.Parameter(torch.randn(1, ch(const_size), const_size, const_size))

        blocks = []
        resolution = const_size
        prev = ch(const_size)
        while resolution <= native_size:
            blocks.append(SynthesisBlock(
                prev, ch(resolution), w_dim, resolution, out_channels,
                is_first=(resolution == const_size),
                noise_enabled=bool(noise),
                noise_at_eval=bool(noise_at_eval),
                noise_max_resolution=noise_max_resolution))
            prev = ch(resolution)
            resolution *= 2
        self.blocks = nn.ModuleList(blocks)
        self.num_ws = sum(block.num_ws for block in self.blocks)

        self.register_buffer("w_avg", torch.zeros(w_dim))

        logger.info("StyleGAN2Generator: native=%d const=%d w_dim=%d blocks=%d "
                    "channels(%d..%d) mixing=%.2f params=%.1fM",
                    self.native_size, self.const_size, w_dim, len(self.blocks),
                    ch(const_size), ch(native_size), self.style_mixing_prob,
                    sum(p.numel() for p in self.parameters()) / 1e6)

    # ── Tap protocol, delegated to the encoder ───────────────────────────────

    @property
    def n_taps(self):
        return self.encoder.n_taps

    def tap_channels(self, tap):
        return self.encoder.tap_channels(tap)

    def tap_spatial(self, tap, input_size):
        return self.encoder.tap_spatial(tap, input_size)

    # ── Style plumbing ───────────────────────────────────────────────────────

    def _styles(self, w):
        """Broadcast w to one style per layer, applying mixing and truncation."""
        batch = w.shape[0]

        if self.training:
            with torch.no_grad():
                self.w_avg.copy_(self.w_avg.lerp(w.detach().float().mean(0), 0.005))
        elif self.truncation_psi != 1.0:
            w = self.w_avg.to(w.dtype).lerp(w, self.truncation_psi)

        ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)

        if self.training and self.style_mixing_prob > 0.0 and batch > 1:
            if float(torch.rand(())) < self.style_mixing_prob:
                cutoff = int(torch.randint(1, self.num_ws, ()))
                other = w[torch.randperm(batch, device=w.device)]
                other = other.unsqueeze(1).repeat(1, self.num_ws, 1)
                # torch.where rather than an in-place slice assignment: `ws` is a
                # non-leaf tensor inside the autograd graph, and writing into it
                # in place is the kind of thing that works until the day a version
                # bump turns it into a "modified by an inplace operation" error.
                keep = (torch.arange(self.num_ws, device=w.device) < cutoff)
                ws = torch.where(keep.view(1, -1, 1), ws, other)

        return ws

    def synthesise_with_styles(self, x):
        """
        Generate one native_size tile, returning (image, ws).

        Path-length regularization needs the per-layer style tensor as a node in
        the graph so it can differentiate the image with respect to it, which is
        why this is separate from the plain forward. `ws` is deliberately NOT
        detached — the penalty has to reach the mapping network and the encoder.
        """
        if x.shape[-1] != self.native_size or x.shape[-2] != self.native_size:
            raise ValueError(
                f"StyleGAN2Generator synthesises {self.native_size}x{self.native_size} "
                f"only, got {tuple(x.shape[-2:])}. Larger inputs are handled by "
                f"tiling; a smaller one means data.crop_size and "
                f"model.generator.native_size disagree."
            )

        ws = self._styles(self.mapping(self.encoder(x)))

        h = self.const.to(x.dtype).expand(x.shape[0], -1, -1, -1)
        image, consumed = None, 0
        for block in self.blocks:
            h, image = block(h, ws[:, consumed:consumed + block.num_ws], image)
            consumed += block.num_ws

        # tanh, matching the [-1,1] the dataset converts its [0,1] arrays into.
        return torch.tanh(image), ws

    def _synthesise(self, x):
        """Generate exactly one native_size tile."""
        return self.synthesise_with_styles(x)[0]

    def forward(self, x, tap_layers=None, encode_only=False):
        """
        Contract identical to UnetGenerator.forward:
          out           when tap_layers is None
          (out, feats)  when tap_layers is given
          feats         when encode_only is True
        """
        taps = sorted(set(tap_layers)) if tap_layers else []

        if encode_only:
            return self.encoder(x, tap_layers=taps, encode_only=True)

        if x.shape[-1] > self.native_size or x.shape[-2] > self.native_size:
            out = tiled_forward(self._synthesise, x, self.native_size, self.tile_stride)
        else:
            out = self._synthesise(x)

        if not taps:
            return out
        return out, self.encoder(x, tap_layers=taps, encode_only=True)


# ── Discriminator ─────────────────────────────────────────────────────────────

class StyleGAN2Discriminator(nn.Module):
    """
    Residual discriminator with minibatch standard deviation.

    NO NORMALISATION LAYERS anywhere, and no spectral norm: equalized learning rate
    plus the R1 gradient penalty is StyleGAN2's entire stability story, and layering
    this project's usual spectral norm on top would constrain D twice.

    Unlike the PatchGAN this replaces, the output is ONE SCALAR PER IMAGE rather
    than a grid of per-patch scores. That is a real change in what the adversarial
    term polices — global plausibility instead of local texture — and it is part of
    what exp5/exp6 are testing.
    """

    def __init__(self, in_channels=2, resolution=256, channel_base=32768,
                 channel_max=512, const_size=4, mbstd_group_size=4,
                 mbstd_num_channels=1):
        super().__init__()

        def ch(res):
            return channels_at(res, channel_base, channel_max)

        self.resolution = int(resolution)
        self.from_image = EqualizedConv2d(in_channels, ch(resolution), 1)

        blocks = []
        current = resolution
        while current > const_size:
            blocks.append(ResDownBlock(ch(current), ch(current // 2)))
            current //= 2
        self.blocks = nn.ModuleList(blocks)

        self.mbstd = MinibatchStdDev(mbstd_group_size, mbstd_num_channels)
        self.final_conv = EqualizedConv2d(ch(const_size) + mbstd_num_channels,
                                          ch(const_size), 3, padding=1)
        self.final_linear = EqualizedLinear(
            ch(const_size) * const_size * const_size, ch(const_size))
        # No sigmoid: every objective in losses/gan_loss.py consumes raw scores.
        self.out = EqualizedLinear(ch(const_size), 1)

        logger.info("StyleGAN2Discriminator: in=%d resolution=%d mbstd_group=%d "
                    "params=%.1fM", in_channels, self.resolution, mbstd_group_size,
                    sum(p.numel() for p in self.parameters()) / 1e6)

    def forward(self, x):
        h = leaky(self.from_image(x))
        for block in self.blocks:
            h = block(h)
        h = self.mbstd(h)
        h = leaky(self.final_conv(h))
        h = leaky(self.final_linear(h.flatten(1)))
        return self.out(h)


# ── Builders ──────────────────────────────────────────────────────────────────

def _check_num_downs(cfg_generator, native_size, const_size):
    """
    Keep model.generator.num_downs honest.

    The architecture derives its depth from native_size and const_size, but
    data/dataset.py:317 reads `num_downs` to decide the validation padding
    multiple. If the two disagree, validation pads to a size the generator's
    tiling was not planned around — so make the mismatch a startup error.
    """
    derived = int(math.log2(native_size // const_size))
    declared = cfg_generator.get("num_downs", derived)
    if int(declared) != derived:
        raise ValueError(
            f"model.generator.num_downs is {declared}, but native_size "
            f"{native_size} over const_size {const_size} is {derived} "
            f"downsamplings. dataset.py uses num_downs to set the validation "
            f"padding multiple, so these must agree — set num_downs: {derived}."
        )
    return derived


def build_stylegan2_generator(cfg_generator):
    """Construct the generator described by a cfg.model.generator with type stylegan2."""
    native_size = int(cfg_generator.get("native_size", 256))
    const_size = int(cfg_generator.get("const_size", 4))
    _check_num_downs(cfg_generator, native_size, const_size)

    return StyleGAN2Generator(
        in_channels=cfg_generator.get("in_channels", 1),
        out_channels=cfg_generator.get("out_channels", 1),
        w_dim=cfg_generator.get("w_dim", 512),
        n_mapping=cfg_generator.get("n_mapping", 8),
        channel_base=cfg_generator.get("channel_base", 32768),
        channel_max=cfg_generator.get("channel_max", 512),
        const_size=const_size,
        native_size=native_size,
        tile_stride=cfg_generator.get("tile_stride", 128),
        noise=cfg_generator.get("noise", True),
        noise_at_eval=cfg_generator.get("noise_at_eval", False),
        noise_max_resolution=cfg_generator.get("noise_max_resolution", 0),
        style_mixing_prob=cfg_generator.get("style_mixing_prob", 0.0),
        truncation_psi=cfg_generator.get("truncation_psi", 1.0),
    )


def build_stylegan2_discriminator(cfg_model, spectral, batch_size=None):
    """Construct the discriminator described by cfg.model.discriminator."""
    disc = cfg_model.discriminator
    gen = cfg_model.generator

    if spectral:
        logger.warning(
            "stabilizers.spectral_norm_d is on, but the StyleGAN2 discriminator "
            "does not use it — equalized learning rate plus R1 is its stability "
            "mechanism, and spectral norm on top would constrain D twice. "
            "Ignoring it; set spectral_norm_d: false to silence this.")

    in_channels = gen.get("out_channels", 1)
    if disc.get("conditional", True):
        in_channels += gen.get("in_channels", 1)

    group_size = int(disc.get("mbstd_group_size", 4))
    if batch_size is not None and batch_size % group_size:
        raise ValueError(
            f"model.discriminator.mbstd_group_size ({group_size}) must divide "
            f"train.batch_size ({batch_size}); minibatch-stddev groups the batch "
            f"and a remainder would silently change the group size mid-run."
        )

    return StyleGAN2Discriminator(
        in_channels=in_channels,
        resolution=int(gen.get("native_size", 256)),
        channel_base=disc.get("channel_base", 32768),
        channel_max=disc.get("channel_max", 512),
        const_size=int(gen.get("const_size", 4)),
        mbstd_group_size=group_size,
        mbstd_num_channels=int(disc.get("mbstd_num_channels", 1)),
    )
