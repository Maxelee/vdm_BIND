"""Helper functions for constructing BIND models.

Separated from `model_manager` so that initialization logic can be reused
without pulling in the entire manager class.
"""

import os

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import seed_everything

from vdm.astro_dataset import get_astro_data
from vdm import networks_clean as networks
from vdm import vdm_model_clean as vdm_module
from vdm import vdm_model_triple as vdm_triple_module


def _load_dataset_if_requested(config, skip_data_loading, verbose=False):
	"""Return the hydro dataloader unless loading is skipped."""

	if skip_data_loading:
		return None

	if not getattr(config, 'train_samples', None):
		data_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test/'
	else:
		data_root = '/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/train/'

	hydro = get_astro_data(
		config.dataset,
		data_root,
		num_workers=config.num_workers,
		batch_size=config.batch_size,
		stage='test',
		quantile_path=getattr(config, 'quantile_path', None),
	)

	if verbose:
		print(f"[ModelManager] Dataset loaded.")

	return hydro


def initialize_triple(config, verbose=False, skip_data_loading=False):
	"""Initialize a triple VDM model (3 separate 1-channel models)."""

	if verbose:
		print("[ModelManager] Initializing TRIPLE model (3 separate 1-channel models)...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"Triple model requires a valid checkpoint. Got: {config.best_ckpt}")

	checkpoint = torch.load(config.best_ckpt, map_location='cuda', weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in hparams.items():
			print(f"  {k}: {v}")

	state_dict = checkpoint.get('state_dict', {})

	conv_in_keys = [k for k in state_dict.keys() if 'conv_in.weight' in k]
	if conv_in_keys:
		first_conv_in = state_dict[conv_in_keys[0]]
		total_in_channels = first_conv_in.shape[1]
		input_channels = 1

		legacy_keys = [k for k in state_dict.keys() if 'fourier_features.freqs' in k]
		new_multiscale_keys = [
			k
			for k in state_dict.keys()
			if 'fourier_features_halo.frequencies' in k or 'fourier_features_largescale.frequencies' in k
		]

		if legacy_keys:
			config.fourier_legacy = True
			config.use_fourier_features = True
		elif new_multiscale_keys:
			config.fourier_legacy = False
			config.use_fourier_features = True
		else:
			config.fourier_legacy = False
			config.use_fourier_features = False

		if verbose:
			print(f"[ModelManager] Triple model Fourier detection:")
			print(f"  Legacy keys found: {len(legacy_keys)}")
			print(f"  New multi-scale keys found: {len(new_multiscale_keys)}")
			print(f"  fourier_legacy: {config.fourier_legacy}")
			print(f"  use_fourier_features: {config.use_fourier_features}")

		if config.use_fourier_features:
			if config.fourier_legacy:
				fourier_channels = 4
				base_channels = input_channels + 1 + fourier_channels
				config.large_scale_channels = total_in_channels - base_channels
				config.conditioning_channels = 1
			else:
				config.conditioning_channels = 1
				config.large_scale_channels = (total_in_channels - 10) // 9
		else:
			config.conditioning_channels = 1
			config.large_scale_channels = total_in_channels - input_channels - 1

		if verbose:
			print(f"[ModelManager] Triple model channel configuration:")
			print(f"  Total conv_in channels: {total_in_channels}")
			print(f"  conditioning_channels: {config.conditioning_channels}")
			print(f"  large_scale_channels: {config.large_scale_channels}")

	unet_params = {
		'input_channels': 1,
		'conditioning_channels': config.conditioning_channels,
		'large_scale_channels': config.large_scale_channels,
		'gamma_min': hparams.get('gamma_min', config.gamma_min),
		'gamma_max': hparams.get('gamma_max', config.gamma_max),
		'embedding_dim': getattr(config, 'embedding_dim', 256),
		'norm_groups': getattr(config, 'norm_groups', 32),
		'n_blocks': getattr(config, 'n_blocks', 4),
		'add_attention': getattr(config, 'add_attention', True),
		'n_attention_heads': getattr(config, 'n_attention_heads', 8),
		'use_fourier_features': config.use_fourier_features,
		'legacy_fourier': config.fourier_legacy,
		'use_param_conditioning': getattr(config, 'use_param_conditioning', False),
		'param_min': getattr(config, 'min', None),
		'param_max': getattr(config, 'max', None),
		'use_cross_attention': getattr(config, 'use_cross_attention', False),
		'cross_attention_location': getattr(config, 'cross_attention_location', 'bottleneck'),
		'cross_attention_heads': getattr(config, 'cross_attention_heads', 8),
		'cross_attention_dropout': getattr(config, 'cross_attention_dropout', 0.1),
		'use_chunked_cross_attention': getattr(config, 'use_chunked_cross_attention', True),
		'cross_attention_chunk_size': getattr(config, 'cross_attention_chunk_size', 512),
		'downsample_cross_attn_cond': getattr(config, 'downsample_cross_attn_cond', False),
		'cross_attn_cond_downsample_factor': getattr(config, 'cross_attn_cond_downsample_factor', 2),
	}

	if verbose:
		print(f"[ModelManager] Creating three UNet score models with params:")
		for k, v in unet_params.items():
			print(f"  {k}: {v}")

	hydro_dm_score_model = networks.UNetVDM(**unet_params)
	gas_score_model = networks.UNetVDM(**unet_params)
	stars_score_model = networks.UNetVDM(**unet_params)

	if verbose:
		print(f"[ModelManager] Created three UNet models. Creating LightTripleVDM...")

	triple_params = {
		'hydro_dm_score_model': hydro_dm_score_model,
		'gas_score_model': gas_score_model,
		'stars_score_model': stars_score_model,
		'learning_rate': hparams.get('learning_rate', getattr(config, 'learning_rate', 3e-4)),
		'lr_scheduler': hparams.get('lr_scheduler', getattr(config, 'lr_scheduler', 'cosine')),
		'noise_schedule': hparams.get('noise_schedule', getattr(config, 'noise_schedule', 'fixed_linear')),
		'gamma_min': hparams.get('gamma_min', config.gamma_min),
		'gamma_max': hparams.get('gamma_max', config.gamma_max),
		'image_shape': hparams.get('image_shape', (1, config.cropsize, config.cropsize)),
		'data_noise': hparams.get('data_noise', getattr(config, 'data_noise', 1e-3)),
		'antithetic_time_sampling': hparams.get('antithetic_time_sampling', getattr(config, 'antithetic_time_sampling', True)),
		'channel_weights': hparams.get('channel_weights', (1.0, 1.0, 1.0)),
		'use_focal_loss_hydro_dm': hparams.get('use_focal_loss_hydro_dm', False),
		'use_focal_loss_gas': hparams.get('use_focal_loss_gas', False),
		'use_focal_loss_stars': hparams.get('use_focal_loss_stars', getattr(config, 'use_focal_loss', False)),
		'focal_gamma': hparams.get('focal_gamma', getattr(config, 'focal_gamma', 2.0)),
		'use_param_prediction': hparams.get('use_param_prediction', getattr(config, 'use_param_prediction', False)),
		'param_prediction_weight': hparams.get('param_prediction_weight', getattr(config, 'param_prediction_weight', 0.01)),
	}

	vdm_model = vdm_triple_module.LightTripleVDM(**triple_params)

	if verbose:
		print(f"[ModelManager] Loading state dict into LightTripleVDM...")

	model_state = vdm_model.state_dict()
	missing_keys = [k for k in model_state.keys() if k not in state_dict]
	if missing_keys and verbose:
		print(f"[ModelManager] Warning: checkpoint is missing {len(missing_keys)} keys. Injecting defaults.")
	for k in missing_keys:
		state_dict[k] = model_state[k]

	vdm_model.load_state_dict(state_dict)
	vdm_model = vdm_model.eval()

	if verbose:
		print("[ModelManager] Triple model loaded successfully.")
		print(f"[ModelManager] Model checkpoint: {config.best_ckpt}")
		print(f"[ModelManager] Using {config.conditioning_channels} conditioning channel(s)")
		print(f"[ModelManager] Using {config.large_scale_channels} large-scale channel(s)")
		print(
			f"[ModelManager] Fourier features: {'LEGACY' if config.fourier_legacy else 'MULTI-SCALE' if config.use_fourier_features else 'DISABLED'}"
		)

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, vdm_model


def initialize_ddpm(config, verbose=False, skip_data_loading=False):
	"""Initialize a DDPM/score_models model (NCSNpp or DDPM architecture)."""

	if verbose:
		print("[ModelManager] Initializing DDPM/Score Model (direct score_models approach)...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	try:
		from score_models import ScoreModel, NCSNpp, DDPM  # noqa: F401
	except ImportError as exc:
		raise ImportError(f"DDPM model requires score_models package: {exc}")

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"DDPM model requires a valid checkpoint. Got: {config.best_ckpt}")

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})
	state_dict = checkpoint.get('state_dict', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in list(hparams.items())[:20]:
			print(f"  {k}: {v}")

	sde_type = hparams.get('sde_type', hparams.get('sde', 'vp')).lower()
	beta_min = hparams.get('beta_min', 0.1)
	beta_max = hparams.get('beta_max', 20.0)
	sigma_min = hparams.get('sigma_min', 0.01)
	sigma_max = hparams.get('sigma_max', 50.0)
	n_sampling_steps = hparams.get('n_sampling_steps', 1000)
	use_param_conditioning = hparams.get('use_param_conditioning', False)

	conditioning_channels = getattr(config, 'conditioning_channels', 1)
	large_scale_channels = getattr(config, 'large_scale_channels', 3)
	n_params = getattr(config, 'n_params', 35) if use_param_conditioning else 0

	total_spatial_cond = 1 + large_scale_channels
	output_channels = 3

	if use_param_conditioning:
		condition_types = ['input', 'vector']
	else:
		condition_types = ['input']

	nf = hparams.get('nf', getattr(config, 'nf', 96))
	ch_mult_str = hparams.get('ch_mult', getattr(config, 'ch_mult', '1,2,4,8'))
	if isinstance(ch_mult_str, str):
		ch_mult = tuple(map(int, ch_mult_str.split(',')))
	else:
		ch_mult = tuple(ch_mult_str) if hasattr(ch_mult_str, '__iter__') else (1, 2, 4, 8)

	if verbose:
		print(f"[ModelManager] DDPM Model Configuration:")
		print(f"  SDE type: {sde_type}")
		print(f"  nf: {nf}, ch_mult: {ch_mult}")
		print(f"  Spatial conditioning channels: {total_spatial_cond}")
		print(f"  Parameter conditioning: {use_param_conditioning} ({n_params} params)")
		print(f"  Sampling steps: {n_sampling_steps}")

	net_kwargs = {
		'channels': output_channels,
		'dimensions': 2,
		'nf': nf,
		'ch_mult': ch_mult,
		'attention': hparams.get('attention', True),
		'condition': condition_types,
		'condition_input_channels': total_spatial_cond,
	}
	if use_param_conditioning:
		net_kwargs['condition_vector_channels'] = n_params

	net = NCSNpp(**net_kwargs)

	if sde_type == 'vp':
		score_model = ScoreModel(
			model=net,
			sde='vp',
			beta_min=beta_min,
			beta_max=beta_max,
			T=1.0,
			epsilon=1e-5,
			device=device,
		)
	else:
		score_model = ScoreModel(
			model=net,
			sde='ve',
			sigma_min=sigma_min,
			sigma_max=sigma_max,
			T=1.0,
			device=device,
		)

	if verbose:
		print(f"[ModelManager] Created ScoreModel with {sde_type.upper()}-SDE")
		print(f"[ModelManager] Loading weights from Lightning checkpoint...")

	model_state = {}
	for key, value in state_dict.items():
		if key.startswith('score_model.model.'):
			new_key = key.replace('score_model.model.', '')
			model_state[new_key] = value
		elif key.startswith('model.score_model.model.'):
			new_key = key.replace('model.score_model.model.', '')
			model_state[new_key] = value
		elif key.startswith('model.'):
			new_key = key.replace('model.', '')
			model_state[new_key] = value

	if len(model_state) == 0:
		raise ValueError(
			f"Could not extract model weights from checkpoint. "
			f"State dict keys: {list(state_dict.keys())[:10]}..."
		)

	missing, unexpected = score_model.model.load_state_dict(model_state, strict=False)

	if verbose:
		print(f"[ModelManager] Loaded {len(model_state)} weight tensors")
		if missing:
			print(
				f"[ModelManager] Missing keys: {missing[:5]}..." if len(missing) > 5 else f"[ModelManager] Missing keys: {missing}"
			)
		if unexpected:
			print(
				f"[ModelManager] Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"[ModelManager] Unexpected keys: {unexpected}"
			)

	score_model.model.eval()
	score_model.model.to(device)

	class DDPMModelWrapper:
		"""Wrapper that provides BIND-compatible interface for score_models.ScoreModel."""

		def __init__(self, score_model, n_sampling_steps, use_param_conditioning, hparams):
			self.score_model = score_model
			self.n_sampling_steps = n_sampling_steps
			self.use_param_conditioning = use_param_conditioning
			self.hparams = type('HParams', (), hparams)()
			self.hparams.n_sampling_steps = n_sampling_steps
			self._device = device

		def to(self, device):
			self._device = device
			self.score_model.model.to(device)
			return self

		def eval(self):
			self.score_model.model.eval()
			return self

		def train(self, mode=True):
			self.score_model.model.train(mode)
			return self

		def parameters(self):
			return self.score_model.model.parameters()

		def draw_samples(self, conditioning, batch_size, n_sampling_steps=None, param_conditioning=None, verbose=False):
			"""BIND-compatible sampling interface."""

			_ = verbose  # Unused but kept for signature parity
			_, _, height, width = conditioning.shape
			steps = n_sampling_steps or self.n_sampling_steps

			condition_list = [conditioning.to(self._device)]
			if param_conditioning is not None:
				condition_list.append(param_conditioning.to(self._device))

			with torch.no_grad():
				samples = self.score_model.sample(
					shape=[batch_size, 3, height, width],
					steps=steps,
					condition=condition_list,
				)

			return samples

	model = DDPMModelWrapper(
		score_model=score_model,
		n_sampling_steps=n_sampling_steps,
		use_param_conditioning=use_param_conditioning,
		hparams=dict(hparams),
	)

	if verbose:
		n_params_total = sum(p.numel() for p in score_model.model.parameters())
		print(f"[ModelManager] ✓ DDPM model loaded successfully")
		print(f"[ModelManager] Model parameters: {n_params_total:,}")
		print(f"[ModelManager] Sampling steps: {n_sampling_steps}")

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, model


def initialize_clean(config, verbose=False, skip_data_loading=False):
	"""Initialize a clean VDM model (single 3-channel model)."""

	verbose_flag = getattr(config, 'verbose', False) or verbose

	if verbose_flag:
		print("[ModelManager] Initializing CLEAN model (single 3-channel model)...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Dataset: {config.dataset}")
		print(f"[ModelManager] Data root: {config.data_root}")
		print(f"[ModelManager] Batch size: {config.batch_size}")
		if skip_data_loading:
			print("[ModelManager] ⚠️  Skipping data loading (using pre-loaded samples)")
		print(f"[ModelManager] BEFORE auto-detect: conditioning_channels = {getattr(config, 'conditioning_channels', 'NOT SET')}")
		print(f"[ModelManager] BEFORE auto-detect: large_scale_channels = {getattr(config, 'large_scale_channels', 'NOT SET')}")
		print(f"[ModelManager] BEFORE auto-detect: use_fourier_features = {getattr(config, 'use_fourier_features', 'NOT SET')}")
		print(f"[ModelManager] BEFORE auto-detect: fourier_legacy = {getattr(config, 'fourier_legacy', 'NOT SET')}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"Clean model requires a valid checkpoint. Got: {config.best_ckpt}")

	# Ensure required config attributes exist with sensible defaults
	config.use_cross_attention = getattr(config, 'use_cross_attention', False)
	config.cross_attention_location = getattr(config, 'cross_attention_location', 'bottleneck')
	config.cross_attention_heads = getattr(config, 'cross_attention_heads', 8)
	config.cross_attention_dropout = getattr(config, 'cross_attention_dropout', 0.1)
	config.use_chunked_cross_attention = getattr(config, 'use_chunked_cross_attention', True)
	config.cross_attention_chunk_size = getattr(config, 'cross_attention_chunk_size', 512)
	config.downsample_cross_attn_cond = getattr(config, 'downsample_cross_attn_cond', False)
	config.cross_attn_cond_downsample_factor = getattr(config, 'cross_attn_cond_downsample_factor', 2)
	config.use_fourier_features = getattr(config, 'use_fourier_features', True)
	config.fourier_legacy = getattr(config, 'fourier_legacy', None)

	if config.best_ckpt is not None:
		try:
			state_dict = torch.load(config.best_ckpt, map_location='cuda', weights_only=False)["state_dict"]

			cross_attn_keys = [k for k in state_dict.keys() if 'cross_attn' in k or 'mid_cross' in k]
			has_cross_attention = len(cross_attn_keys) > 0

			if has_cross_attention and verbose_flag:
				print(f"[ModelManager] Detected cross-attention model from checkpoint")

			if has_cross_attention:
				kv_keys = [
					k
					for k in state_dict.keys()
					if ('to_k.weight' in k or 'to_v.weight' in k) and 'cross' in k
				]
				if kv_keys:
					kv_shape = state_dict[kv_keys[0]].shape
					total_cond_channels = kv_shape[1]
					config.conditioning_channels = 1
					config.large_scale_channels = total_cond_channels - 1

					if verbose_flag:
						print(f"[ModelManager] Cross-attention conditioning channel breakdown:")
						print(f"  K/V projection input: {total_cond_channels}")
						print(f"  = conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels})")

					has_cross_attention = True
				else:
					if verbose_flag:
						print(f"[ModelManager] Warning: Cross-attention detected but no K/V projections found")
					has_cross_attention = False
			else:
				has_cross_attention = False

			has_legacy_fourier = 'model.score_model.fourier_features.freqs_exponent' in state_dict
			has_new_fourier_halo = 'model.score_model.fourier_features_halo.frequencies' in state_dict
			has_new_fourier_largescale = 'model.score_model.fourier_features_largescale.frequencies' in state_dict

			if config.fourier_legacy is None:
				if has_legacy_fourier:
					config.fourier_legacy = True
					config.use_fourier_features = True
					if verbose_flag:
						print(f"[ModelManager] Auto-detected LEGACY Fourier features from checkpoint")
				elif has_new_fourier_halo or has_new_fourier_largescale:
					config.fourier_legacy = False
					config.use_fourier_features = True
					if verbose_flag:
						print(f"[ModelManager] Auto-detected NEW multi-scale Fourier features from checkpoint")
				else:
					config.fourier_legacy = False
					config.use_fourier_features = False
					if verbose_flag:
						print(f"[ModelManager] No Fourier features detected in checkpoint")
			elif verbose_flag:
				print(f"[ModelManager] Using fourier_legacy={config.fourier_legacy} from config file (not auto-detecting)")

			if not has_cross_attention and 'model.score_model.conv_in.weight' in state_dict:
				conv_in_shape = state_dict['model.score_model.conv_in.weight'].shape
				total_in_channels = conv_in_shape[1]
				input_channels = 3

				if not config.use_fourier_features:
					total_conditioning = total_in_channels - input_channels
					config.conditioning_channels = 1
					config.large_scale_channels = total_conditioning - 1
					if verbose_flag:
						print(f"[ModelManager] No Fourier features mode channel breakdown:")
						print(f"  Total conv_in: {total_in_channels}")
						print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels})")
				elif config.fourier_legacy:
					fourier_features = 8
					total_conditioning = total_in_channels - input_channels - fourier_features
					config.conditioning_channels = 1
					config.large_scale_channels = total_conditioning - 1
					if verbose_flag:
						print(f"[ModelManager] Legacy Fourier mode channel breakdown:")
						print(f"  Total conv_in: {total_in_channels}")
						print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels}) + fourier_legacy({fourier_features})")
				else:
					if (total_in_channels - 12) % 9 == 0:
						config.large_scale_channels = (total_in_channels - 12) // 9
						config.conditioning_channels = 1
						if verbose_flag:
							per_channel = 8
							print(f"[ModelManager] New multi-scale Fourier mode channel breakdown:")
							print(f"  Total conv_in: {total_in_channels}")
							print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + fourier_halo({per_channel}) + large_scale({config.large_scale_channels}) + fourier_largescale({per_channel * config.large_scale_channels})")
					else:
						if verbose_flag:
							print(f"[ModelManager] ERROR: Channel count {total_in_channels} doesn't match expected formula for new Fourier mode")
							print(f"[ModelManager] Expected: 12 + 9*N where N = large_scale_channels")
							print(f"[ModelManager] This may indicate a config/checkpoint mismatch")
						total_conditioning = total_in_channels - input_channels
						config.conditioning_channels = 1
						config.large_scale_channels = total_conditioning - 1
						if verbose_flag:
							print(f"[ModelManager] Attempting fallback calculation:")
							print(f"  Total conv_in: {total_in_channels}")
							print(f"  = input({input_channels}) + conditioning({config.conditioning_channels}) + large_scale({config.large_scale_channels}) + ??? (unknown Fourier channels)")
							print(f"  WARNING: This may result in errors during model loading!")

			if verbose_flag:
				print(f"[ModelManager] Final channel configuration:")
				print(f"  conditioning_channels: {config.conditioning_channels}")
				print(f"  large_scale_channels: {config.large_scale_channels}")
				print(f"  fourier_legacy: {config.fourier_legacy}")
		except Exception as exc:
			if verbose_flag:
				print(f"[ModelManager] Could not auto-detect from checkpoint: {exc}")
			config.conditioning_channels = None

	if getattr(config, 'conditioning_channels', None) is None:
		config.conditioning_channels = 1
		if not hasattr(config, 'large_scale_channels'):
			config.large_scale_channels = 0
		if verbose_flag:
			print(
				f"[ModelManager] Using config values: conditioning_channels={config.conditioning_channels}, "
				f"large_scale_channels={config.large_scale_channels}"
			)

	if verbose_flag:
		print(f"[ModelManager] Setting up parameter conditioning...")

	use_param_conditioning = getattr(config, 'use_param_conditioning', False)
	param_norm_path = getattr(config, 'param_norm_path', None)
	param_min = getattr(config, 'min', None)
	param_max = getattr(config, 'max', None)

	if param_norm_path:
		use_param_conditioning = True
		if os.path.exists(param_norm_path):
			minmax_df = pd.read_csv(param_norm_path)
			param_min = np.array(minmax_df['MinVal'].values)
			param_max = np.array(minmax_df['MaxVal'].values)
			if verbose_flag:
				print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")
		else:
			if verbose_flag:
				print(f"[ModelManager] WARNING: param_norm_path not found: {param_norm_path}")

	if verbose_flag:
		if not skip_data_loading:
			print(f"[ModelManager] Loading dataset (stage='test')...")
		else:
			print(f"[ModelManager] Skipping dataset loading (inference-only mode)")

	seed_everything(config.seed)

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose_flag)

	if verbose_flag:
		print(f"[ModelManager] Creating UNetVDM model...")
		print(f"[ModelManager] Model configuration:")
		print(f"  - use_fourier_features: {config.use_fourier_features}")
		print(f"  - fourier_legacy: {config.fourier_legacy}")
		print(f"  - conditioning_channels: {config.conditioning_channels}")
		print(f"  - large_scale_channels: {config.large_scale_channels}")
		print(f"  - use_cross_attention: {config.use_cross_attention}")
		if config.use_cross_attention:
			print(f"    • location: {config.cross_attention_location}")
			print(f"    • heads: {config.cross_attention_heads}")
			print(f"    • dropout: {config.cross_attention_dropout}")
			print(f"    • chunked: {config.use_chunked_cross_attention} (chunk_size={config.cross_attention_chunk_size})")

	score_model = networks.UNetVDM(
		input_channels=3,
		gamma_min=config.gamma_min,
		gamma_max=config.gamma_max,
		embedding_dim=config.embedding_dim,
		norm_groups=config.norm_groups,
		n_blocks=config.n_blocks,
		add_attention=True,
		n_attention_heads=config.n_attention_heads,
		use_fourier_features=config.use_fourier_features,
		legacy_fourier=config.fourier_legacy,
		use_param_conditioning=use_param_conditioning,
		param_min=param_min,
		param_max=param_max,
		conditioning_channels=config.conditioning_channels,
		large_scale_channels=config.large_scale_channels,
		use_cross_attention=config.use_cross_attention,
		cross_attention_location=config.cross_attention_location,
		cross_attention_heads=config.cross_attention_heads,
		cross_attention_dropout=config.cross_attention_dropout,
		use_chunked_cross_attention=config.use_chunked_cross_attention,
		cross_attention_chunk_size=config.cross_attention_chunk_size,
		downsample_cross_attn_cond=config.downsample_cross_attn_cond,
		cross_attn_cond_downsample_factor=config.cross_attn_cond_downsample_factor,
	)

	if verbose_flag:
		print(f"[ModelManager] UNetVDM created. Wrapping in LightCleanVDM...")

	channel_weights_str = getattr(config, 'channel_weights', '1.0,1.0,2.0')
	channel_weights = tuple(map(float, channel_weights_str.split(',')))

	lambdas_value = getattr(config, 'lambdas', '1.0,1.0,1.0')

	image_shape = (3, config.cropsize, config.cropsize)

	vdm_hydro = vdm_module.LightCleanVDM(
		score_model=score_model,
		learning_rate=config.learning_rate,
		lr_scheduler=getattr(config, 'lr_scheduler', 'cosine'),
		gamma_min=config.gamma_min,
		gamma_max=config.gamma_max,
		image_shape=image_shape,
		noise_schedule=config.noise_schedule,
		data_noise=getattr(config, 'data_noise', 1e-5),
		antithetic_time_sampling=getattr(config, 'antithetic_time_sampling', True),
		lambdas=lambdas_value,
		channel_weights=(1, 1, 1),
		use_focal_loss=getattr(config, 'use_focal_loss', False),
		focal_gamma=getattr(config, 'focal_gamma', 2.0),
		use_param_prediction=getattr(config, 'use_param_prediction', False),
		param_prediction_weight=getattr(config, 'param_prediction_weight', 0.01),
	)

	if verbose_flag:
		print(f"[ModelManager] Loading checkpoint weights from disk...")

	state_dict = torch.load(config.best_ckpt, map_location='cuda', weights_only=False)["state_dict"]

	if verbose_flag:
		print(f"[ModelManager] Checkpoint loaded. Loading state dict into model...")

	model_state = vdm_hydro.state_dict()
	missing_keys = [k for k in model_state.keys() if k not in state_dict]
	if missing_keys and verbose_flag:
		print(
			f"[ModelManager] Warning: checkpoint is missing {len(missing_keys)} keys. "
			f"Injecting defaults from model: {missing_keys}"
		)
	for key in missing_keys:
		state_dict[key] = model_state[key]

	vdm_hydro.load_state_dict(state_dict)
	vdm_hydro = vdm_hydro.eval()

	if verbose_flag:
		print("[ModelManager] Model and weights loaded successfully.")
		print(f"[ModelManager] Model checkpoint: {config.best_ckpt}")
		print(f"[ModelManager] Using {config.conditioning_channels} conditioning channel(s)")
		print(f"[ModelManager] Using {config.large_scale_channels} large-scale channel(s)")
		print(
			f"[ModelManager] Fourier features: {'LEGACY' if config.fourier_legacy else 'MULTI-SCALE' if config.use_fourier_features else 'DISABLED'}"
		)

	return hydro, vdm_hydro


def initialize_ot_flow(config, verbose=False, skip_data_loading=False):
	"""Initialize an Optimal Transport Flow Matching model."""

	if verbose:
		print("[ModelManager] Initializing OT FLOW MATCHING model...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	from vdm.ot_flow_model import LightOTFlow, OTVelocityNetWrapper
	from vdm.networks_clean import UNetVDM

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"OT Flow model requires a valid checkpoint. Got: {config.best_ckpt}")

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})
	state_dict = checkpoint.get('state_dict', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in list(hparams.items())[:15]:
			print(f"  {k}: {v}")

	n_sampling_steps = hparams.get('n_sampling_steps', 50)
	x0_mode = hparams.get('x0_mode', 'zeros')
	ot_method = hparams.get('ot_method', 'exact')
	ot_reg = hparams.get('ot_reg', 0.01)
	use_stochastic_interpolant = hparams.get('use_stochastic_interpolant', False)
	sigma = hparams.get('sigma', 0.0)
	learning_rate = hparams.get('learning_rate', 1e-4)

	conditioning_channels = getattr(config, 'conditioning_channels', 1)
	large_scale_channels = getattr(config, 'large_scale_channels', 3)
	output_channels = 3

	embedding_dim = getattr(config, 'embedding_dim', 256)
	n_blocks = getattr(config, 'n_blocks', 32)
	norm_groups = getattr(config, 'norm_groups', 8)
	n_attention_heads = getattr(config, 'n_attention_heads', 8)
	use_fourier_features = getattr(config, 'use_fourier_features', True)
	fourier_legacy = getattr(config, 'fourier_legacy', False)
	add_attention = getattr(config, 'add_attention', True)
	use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))

	if verbose:
		print(f"[ModelManager] OT Flow Model Configuration:")
		print(f"  x0 mode: {x0_mode}")
		print(f"  OT method: {ot_method}")
		print(f"  OT regularization: {ot_reg}")
		print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
		print(f"  Sampling steps: {n_sampling_steps}")
		print(f"  Conditioning channels: {conditioning_channels + large_scale_channels}")
		print(f"  Param conditioning: {use_param_conditioning}")

	param_min = None
	param_max = None
	if use_param_conditioning:
		param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
		if param_norm_path and os.path.exists(param_norm_path):
			minmax_df = pd.read_csv(param_norm_path)
			param_min = np.array(minmax_df['MinVal'])
			param_max = np.array(minmax_df['MaxVal'])
			if verbose:
				print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")

	unet = UNetVDM(
		input_channels=output_channels,
		conditioning_channels=conditioning_channels,
		large_scale_channels=large_scale_channels,
		embedding_dim=embedding_dim,
		n_blocks=n_blocks,
		norm_groups=norm_groups,
		n_attention_heads=n_attention_heads,
		use_fourier_features=use_fourier_features,
		legacy_fourier=fourier_legacy,
		add_attention=add_attention,
		use_param_conditioning=use_param_conditioning,
		param_min=param_min,
		param_max=param_max,
	)

	velocity_model = OTVelocityNetWrapper(
		net=unet,
		output_channels=output_channels,
		conditioning_channels=conditioning_channels + large_scale_channels,
	)

	model = LightOTFlow(
		velocity_model=velocity_model,
		learning_rate=learning_rate,
		n_sampling_steps=n_sampling_steps,
		use_stochastic_interpolant=use_stochastic_interpolant,
		sigma=sigma,
		x0_mode=x0_mode,
		use_param_conditioning=use_param_conditioning,
		ot_method=ot_method,
		ot_reg=ot_reg,
	)

	if verbose:
		print(f"[ModelManager] Loading state dict into LightOTFlow...")

	model.load_state_dict(state_dict)
	model = model.eval().to(device)

	if verbose:
		n_params_total = sum(p.numel() for p in model.parameters())
		print(f"[ModelManager] ✓ OT Flow model loaded successfully")
		print(f"[ModelManager] Model parameters: {n_params_total:,}")
		print(f"[ModelManager] Sampling steps: {n_sampling_steps}")

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, model


def initialize_consistency(config, verbose=False, skip_data_loading=False):
	"""Initialize a Consistency Model (Song et al., 2023)."""

	if verbose:
		print("[ModelManager] Initializing CONSISTENCY model (single/few-step sampling)...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	from vdm.consistency_model import (
		ConsistencyFunction,
		ConsistencyModel,
		ConsistencyNetWrapper,
		ConsistencyNoiseSchedule,
		LightConsistency,
	)
	from vdm.networks_clean import UNetVDM

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"Consistency model requires a valid checkpoint. Got: {config.best_ckpt}")

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})
	state_dict = checkpoint.get('state_dict', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in list(hparams.items())[:15]:
			print(f"  {k}: {v}")

	n_sampling_steps = hparams.get('n_sampling_steps', 1)
	ct_n_steps = hparams.get('ct_n_steps', 18)
	x0_mode = hparams.get('x0_mode', 'zeros')
	sigma_min = hparams.get('sigma_min', 0.002)
	sigma_max = hparams.get('sigma_max', 80.0)
	sigma_data = hparams.get('sigma_data', 0.5)
	learning_rate = hparams.get('learning_rate', 1e-4)

	conditioning_channels = getattr(config, 'conditioning_channels', 1)
	large_scale_channels = getattr(config, 'large_scale_channels', 3)
	output_channels = 3

	embedding_dim = getattr(config, 'embedding_dim', 256)
	n_blocks = getattr(config, 'n_blocks', 32)
	norm_groups = getattr(config, 'norm_groups', 8)
	n_attention_heads = getattr(config, 'n_attention_heads', 8)
	use_fourier_features = getattr(config, 'use_fourier_features', True)
	fourier_legacy = getattr(config, 'fourier_legacy', False)
	add_attention = getattr(config, 'add_attention', True)
	use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))

	if verbose:
		print(f"[ModelManager] Consistency Model Configuration:")
		print(f"  Sampling steps: {n_sampling_steps}")
		print(f"  CT discretization steps: {ct_n_steps}")
		print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
		print(f"  Conditioning channels: {conditioning_channels + large_scale_channels}")
		print(f"  Param conditioning: {use_param_conditioning}")

	param_min = None
	param_max = None
	if use_param_conditioning:
		param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
		if param_norm_path and os.path.exists(param_norm_path):
			minmax_df = pd.read_csv(param_norm_path)
			param_min = np.array(minmax_df['MinVal'])
			param_max = np.array(minmax_df['MaxVal'])
			if verbose:
				print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")

	unet = UNetVDM(
		input_channels=output_channels,
		conditioning_channels=conditioning_channels,
		large_scale_channels=large_scale_channels,
		embedding_dim=embedding_dim,
		n_blocks=n_blocks,
		norm_groups=norm_groups,
		n_attention_heads=n_attention_heads,
		use_fourier_features=use_fourier_features,
		legacy_fourier=fourier_legacy,
		add_attention=add_attention,
		use_param_conditioning=use_param_conditioning,
		param_min=param_min,
		param_max=param_max,
	)

	net_wrapper = ConsistencyNetWrapper(
		net=unet,
		output_channels=output_channels,
		conditioning_channels=conditioning_channels + large_scale_channels,
	)

	noise_schedule = ConsistencyNoiseSchedule(
		sigma_min=sigma_min,
		sigma_max=sigma_max,
	)

	consistency_fn = ConsistencyFunction(
		net=net_wrapper,
		sigma_data=sigma_data,
		sigma_min=sigma_min,
	)

	consistency_model = ConsistencyModel(
		consistency_fn=consistency_fn,
		noise_schedule=noise_schedule,
		sigma_data=sigma_data,
	)

	model = LightConsistency(
		consistency_model=consistency_model,
		learning_rate=learning_rate,
		n_sampling_steps=n_sampling_steps,
		x0_mode=x0_mode,
		use_param_conditioning=use_param_conditioning,
		ct_n_steps=ct_n_steps,
	)

	if verbose:
		print(f"[ModelManager] Loading state dict into LightConsistency...")

	missing, unexpected = model.load_state_dict(state_dict, strict=False)

	unexpected_filtered = [k for k in unexpected if not k.startswith('target_model.')]

	if verbose:
		if unexpected:
			n_target = len([k for k in unexpected if k.startswith('target_model.')])
			print(f"[ModelManager] Skipped {n_target} target_model keys (EMA copy, not needed for inference)")
		if unexpected_filtered:
			print(f"[ModelManager] WARNING: {len(unexpected_filtered)} unexpected keys: {unexpected_filtered[:5]}...")
		if missing:
			print(f"[ModelManager] WARNING: {len(missing)} missing keys: {missing[:5]}...")

	model = model.eval().to(device)

	if verbose:
		n_params_total = sum(p.numel() for p in model.parameters())
		print(f"[ModelManager] ✓ Consistency model loaded successfully")
		print(f"[ModelManager] Model parameters: {n_params_total:,}")
		print(f"[ModelManager] Sampling steps: {n_sampling_steps}")

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, model


def initialize_dit(config, verbose=False, skip_data_loading=False):
	"""Initialize a DiT (Diffusion Transformer) model."""

	if verbose:
		print("[ModelManager] Initializing DiT (Diffusion Transformer) model...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	from vdm.dit_model import LightDiTVDM

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"DiT model requires a valid checkpoint. Got: {config.best_ckpt}")

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})
	state_dict = checkpoint.get('state_dict', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in list(hparams.items())[:20]:
			print(f"  {k}: {v}")

	dit_variant = hparams.get('dit_model', getattr(config, 'dit_variant', 'DiT-B/4'))
	learning_rate = hparams.get('learning_rate', getattr(config, 'learning_rate', 1e-4))
	weight_decay = hparams.get('weight_decay', getattr(config, 'weight_decay', 1e-5))
	gamma_min = hparams.get('gamma_min', getattr(config, 'gamma_min', -13.3))
	gamma_max = hparams.get('gamma_max', getattr(config, 'gamma_max', 5.0))
	n_sampling_steps = hparams.get('n_sampling_steps', getattr(config, 'n_sampling_steps', 256))
	loss_type = hparams.get('loss_type', getattr(config, 'loss_type', 'mse'))

	conditioning_channels = hparams.get('conditioning_channels', getattr(config, 'conditioning_channels', 1))
	large_scale_channels = hparams.get('large_scale_channels', getattr(config, 'large_scale_channels', 3))
	use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))
	n_params = hparams.get('n_params', getattr(config, 'Nparams', 35))

	patch_size = hparams.get('patch_size', getattr(config, 'patch_size', 4))
	hidden_size = hparams.get('hidden_size', getattr(config, 'hidden_size', 768))
	depth = hparams.get('depth', getattr(config, 'depth', 12))
	num_heads = hparams.get('num_heads', getattr(config, 'num_heads', 12))
	mlp_ratio = hparams.get('mlp_ratio', getattr(config, 'mlp_ratio', 4.0))
	dropout = hparams.get('dropout', getattr(config, 'dropout', 0.0))

	if 'score_model.pos_embed' in state_dict:
		num_patches = state_dict['score_model.pos_embed'].shape[1]
		detected_img_size = int((num_patches ** 0.5) * patch_size)
		img_size = detected_img_size
		if verbose:
			print(f"[ModelManager] Auto-detected img_size={img_size} from pos_embed (num_patches={num_patches})")
	else:
		img_size = getattr(config, 'cropsize', 128)
		if verbose:
			print(f"[ModelManager] Using img_size={img_size} from config.cropsize")

	if verbose:
		print(f"[ModelManager] DiT Model Configuration:")
		print(f"  Variant: {dit_variant}")
		print(f"  Image size: {img_size}")
		print(f"  Patch size: {patch_size}")
		print(f"  Hidden size: {hidden_size}, Depth: {depth}, Heads: {num_heads}")
		print(f"  Gamma range: [{gamma_min}, {gamma_max}]")
		print(f"  Sampling steps: {n_sampling_steps}")
		print(f"  Conditioning: {conditioning_channels} + {large_scale_channels} large-scale")
		print(f"  Param conditioning: {use_param_conditioning} ({n_params} params)")

	model = LightDiTVDM(
		dit_model=dit_variant,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		gamma_min=gamma_min,
		gamma_max=gamma_max,
		n_sampling_steps=n_sampling_steps,
		loss_type=loss_type,
		image_shape=(3, img_size, img_size),
		img_size=img_size,
		patch_size=patch_size,
		hidden_size=hidden_size,
		depth=depth,
		num_heads=num_heads,
		mlp_ratio=mlp_ratio,
		n_params=n_params,
		conditioning_channels=conditioning_channels,
		large_scale_channels=large_scale_channels,
		dropout=dropout,
	)

	if verbose:
		print(f"[ModelManager] Loading state dict into LightDiTVDM...")

	model.load_state_dict(state_dict)
	model = model.eval().to(device)

	if verbose:
		n_params_total = sum(p.numel() for p in model.parameters())
		print(f"[ModelManager] ✓ DiT model loaded successfully")
		print(f"[ModelManager] Model parameters: {n_params_total:,}")
		print(f"[ModelManager] Sampling steps: {n_sampling_steps}")

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, model


def initialize_interpolant(config, verbose=False, skip_data_loading=False):
	"""Initialize an Interpolant/Flow Matching model."""

	if verbose:
		print("[ModelManager] Initializing INTERPOLANT/Flow Matching model...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	from vdm.interpolant_model import LightInterpolant, VelocityNetWrapper
	from vdm.networks_clean import UNetVDM

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"Interpolant model requires a valid checkpoint. Got: {config.best_ckpt}")

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})
	state_dict = checkpoint.get('state_dict', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in list(hparams.items())[:15]:
			print(f"  {k}: {v}")

	n_sampling_steps = hparams.get('n_sampling_steps', 50)
	x0_mode = hparams.get('x0_mode', 'zeros')
	use_stochastic_interpolant = hparams.get('use_stochastic_interpolant', False)
	sigma = hparams.get('sigma', 0.0)
	learning_rate = hparams.get('learning_rate', 1e-4)

	conditioning_channels = getattr(config, 'conditioning_channels', 1)
	large_scale_channels = getattr(config, 'large_scale_channels', 3)
	output_channels = 3

	embedding_dim = getattr(config, 'embedding_dim', 256)
	n_blocks = getattr(config, 'n_blocks', 32)
	norm_groups = getattr(config, 'norm_groups', 8)
	n_attention_heads = getattr(config, 'n_attention_heads', 8)
	use_fourier_features = getattr(config, 'use_fourier_features', True)
	fourier_legacy = getattr(config, 'fourier_legacy', False)
	add_attention = getattr(config, 'add_attention', True)
	use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))

	if verbose:
		print(f"[ModelManager] Interpolant Model Configuration:")
		print(f"  x0 mode: {x0_mode}")
		print(f"  Stochastic: {use_stochastic_interpolant} (sigma={sigma})")
		print(f"  Sampling steps: {n_sampling_steps}")
		print(f"  Conditioning channels: {conditioning_channels + large_scale_channels}")
		print(f"  Param conditioning: {use_param_conditioning}")

	param_min = None
	param_max = None
	if use_param_conditioning:
		param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
		if param_norm_path and os.path.exists(param_norm_path):
			minmax_df = pd.read_csv(param_norm_path)
			param_min = np.array(minmax_df['MinVal'])
			param_max = np.array(minmax_df['MaxVal'])
			if verbose:
				print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")

	unet = UNetVDM(
		input_channels=output_channels,
		conditioning_channels=conditioning_channels,
		large_scale_channels=large_scale_channels,
		embedding_dim=embedding_dim,
		n_blocks=n_blocks,
		norm_groups=norm_groups,
		n_attention_heads=n_attention_heads,
		use_fourier_features=use_fourier_features,
		legacy_fourier=fourier_legacy,
		add_attention=add_attention,
		use_param_conditioning=use_param_conditioning,
		param_min=param_min,
		param_max=param_max,
	)

	velocity_model = VelocityNetWrapper(
		net=unet,
		output_channels=output_channels,
		conditioning_channels=conditioning_channels + large_scale_channels,
	)

	model = LightInterpolant(
		velocity_model=velocity_model,
		learning_rate=learning_rate,
		n_sampling_steps=n_sampling_steps,
		use_stochastic_interpolant=use_stochastic_interpolant,
		sigma=sigma,
		x0_mode=x0_mode,
		use_param_conditioning=use_param_conditioning,
	)

	if verbose:
		print(f"[ModelManager] Loading state dict into LightInterpolant...")

	model.load_state_dict(state_dict)
	model = model.eval().to(device)

	if verbose:
		n_params_total = sum(p.numel() for p in model.parameters())
		print(f"[ModelManager] ✓ Interpolant model loaded successfully")
		print(f"[ModelManager] Model parameters: {n_params_total:,}")
		print(f"[ModelManager] Sampling steps: {n_sampling_steps}")

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, model


def initialize_dsm(config, verbose=False, skip_data_loading=False):
	"""Initialize a DSM (Denoising Score Matching) model."""

	if verbose:
		print("[ModelManager] Initializing DSM (Denoising Score Matching) model...")
		print(f"[ModelManager] Using seed: {config.seed}")
		print(f"[ModelManager] Checkpoint path: {getattr(config, 'best_ckpt', 'NOT SET')}")

	seed_everything(config.seed)

	from vdm.dsm_model import LightDSM
	from vdm.networks_clean import UNetVDM

	if not config.best_ckpt or not os.path.exists(config.best_ckpt):
		raise ValueError(f"DSM model requires a valid checkpoint. Got: {config.best_ckpt}")

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	checkpoint = torch.load(config.best_ckpt, map_location=device, weights_only=False)
	hparams = checkpoint.get('hyper_parameters', {})
	state_dict = checkpoint.get('state_dict', {})

	if verbose:
		print(f"[ModelManager] Loaded hyperparameters from checkpoint:")
		for k, v in list(hparams.items())[:15]:
			print(f"  {k}: {v}")

	beta_min = hparams.get('beta_min', 0.1)
	beta_max = hparams.get('beta_max', 20.0)
	n_sampling_steps = hparams.get('n_sampling_steps', 250)
	use_snr_weighting = hparams.get('use_snr_weighting', True)
	learning_rate = hparams.get('learning_rate', 1e-4)
	channel_weights = hparams.get('channel_weights', (1.0, 1.0, 1.0))

	conditioning_channels = getattr(config, 'conditioning_channels', 1)
	large_scale_channels = getattr(config, 'large_scale_channels', 3)
	output_channels = 3

	embedding_dim = getattr(config, 'embedding_dim', 96)
	n_blocks = getattr(config, 'n_blocks', 5)
	norm_groups = getattr(config, 'norm_groups', 8)
	n_attention_heads = getattr(config, 'n_attention_heads', 8)
	use_fourier_features = getattr(config, 'use_fourier_features', True)
	fourier_legacy = getattr(config, 'fourier_legacy', False)
	add_attention = getattr(config, 'add_attention', True)
	use_param_conditioning = hparams.get('use_param_conditioning', getattr(config, 'use_param_conditioning', True))

	if verbose:
		print(f"[ModelManager] DSM Model Configuration:")
		print(f"  Beta range: [{beta_min}, {beta_max}]")
		print(f"  Sampling steps: {n_sampling_steps}")
		print(f"  SNR weighting: {use_snr_weighting}")
		print(f"  Conditioning channels: {conditioning_channels + large_scale_channels}")
		print(f"  Param conditioning: {use_param_conditioning}")

	param_min = None
	param_max = None
	if use_param_conditioning:
		param_norm_path = hparams.get('param_norm_path', getattr(config, 'param_norm_path', None))
		if param_norm_path and os.path.exists(param_norm_path):
			minmax_df = pd.read_csv(param_norm_path)
			param_min = np.array(minmax_df['MinVal'])
			param_max = np.array(minmax_df['MaxVal'])
			if verbose:
				print(f"[ModelManager] Loaded {len(param_min)} param bounds from {param_norm_path}")

	unet = UNetVDM(
		input_channels=output_channels,
		conditioning_channels=conditioning_channels,
		large_scale_channels=large_scale_channels,
		embedding_dim=embedding_dim,
		n_blocks=n_blocks,
		norm_groups=norm_groups,
		n_attention_heads=n_attention_heads,
		use_fourier_features=use_fourier_features,
		legacy_fourier=fourier_legacy,
		add_attention=add_attention,
		use_param_conditioning=use_param_conditioning,
		param_min=param_min,
		param_max=param_max,
	)

	model = LightDSM(
		score_model=unet,
		beta_min=beta_min,
		beta_max=beta_max,
		learning_rate=learning_rate,
		n_sampling_steps=n_sampling_steps,
		use_param_conditioning=use_param_conditioning,
		use_snr_weighting=use_snr_weighting,
		channel_weights=channel_weights,
	)

	if verbose:
		print(f"[ModelManager] Loading state dict into LightDSM...")

	model.load_state_dict(state_dict)
	model = model.eval().to(device)

	if verbose:
		n_params_total = sum(p.numel() for p in model.parameters())
		print(f"[ModelManager] ✓ DSM model loaded successfully")
		print(f"[ModelManager] Model parameters: {n_params_total:,}")
		print(f"[ModelManager] Sampling steps: {n_sampling_steps}")

	hydro = _load_dataset_if_requested(config, skip_data_loading, verbose)

	return hydro, model
