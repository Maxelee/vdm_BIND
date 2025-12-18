"""ModelManager orchestrates model initialization across model families."""

import os
from typing import Callable, Dict

import torch

from .model_initializers import (
    initialize_clean,
    initialize_consistency,
    initialize_ddpm,
    initialize_dit,
    initialize_dsm,
    initialize_interpolant,
    initialize_ot_flow,
    initialize_triple,
)


class ModelManager:
    """
    Model manager supporting multiple model types.

    Supported Model Types
    ---------------------
    - clean: Single 3-channel model (LightCleanVDM)
    - triple: Three independent 1-channel models (LightTripleVDM)
    - ddpm: Score-based diffusion model (score_models package)
    - dsm: Denoising Score Matching with custom UNet (LightDSM)
    - interpolant: Flow matching / stochastic interpolant (LightInterpolant)
    - consistency: Consistency models (Song et al., 2023) - single/few-step sampling
    - ot_flow: Optimal Transport Flow Matching (Lipman et al., 2022)
    - dit: Diffusion Transformer (Peebles & Xie, 2023) - LightDiTVDM

    The model type is auto-detected from:
    1. config.model_name containing keywords
    2. Checkpoint state_dict structure

    Examples
    --------
    >>> config = ConfigLoader('configs/interpolant.ini')
    >>> _, model = ModelManager.initialize(config, verbose='summary')
    >>> model = model.to('cuda').eval()
    """

    @staticmethod
    def detect_model_type(config, verbose=False):
        """
        Detect model type from config and/or checkpoint.

        Parameters
        ----------
        config : ConfigLoader
            Configuration object with model_name and best_ckpt.
        verbose : bool, optional
            Verbosity level.

        Returns
        -------
        str
            Model type: 'clean', 'triple', 'ddpm', 'dsm', 'interpolant',
            'consistency', 'ot_flow', or 'dit'
        """
        model_name = getattr(config, 'model_name', '').lower()

        # Check model_name for keywords (most specific first)
        if any(x in model_name for x in ['dit', 'diffusion_transformer', 'transformer']):
            if verbose:
                print(f"[ModelManager] Detected 'dit' model from model_name: {model_name}")
            return 'dit'

        if 'consistency' in model_name:
            if verbose:
                print(f"[ModelManager] Detected 'consistency' model from model_name: {model_name}")
            return 'consistency'

        if any(x in model_name for x in ['ot_flow', 'ot-flow', 'otflow', 'optimal_transport']):
            if verbose:
                print(f"[ModelManager] Detected 'ot_flow' model from model_name: {model_name}")
            return 'ot_flow'

        if 'dsm' in model_name:
            if verbose:
                print(f"[ModelManager] Detected 'dsm' model from model_name: {model_name}")
            return 'dsm'

        if any(x in model_name for x in ['interpolant', 'flow']):
            if verbose:
                print(f"[ModelManager] Detected 'interpolant' model from model_name: {model_name}")
            return 'interpolant'

        if any(x in model_name for x in ['ddpm', 'ncsnpp', 'score']):
            if verbose:
                print(f"[ModelManager] Detected 'ddpm' model from model_name: {model_name}")
            return 'ddpm'

        if 'triple' in model_name:
            if verbose:
                print(f"[ModelManager] Detected 'triple' model from model_name: {model_name}")
            return 'triple'

        # Check checkpoint state_dict structure
        if config.best_ckpt and os.path.exists(config.best_ckpt):
            try:
                checkpoint = torch.load(config.best_ckpt, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('state_dict', {})
                hparams = checkpoint.get('hyper_parameters', {})

                # DiT model
                dit_keys = [k for k in state_dict.keys() if 'score_model.blocks' in k or 'score_model.x_embedder' in k]
                if dit_keys or hparams.get('dit_variant') or hparams.get('patch_size'):
                    if verbose:
                        print(f"[ModelManager] Detected 'dit' model from checkpoint structure")
                    return 'dit'

                # Consistency model
                consistency_keys = [k for k in state_dict.keys() if 'consistency_model' in k or 'target_model' in k]
                if consistency_keys or hparams.get('ct_n_steps') or hparams.get('denoising_warmup_epochs'):
                    if verbose:
                        print(f"[ModelManager] Detected 'consistency' model from checkpoint structure")
                    return 'consistency'

                # OT flow model
                ot_flow_keys = [k for k in state_dict.keys() if 'ot_interpolant' in k]
                if ot_flow_keys or hparams.get('ot_method') or hparams.get('ot_reg'):
                    if verbose:
                        print(f"[ModelManager] Detected 'ot_flow' model from checkpoint structure")
                    return 'ot_flow'

                # Interpolant model
                interpolant_keys = [k for k in state_dict.keys() if 'interpolant' in k]
                if interpolant_keys or hparams.get('x0_mode'):
                    if verbose:
                        print(f"[ModelManager] Detected 'interpolant' model from checkpoint structure")
                    return 'interpolant'

                # DSM model
                dsm_keys = [k for k in state_dict.keys() if k.startswith('model.net.')]
                has_vp_schedule = hparams.get('beta_min') is not None and hparams.get('beta_max') is not None
                has_sde = hparams.get('sde_type') or hparams.get('sde')
                if dsm_keys and has_vp_schedule and not has_sde:
                    if verbose:
                        print(f"[ModelManager] Detected 'dsm' model from checkpoint structure")
                    return 'dsm'

                # DDPM model
                ddpm_keys = [k for k in state_dict.keys() if 'score_model' in k and 'model.score_model' not in k]
                if ddpm_keys or hparams.get('sde_type') or hparams.get('sde'):
                    if verbose:
                        print(f"[ModelManager] Detected 'ddpm' model from checkpoint structure")
                    return 'ddpm'

                # Triple model
                triple_keys = [
                    k for k in state_dict.keys()
                    if any(x in k for x in ['hydro_dm_model', 'gas_model', 'stars_model'])
                ]
                if triple_keys:
                    if verbose:
                        print(f"[ModelManager] Detected 'triple' model from checkpoint structure ({len(triple_keys)} keys)")
                    return 'triple'

            except Exception as exc:
                if verbose:
                    print(f"[ModelManager] Could not load checkpoint for model type detection: {exc}")

        if verbose:
            print(f"[ModelManager] Defaulting to 'clean' model type")
        return 'clean'

    @staticmethod
    def initialize(config, verbose=False, skip_data_loading=False):
        """
        Initialize the configured model and optionally the dataloader.

        Parameters
        ----------
        config : ConfigLoader
            Configuration object.
        verbose : bool, optional
            Print debug information.
        skip_data_loading : bool, optional
            If True, skip dataset loading (faster for inference-only).

        Returns
        -------
        tuple
            (dataloader_or_None, model)
        """
        model_type = ModelManager.detect_model_type(config, verbose=verbose)

        initializer_map: Dict[str, Callable] = {
            'dit': initialize_dit,
            'consistency': initialize_consistency,
            'ot_flow': initialize_ot_flow,
            'dsm': initialize_dsm,
            'interpolant': initialize_interpolant,
            'ddpm': initialize_ddpm,
            'triple': initialize_triple,
            'clean': initialize_clean,
        }

        initializer = initializer_map.get(model_type, initialize_clean)
        return initializer(config, verbose=verbose, skip_data_loading=skip_data_loading)
