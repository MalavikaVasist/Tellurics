"""Production inference pipeline."""

import torch
import numpy as np
import numpy.typing as npt

from tellurics.configs.inference import InferenceConfig
from tellurics.models.output import ModelOutput
from tellurics.training.module import TelluricLightningModule
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline for telluric correction.

    Loads a trained model and provides prediction methods for
    single spectra or batches.
    """

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the inference pipeline.

        Args:
            config: Inference configuration with checkpoint path and options.
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Load model from checkpoint
        logger.info(f"Loading model from {config.checkpoint_path}")
        self.module = TelluricLightningModule.load_from_checkpoint(
            str(config.checkpoint_path),
            map_location=self.device,
        )
        self.module.eval()
        self.module.to(self.device)
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def predict(
        self,
        observed: npt.NDArray[np.float64] | torch.Tensor,
        stellar: npt.NDArray[np.float64] | torch.Tensor,
        atm_params: npt.NDArray[np.float64] | torch.Tensor,
    ) -> ModelOutput:
        """Run inference on a single observation or batch.

        Computes R = X/S, predicts telluric, and optionally recovers planet.

        Args:
            observed: Observed spectrum X. Shape: (N,) or (B, N).
            stellar: Stellar template S. Shape: (N,) or (B, N).
            atm_params: Atmospheric parameters. Shape: (N_params,) or (B, N_params).

        Returns:
            ModelOutput with telluric prediction and optional planet recovery.
        """
        # Convert to tensors
        if isinstance(observed, np.ndarray):
            observed = torch.from_numpy(observed.astype(np.float32))
        if isinstance(stellar, np.ndarray):
            stellar = torch.from_numpy(stellar.astype(np.float32))
        if isinstance(atm_params, np.ndarray):
            atm_params = torch.from_numpy(atm_params.astype(np.float32))

        # Ensure batch dimension
        if observed.dim() == 1:
            observed = observed.unsqueeze(0)
            stellar = stellar.unsqueeze(0)
            atm_params = atm_params.unsqueeze(0)

        # Move to device
        observed = observed.to(self.device)
        stellar = stellar.to(self.device)
        atm_params = atm_params.to(self.device)

        # Compute R = X / S
        epsilon = 1e-10
        spectrum = observed / (stellar + epsilon)

        # Model prediction
        output = self.module(spectrum, atm_params)

        # Recover planet spectrum if requested
        planet = None
        if self.config.compute_planet:
            planet = spectrum / (output.telluric + epsilon)

        # Uncertainty estimation via MC Dropout
        uncertainty = None
        if self.config.compute_uncertainty:
            uncertainty = self._mc_dropout_uncertainty(spectrum, atm_params)

        return ModelOutput(
            telluric=output.telluric,
            planet=planet,
            uncertainty=uncertainty,
            latent=output.latent if self.config.return_latent else None,
        )

    def _mc_dropout_uncertainty(
        self,
        spectrum: torch.Tensor,
        atm_params: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate uncertainty via Monte Carlo Dropout.

        Args:
            spectrum: Preprocessed spectrum R. Shape: (B, N).
            atm_params: Atmospheric parameters. Shape: (B, N_params).

        Returns:
            Uncertainty estimate (standard deviation). Shape: (B, N).
        """
        # Enable dropout at inference time
        self.module.train()

        predictions = []
        for _ in range(self.config.mc_dropout_samples):
            output = self.module(spectrum, atm_params)
            predictions.append(output.telluric)

        # Restore eval mode
        self.module.eval()

        stacked = torch.stack(predictions, dim=0)  # (S, B, N)
        return stacked.std(dim=0)  # (B, N)

    @torch.no_grad()
    def predict_batch(
        self,
        spectra: npt.NDArray[np.float64] | torch.Tensor,
        atm_params: npt.NDArray[np.float64] | torch.Tensor,
    ) -> ModelOutput:
        """Run inference on pre-processed spectra (already R = X/S).

        Args:
            spectra: Pre-processed spectra R. Shape: (B, N).
            atm_params: Atmospheric parameters. Shape: (B, N_params).

        Returns:
            ModelOutput with predictions.
        """
        if isinstance(spectra, np.ndarray):
            spectra = torch.from_numpy(spectra.astype(np.float32))
        if isinstance(atm_params, np.ndarray):
            atm_params = torch.from_numpy(atm_params.astype(np.float32))

        spectra = spectra.to(self.device)
        atm_params = atm_params.to(self.device)

        return self.module(spectra, atm_params)
