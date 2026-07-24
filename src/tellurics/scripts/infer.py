"""Inference script entry point."""

import argparse
import json
from pathlib import Path

import numpy as np

from tellurics.configs.inference import InferenceConfig
from tellurics.inference.pipeline import InferencePipeline
from tellurics.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run inference from command line."""
    parser = argparse.ArgumentParser(description="Run telluric correction inference")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON inference configuration file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input data (numpy .npz file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        raw_config = json.load(f)
    config = InferenceConfig(**raw_config)

    # Load input data
    logger.info(f"Loading input data from {args.input}")
    data = np.load(args.input)
    observed = data["observed"]
    stellar = data["stellar"]
    atm_params = data["atm_params"]

    # Run inference
    logger.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(config)

    logger.info("Running predictions...")
    output = pipeline.predict(observed, stellar, atm_params)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {
        "telluric": output.telluric.cpu().numpy(),
    }
    if output.planet is not None:
        results["planet"] = output.planet.cpu().numpy()
    if output.uncertainty is not None:
        results["uncertainty"] = output.uncertainty.cpu().numpy()

    output_path = args.output / "predictions.npz"
    np.savez(output_path, **results)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
