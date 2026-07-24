"""Loss function configuration."""

from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    """Configuration for loss functions and their weights."""

    # Loss weights
    lambda_telluric: float = Field(default=1.0, ge=0.0)
    lambda_planet: float = Field(default=0.1, ge=0.0)
    lambda_smoothness: float = Field(default=0.01, ge=0.0)
    lambda_physical: float = Field(default=0.1, ge=0.0)

    # Physical constraint parameters
    enforce_bounds: bool = True
    bounds_penalty_weight: float = Field(default=1.0, ge=0.0)

    # Smoothness parameters
    smoothness_order: int = Field(default=2, ge=1, le=3)

    # Real data loss
    use_real_regularization: bool = True
    lambda_real_smoothness: float = Field(default=0.01, ge=0.0)
    lambda_real_physical: float = Field(default=0.1, ge=0.0)
