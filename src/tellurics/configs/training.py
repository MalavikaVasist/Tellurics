"""Training configuration."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class OptimizerType(str, Enum):
    """Available optimizers."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Available learning rate schedulers."""

    COSINE = "cosine"
    STEP = "step"
    ONE_CYCLE = "one_cycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.9, ge=0.0, description="For SGD only")


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = Field(default=5, ge=0)
    min_lr: float = Field(default=1e-7, ge=0.0)
    step_size: int = Field(default=30, gt=0, description="For StepLR")
    gamma: float = Field(default=0.1, gt=0.0, lt=1.0, description="For StepLR")
    patience: int = Field(default=10, gt=0, description="For ReduceOnPlateau")


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    max_epochs: int = Field(default=100, gt=0)
    gradient_clip_val: float | None = Field(default=1.0, ge=0.0)
    accumulate_grad_batches: int = Field(default=1, gt=0)
    precision: str = Field(default="16-mixed")
    early_stopping_patience: int = Field(default=15, gt=0)
    early_stopping_metric: str = "val_loss"
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    log_every_n_steps: int = Field(default=10, gt=0)
    val_check_interval: float | int = 1.0
    seed: int = Field(default=42)
