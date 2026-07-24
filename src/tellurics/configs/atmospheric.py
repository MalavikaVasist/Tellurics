"""Atmospheric parameter configuration."""

from pydantic import BaseModel, Field


class AtmosphericParameters(BaseModel):
    """Represents atmospheric metadata for an observation.

    All fields are optional to support incomplete metadata from real observations.
    """

    airmass: float | None = Field(default=None, ge=1.0, le=10.0)
    humidity: float | None = Field(default=None, ge=0.0, le=100.0)
    pressure: float | None = Field(default=None, ge=0.0, description="Pressure in hPa")
    temperature: float | None = Field(default=None, description="Temperature in K")
    pwv: float | None = Field(default=None, ge=0.0, description="Precipitable water vapor in mm")
    seeing: float | None = Field(default=None, ge=0.0, description="Seeing in arcsec")

    def to_tensor_list(self) -> list[float]:
        """Convert parameters to a list of floats, replacing None with 0.0."""
        return [
            self.airmass or 0.0,
            self.humidity or 0.0,
            self.pressure or 0.0,
            self.temperature or 0.0,
            self.pwv or 0.0,
            self.seeing or 0.0,
        ]

    @property
    def num_parameters(self) -> int:
        """Number of atmospheric parameters."""
        return 6
