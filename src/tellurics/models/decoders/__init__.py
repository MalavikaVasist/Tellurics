"""Per-exposure decoders: map exposure codes to predictions.

* :class:`ParamDecoder`   - ``(..., d) -> (..., P)`` per-exposure telluric /
  atmospheric *parameter* regression (the whole-night estimator head).
* :class:`TelluricDecoder` - ``(..., d) -> (..., N)`` per-exposure telluric
  *transmission* reconstruction (spectral head, available for models that
  regress the full spectrum instead of parameters).
"""

from tellurics.models.decoders.parameter import ParamDecoder
from tellurics.models.decoders.telluric import TelluricDecoder

__all__ = ["ParamDecoder", "TelluricDecoder"]
