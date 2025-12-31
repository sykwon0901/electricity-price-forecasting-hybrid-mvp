"""Models."""

try:
    from .dl_hurdle_tcn import HurdleTCN  # noqa: F401
except Exception:
    # Keep the package importable even in environments without PyTorch.
    HurdleTCN = None  # type: ignore
