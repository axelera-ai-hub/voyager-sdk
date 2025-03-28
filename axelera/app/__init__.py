# Copyright Axelera AI, 2024
import warnings

warnings.filterwarnings(
    "ignore", module="torchvision", message="Failed to load image Python extension"
)
warnings.filterwarnings(
    "ignore",
    module="pytools",
    message="Unable to import recommended hash",
)

try:
    import onnxruntime  # noqa - prevents a crash in the compiler
except ImportError:
    pass
