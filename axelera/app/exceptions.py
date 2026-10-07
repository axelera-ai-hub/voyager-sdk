# Copyright Axelera AI, 2024
# Custom exceptions for the Axelera AI app

import inspect


class PrequantizedModelRequired(Exception):
    """Raised when no prequantized model is found in the specified directory,
    or we want to do a fresh quantize"""

    def __init__(self, model_name: str, directory: str):
        self.model_name = model_name
        self.directory = directory
        self.message = f"Prequantized model must be (re)generated for {model_name} in {directory}"
        super().__init__(self.message)


class NotSupportedForTask(Exception):
    """Raised when a method is not supported for a specific task"""

    def __init__(self, class_name: str, method_name: str):
        frame = inspect.currentframe().f_back
        method_name = frame.f_code.co_name
        class_name = frame.f_locals['self'].__class__.__name__
        self.message = f"{class_name} doesn't support {method_name} method"
        super().__init__(self.message)
