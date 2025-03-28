# Copyright Axelera AI, 2024
# Custom exceptions for the Axelera AI app


class PrequantizedModelRequired(Exception):
    """Raised when no prequantized model is found in the specified directory,
    or we want to do a fresh quantize"""

    def __init__(self, model_name: str, directory: str):
        self.model_name = model_name
        self.directory = directory
        self.message = f"Prequantized model must be (re)generated for {model_name} in {directory}"
        super().__init__(self.message)
