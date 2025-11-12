"""
Custom exceptions for the compression pipeline.

This module defines specific exception types for different error conditions
that can occur during compression and decompression operations.
"""


class CompressionPipelineError(Exception):
    """Base exception for all compression pipeline errors."""
    pass


class DataValidationError(CompressionPipelineError):
    """Raised when input data validation fails."""
    
    def __init__(self, message: str, data_info: dict = None):
        super().__init__(message)
        self.data_info = data_info or {}


class CompressionError(CompressionPipelineError):
    """Raised when compression operation fails."""
    
    def __init__(self, message: str, stage: str = None, original_error: Exception = None):
        super().__init__(message)
        self.stage = stage
        self.original_error = original_error


class DecompressionError(CompressionPipelineError):
    """Raised when decompression operation fails."""
    
    def __init__(self, message: str, stage: str = None, original_error: Exception = None):
        super().__init__(message)
        self.stage = stage
        self.original_error = original_error


class CorruptedDataError(DecompressionError):
    """Raised when compressed data appears to be corrupted."""
    
    def __init__(self, message: str, corruption_type: str = None):
        super().__init__(message, stage="data_validation")
        self.corruption_type = corruption_type


class InsufficientResourcesError(CompressionPipelineError):
    """Raised when system resources are insufficient for operation."""
    
    def __init__(self, message: str, resource_type: str = None, required: str = None, available: str = None):
        super().__init__(message)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class ConfigurationError(CompressionPipelineError):
    """Raised when pipeline configuration is invalid."""
    
    def __init__(self, message: str, parameter: str = None, value: str = None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class UnsupportedDataTypeError(DataValidationError):
    """Raised when data type is not supported by the pipeline."""
    
    def __init__(self, data_type: str, supported_types: list = None):
        message = f"Unsupported data type: {data_type}"
        if supported_types:
            message += f". Supported types: {', '.join(supported_types)}"
        super().__init__(message)
        self.data_type = data_type
        self.supported_types = supported_types or []