class AppError(Exception):
    """Base application exception."""


class ValidationError(AppError):
    """Raised for invalid user input."""


class UnsupportedFileTypeError(AppError):
    """Raised when an unsupported file type is uploaded."""


class DocumentNotFoundError(AppError):
    """Raised when a document cannot be found."""


class LLMServiceError(AppError):
    """Raised when the configured LLM provider fails."""
