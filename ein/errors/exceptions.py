class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message: str):
        super().__init__(message)


class NotFoundError(Exception):
    """Exception raised when a required resource is not found."""

    def __init__(self, resource_name: str):
        super().__init__(f"{resource_name} not found.")
