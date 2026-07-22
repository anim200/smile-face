"""Application specific exceptions.

A small, explicit hierarchy lets the web layer translate failures into the
right HTTP response without inspecting exception messages.
"""


class SmileClassifierError(Exception):
    """Base class for every error raised by this application."""


class ModelNotAvailableError(SmileClassifierError):
    """No usable model is currently active in the registry."""


class InvalidModelArtifactError(SmileClassifierError):
    """A model file exists but failed structural or behavioural validation."""


class InsufficientTrainingDataError(SmileClassifierError):
    """Staged images do not satisfy the minimum required to fit a model."""


class TrainingInProgressError(SmileClassifierError):
    """A training run is already active; concurrent runs are not permitted."""


class InvalidImageError(SmileClassifierError):
    """An uploaded file could not be decoded as an image."""