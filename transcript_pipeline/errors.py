class PipelineError(RuntimeError):
    """Base exception for recoverable pipeline failures."""


class DependencyError(PipelineError):
    """A required external tool or Python package is missing."""


class DownloadError(PipelineError):
    """Audio download or source validation failed."""


class ChunkingError(PipelineError):
    """Audio inspection or chunk preparation failed."""


class ASRError(PipelineError):
    """Speech recognition failed."""


class StateError(PipelineError):
    """The saved job cannot be resumed safely."""


class ExportError(PipelineError):
    """Writing transcript outputs failed."""
