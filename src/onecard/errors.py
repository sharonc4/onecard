class OneCardError(Exception):
    """Base for every error onecard raises deliberately."""


class ConfigError(OneCardError):
    """The configuration is invalid. Raised at validate time, never mid-request."""


class BudgetError(OneCardError):
    """A claim cannot fit within the declared VRAM budget."""


class BackendError(OneCardError):
    """A backend was unreachable or returned an error."""


class EvictionError(OneCardError):
    """A consumer failed to release the GPU when asked."""
