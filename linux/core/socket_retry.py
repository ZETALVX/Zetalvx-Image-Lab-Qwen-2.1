"""Small cross-platform retry helper for transient socket bind collisions."""
from __future__ import annotations
import errno
import time


def is_address_in_use(error: BaseException) -> bool:
    """True only for the OS address-already-in-use condition."""
    return (
        getattr(error, "errno", None) in (errno.EADDRINUSE, 10048)
        or getattr(error, "winerror", None) == 10048
    )


def retry_address_in_use(operation, *, attempts=10, sleeper=time.sleep, on_retry=None):
    """Retry *operation* only when binding loses a port to EADDRINUSE.

    Other socket/TLS/application errors remain fatal. The caller owns cleanup of
    a failed attempt before raising the bind error.
    """
    attempts = max(1, int(attempts))
    for index in range(attempts):
        try:
            return operation(index)
        except OSError as error:
            if not is_address_in_use(error) or index + 1 >= attempts:
                raise
            delay = min(0.15 * (index + 1), 1.0)
            if on_retry is not None:
                on_retry(index + 1, attempts, error, delay)
            sleeper(delay)
    raise AssertionError("unreachable")
