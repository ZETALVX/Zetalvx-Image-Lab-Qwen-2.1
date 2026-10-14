"""Bounded cleanup of unread request bodies after a WSGI response is decided.

Used only by Zetalvx Image Lab's Cheroot request subclass, not a global patch.
The application status/body/authentication decisions are never changed.

Cheroot 11.1.2 drains a Content-Length body before sending headers only on
keep-alive connections. A small early 401/403/429 or redirect with
Connection: close can therefore close TLS while the POST is still unread.
Here the same cleanup is bounded by both byte count and a total deadline,
including for close connections. Large/unknown bodies are NOT buffered or
fully drained, and cannot be reused as another request on a keep-alive socket.
"""
from __future__ import annotations

import time

MAX_DRAIN_BYTES = 64 * 1024
DRAIN_TIMEOUT = 1.0
READ_CHUNK = 8192


def _close_after_response(request) -> None:
    request.close_connection = True
    # A stale Keep-Alive header must not contradict the actual close decision.
    request.outheaders[:] = [
        (key, value) for key, value in request.outheaders
        if key.lower() != b'connection'
    ]
    request.outheaders.append((b'Connection', b'close'))


def finish_unread_body(request, *, max_bytes: int = MAX_DRAIN_BYTES,
                       timeout: float = DRAIN_TIMEOUT, clock=time.monotonic) -> str:
    """Discard only a small, length-delimited remaining body with a deadline.

    Called after the WSGI application has prepared its response. Does not
    consume a normal upload before its route handles it. It never parses
    credentials, trusts forwarding headers, changes a status or retries a POST.
    Returns a diagnostic code, never request content.
    """
    body = getattr(request, 'rfile', None)
    if getattr(request, 'chunked_read', False):
        # Chunk trailers/unknown lengths are left to the HTTP parser. Close
        # rather than interpreting leftover bytes as a subsequent request.
        if not getattr(body, 'closed', False):
            _close_after_response(request)
            return 'chunked-close'
        return 'already-read'

    remaining = getattr(body, 'remaining', None)
    if remaining is None:
        # No KnownLengthRFile: do not touch an unknown/framing stream.
        _close_after_response(request)
        return 'unknown-close'
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        _close_after_response(request)
        return 'invalid-close'
    if remaining == 0:
        return 'already-read'
    if remaining > max_bytes or timeout <= 0:
        _close_after_response(request)
        return 'limit-close'

    sock = request.conn.socket
    raw = getattr(body, 'rfile', None)
    read_once = getattr(raw, 'read1', None)
    # read1 makes at most one raw read, so a peer dribbling bytes cannot reset
    # the total deadline within a BufferedReader.read(n) loop. Fallback reads
    # exactly one byte through KnownLengthRFile for equivalent boundedness.
    use_read1 = callable(read_once) and raw is request.conn.rfile
    old_timeout = sock.gettimeout()
    deadline = clock() + timeout
    result = 'drained'
    try:
        while body.remaining:
            budget = deadline - clock()
            if budget <= 0:
                result = 'timeout-close'
                break
            read_timeout = budget if old_timeout is None else min(old_timeout, budget)
            sock.settimeout(read_timeout)
            if use_read1:
                data = read_once(min(READ_CHUNK, body.remaining))
                if len(data) > body.remaining:
                    result = 'invalid-close'
                    break
                body.remaining -= len(data)
            else:
                data = body.read(1)
            if not data:
                result = 'eof-close'
                break
    except (OSError, ValueError):
        # Restore the response path, not success for an invalid request. Close
        # prevents Cheroot's keep-alive drain from attempting another long read.
        result = 'read-error-close'
    finally:
        try:
            sock.settimeout(old_timeout)
        except OSError:
            result = 'socket-closed'

    if result != 'drained' or body.remaining:
        _close_after_response(request)
    return result
