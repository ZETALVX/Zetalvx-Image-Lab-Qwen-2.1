"""Single-use, host-issued first-account code (local/LAN bootstrap only).

Apache-2.0. Added in 0.1.0.24. No new dependencies. The clear code is returned
only to the host launcher, never saved or served by an HTTP endpoint. All code
issuance and first-account commits share a thread + OS file lock. Not a general
login/password-reset mechanism; an existing account always disables bootstrap.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from core import file_lock
from core.http_security import is_loopback_client
from core.runtime_env import atomic_json

CODE_TTL_SECONDS = 20 * 60
MAX_ATTEMPTS = 10
ATTEMPT_WINDOW_SECONDS = 60
ALPHABET = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'  # 32 symbols, no 0/O or 1/I.
CODE_LENGTH = 24  # 120 random bits, displayed in six groups of four.
_THREAD_LOCK = threading.RLock()


class SetupCodeError(ValueError):
    def __init__(self, message: str, status: int = 403, retry_after: int = 0):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def normalize_code(value: object) -> str:
    if not isinstance(value, str) or len(value) > 128 or not value.isascii():
        return ''
    value = ''.join(c for c in value.upper() if c not in ' -\t\r\n')
    return value if len(value) == CODE_LENGTH and all(c in ALPHABET for c in value) else ''


def local_setup_request(remote_addr: str | None, host_url: str, headers) -> bool:
    """Local convenience path; never trust forwarding headers as identity.

    Forwarded/proxied requests must prove the code even when their peer is
    loopback. Direct LAN peers never qualify, irrespective of Host headers.
    """
    if not is_loopback_client(remote_addr):
        return False
    if any(k.lower() == 'forwarded' or k.lower().startswith('x-forwarded-')
           or k.lower() == 'x-real-ip' for k in headers.keys()):
        return False
    try:
        host = urlsplit(host_url).hostname
        return host == 'localhost' or is_loopback_client(host)
    except (ValueError, TypeError):
        return False


class FirstRunCodes:
    def __init__(self, auth_path: Path, secrets_dir: Path, clock=time.time):
        self.auth_path = Path(auth_path)
        self.secrets_dir = Path(secrets_dir)
        self.state_path = self.secrets_dir / 'first-run-code.json'
        self.lock_path = self.secrets_dir / 'first-run.lock'
        self.clock = clock

    @contextlib.contextmanager
    def locked(self):
        """Hold through verification AND successful account persistence."""
        with _THREAD_LOCK:
            self.secrets_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt':
                self.secrets_dir.chmod(0o700)
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(fd, 'r+b') as fp:
                # Windows byte-range locking needs a real byte in the file.
                if not self.lock_path.stat().st_size:
                    fp.write(b'\0'); fp.flush()
                file_lock.flock(fp.fileno(), file_lock.LOCK_EX)
                try:
                    yield self
                finally:
                    file_lock.flock(fp.fileno(), file_lock.LOCK_UN)

    def account_exists_locked(self) -> bool:
        # A malformed/unreadable auth file must never reopen first setup.
        try:
            cfg = json.loads(self.auth_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError('Cannot read the account configuration; first setup is disabled.') from exc
        if not isinstance(cfg, dict) or not isinstance(cfg.get('password_hash'), str):
            raise RuntimeError('Invalid account configuration; first setup is disabled.')
        return bool(cfg['password_hash'])

    def issue(self) -> str | None:
        with self.locked():
            if self.account_exists_locked():
                self.revoke_locked()
                return None
            raw = ''.join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))
            now = self.clock()
            atomic_json(self.state_path, {
                'schema': 1, 'purpose': 'first-account-only',
                'digest': hashlib.sha256(raw.encode('ascii')).hexdigest(),
                'issued_at': now, 'expires_at': now + CODE_TTL_SECONDS,
                'window_start': now, 'attempts': 0,
            })
            return '-'.join(raw[i:i+4] for i in range(0, CODE_LENGTH, 4))

    def matches_current(self, value: object) -> bool:
        """Return whether *value* is the current live code without consuming attempts.

        Used only to re-display the launcher-held code on the host-local first-run
        page. The clear code is still never written to disk.
        """
        raw = normalize_code(value)
        if not raw:
            return False
        with self.locked():
            if self.account_exists_locked():
                return False
            try:
                state = json.loads(self.state_path.read_text(encoding='utf-8'))
                if (state.get('schema') != 1 or state.get('purpose') != 'first-account-only'
                        or not isinstance(state.get('digest'), str) or len(state['digest']) != 64):
                    return False
                now = self.clock()
                issued = float(state['issued_at']); expires = float(state['expires_at'])
                if not all(math.isfinite(v) for v in (issued, expires)):
                    return False
            except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
                return False
            if now < issued or now >= expires or expires - issued > CODE_TTL_SECONDS:
                return False
            digest = hashlib.sha256(raw.encode('ascii')).hexdigest()
            return secrets.compare_digest(digest, state['digest'])

    def remaining_seconds(self, value: object) -> int:
        """Return whole seconds left for the current code, or 0 if it is stale.

        The clear code is supplied by the caller and is never recovered from disk.
        This is used only by the host-local first-setup page to keep its in-memory
        display synchronized with the hashed state file.
        """
        raw = normalize_code(value)
        if not raw:
            return 0
        with self.locked():
            if self.account_exists_locked():
                return 0
            try:
                state = json.loads(self.state_path.read_text(encoding='utf-8'))
                if (state.get('schema') != 1 or state.get('purpose') != 'first-account-only'
                        or not isinstance(state.get('digest'), str) or len(state['digest']) != 64):
                    return 0
                now = self.clock()
                issued = float(state['issued_at']); expires = float(state['expires_at'])
                if not all(math.isfinite(v) for v in (issued, expires)):
                    return 0
            except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
                return 0
            if now < issued or now >= expires or expires - issued > CODE_TTL_SECONDS:
                return 0
            digest = hashlib.sha256(raw.encode('ascii')).hexdigest()
            if not secrets.compare_digest(digest, state['digest']):
                return 0
            return max(1, int(math.ceil(expires - now)))

    def verify_locked(self, value: object) -> None:
        """Verify without consuming: commit account, then revoke in same lock.

        Invalid credentials do not consume a valid code. Attempts are globally
        bounded per code, independently of spoofable client IP headers.
        """
        if self.account_exists_locked():
            raise SetupCodeError('Account già configurato. Usa il login.', 409)
        try:
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if (state.get('schema') != 1 or state.get('purpose') != 'first-account-only'
                    or not isinstance(state.get('digest'), str) or len(state['digest']) != 64):
                raise ValueError('Invalid setup state')
            now = self.clock()
            issued = float(state['issued_at']); expires = float(state['expires_at'])
            window = float(state['window_start']); attempts = int(state['attempts'])
            if not all(math.isfinite(v) for v in (issued, expires, window)) or attempts < 0:
                raise ValueError('Invalid time/counter')
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            raise SetupCodeError('Codice non disponibile. Genera un nuovo codice dal computer che esegue l’app.') from None
        # Fail closed on a backwards clock adjustment, not just normal expiry.
        if now < issued or now >= expires or expires - issued > CODE_TTL_SECONDS:
            raise SetupCodeError('Codice scaduto. Genera un nuovo codice dal computer che esegue l’app.')
        if now - window >= ATTEMPT_WINDOW_SECONDS:
            window, attempts = now, 0
        if attempts >= MAX_ATTEMPTS:
            wait = max(1, math.ceil(ATTEMPT_WINDOW_SECONDS - (now - window)))
            raise SetupCodeError('Troppi tentativi. Attendi un minuto e riprova.', 429, wait)
        raw = normalize_code(value)
        digest = hashlib.sha256(raw.encode('ascii')).hexdigest()
        if not raw or not secrets.compare_digest(digest, state['digest']):
            state.update(window_start=window, attempts=attempts + 1)
            atomic_json(self.state_path, state)
            raise SetupCodeError('Codice di configurazione non valido.')

    def revoke_locked(self) -> None:
        self.state_path.unlink(missing_ok=True)


def print_setup_code(code: str | None) -> None:
    if not code:
        return
    print('\n=== PRIMA CONFIGURAZIONE / FIRST SETUP ===', flush=True)
    print('Codice / Code: ' + code, flush=True)
    print('Valido 20 minuti, solo per il primo account. Inseriscilo nel browser remoto.', flush=True)
    print('Valid 20 minutes, first account only. Enter it in the remote browser.', flush=True)
    print('Non condividere questo codice. / Do not share this code.', flush=True)
    print('Nuovo codice / New code: bash setup-code.sh (Linux) / SETUP-CODE.cmd (Windows)\n', flush=True)
