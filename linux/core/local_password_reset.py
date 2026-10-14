"""Host-only interactive account recovery (no HTTP endpoints; Apache-2.0).

Run through the installed user's command dispatcher. The account's username,
preferences and session lifetime remain unchanged. Flask validates the password
hash stamp on every authenticated request, so changing the salted hash revokes
previous web sessions without restarting generation workers.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.first_run_security import FirstRunCodes
from core.private_files import atomic_private_json, is_link, protect


def reset_password(auth_path: Path, secrets_dir: Path, new_password: str, confirmation: str, *, hasher=None) -> str:
    """Reset an existing account under the shared first-run/account lock.

    Passwords must be supplied only by the LOCAL host's interactive terminal,
    never in argv, HTTP requests or installer logs. `hasher` is injectable only
    for hermetic unit tests; production uses Werkzeug's salted password hash.
    """
    if not isinstance(new_password, str) or not 10 <= len(new_password) <= 1024:
        raise ValueError('Password must contain 10–1024 characters.')
    if new_password != confirmation:
        raise ValueError('The passwords do not match. Nothing changed.')
    auth_path = Path(auth_path)
    manager = FirstRunCodes(auth_path, secrets_dir)
    with manager.locked():
        if is_link(auth_path) or not auth_path.is_file():
            raise RuntimeError('Existing account file is missing or linked; no reset performed.')
        protect(auth_path)
        try:
            account = json.loads(auth_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError('Cannot read the existing account; no reset performed.') from exc
        if not isinstance(account, dict) or not isinstance(account.get('password_hash'), str) or not account['password_hash']:
            raise RuntimeError('No account has been created yet. Use the first-run setup procedure.')
        username = account.get('username')
        if not isinstance(username, str) or not username.strip():
            raise RuntimeError('Existing account has an invalid username; refusing to overwrite it.')
        if hasher is None:
            from werkzeug.security import generate_password_hash
            hasher = generate_password_hash
        replacement = hasher(new_password)
        if not isinstance(replacement, str) or not replacement or replacement == account['password_hash']:
            raise RuntimeError('Password hashing failed; the account was not modified.')
        # Preserve username, enabled flag, session lifetime and future keys.
        updated = {**account, 'password_hash': replacement}
        atomic_private_json(auth_path, updated)
        # Bootstrap codes are already invalid as soon as the account exists.
        # Remove any orphaned pre-registration state without touching data.
        try:
            manager.revoke_locked()
        except OSError:
            pass  # Account hash is already atomically updated; login sessions are revoked.
    return username
