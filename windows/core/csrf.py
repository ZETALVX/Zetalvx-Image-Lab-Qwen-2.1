"""Scoped CSRF tokens without concurrent Set-Cookie writes from API GETs."""
import hashlib,hmac,json
from flask import current_app,session

def token(scope):
    key=current_app.secret_key
    if isinstance(key,str):key=key.encode()
    # New logins have an independent nonce. Existing signed sessions continue to
    # work after upgrading; no random value is set lazily from parallel requests.
    material=json.dumps(['zetalvx-csrf-v1',scope,session.get('creator_username',''),
        session.get('auth_stamp',''),session.get('creator_csrf_nonce','legacy')],separators=(',',':')).encode()
    return hmac.new(key,material,hashlib.sha256).hexdigest()

def valid(scope,value):return bool(value) and hmac.compare_digest(str(value),token(scope))
