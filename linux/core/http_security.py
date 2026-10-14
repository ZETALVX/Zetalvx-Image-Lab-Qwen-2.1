"""Pure validation helpers; forwarded addresses are deliberately not trusted.
Modified in Zetalvx Image Lab - Qwen 2.1 Edition 0.1.0.21. Apache-2.0.
"""
import ipaddress
from urllib.parse import urlsplit

def is_loopback_client(address):
    try:
        ip = ipaddress.ip_address(address or "")
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return ip.is_loopback
    except ValueError:
        return False

def same_origin(origin, host_url):
    # No Origin is permitted for local native clients; authenticated routes
    # still retain the application's existing scoped CSRF checks.
    if origin is None:
        return True
    try:
        a, b = urlsplit(origin), urlsplit(host_url)
        def identity(u):
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
                return None
            return u.scheme, u.hostname.lower(), u.port or (443 if u.scheme == "https" else 80)
        return (origin != "null" and a.path in ("", "/") and not a.query and not a.fragment
                and identity(a) is not None and identity(a) == identity(b))
    except (ValueError, AttributeError, TypeError):
        return False
