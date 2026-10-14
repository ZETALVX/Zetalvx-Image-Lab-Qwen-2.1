"""Portable untrusted download paths. No drive paths, ADS, links or special files.
Zetalvx 0.1.0.42, Apache-2.0. Does not claim protection from a malicious local owner.
"""
from __future__ import annotations
import re, stat
from pathlib import Path

_RESERVED=re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)

def portable_relative(name: str) -> Path:
    if not isinstance(name,str) or not name or len(name)>2048:
        raise ValueError('Invalid download path')
    if any(ord(c)<32 or c in '\\:<>"|?*' for c in name):
        raise ValueError('Non-portable download path')
    parts=name.split('/')
    for part in parts:
        if not part or part in ('.','..') or len(part)>180 or part.endswith((' ','.')) or _RESERVED.match(part):
            raise ValueError('Invalid download path component')
    return Path(*parts)

def is_link(path: Path) -> bool:
    try:
        s=path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(s.st_mode) or bool(getattr(s,'st_file_attributes',0)&0x400)

def safe_child(root: Path, name: str) -> Path:
    root=Path(root).absolute(); rel=portable_relative(name)
    cursor=root
    if is_link(cursor):raise ValueError('Download root cannot be a link or reparse point')
    for part in rel.parts:
        cursor=cursor/part
        if is_link(cursor):raise ValueError('Download destination cannot traverse a link or reparse point')
    resolved_root=root.resolve()
    if not cursor.resolve().is_relative_to(resolved_root) or cursor.resolve()==resolved_root:
        raise ValueError('Download destination escapes its root')
    return cursor
