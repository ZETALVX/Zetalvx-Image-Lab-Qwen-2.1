"""Private files on POSIX and Windows Python 3.11+ (no optional dependencies).
Protected Windows DACL grants only current user, SYSTEM and Administrators.
Files remain plaintext; local administrators/the OS are outside this boundary.
"""
from __future__ import annotations
import json, os, stat, tempfile
from pathlib import Path
from core.safe_paths import is_link


def _windows_dacl(path: Path, directory: bool):
    import ctypes
    from ctypes import wintypes as w
    adv=ctypes.WinDLL('advapi32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    adv.OpenProcessToken.argtypes=(w.HANDLE,w.DWORD,ctypes.POINTER(w.HANDLE));adv.OpenProcessToken.restype=w.BOOL
    adv.GetTokenInformation.argtypes=(w.HANDLE,ctypes.c_int,w.LPVOID,w.DWORD,ctypes.POINTER(w.DWORD));adv.GetTokenInformation.restype=w.BOOL
    adv.ConvertSidToStringSidW.argtypes=(w.LPVOID,ctypes.POINTER(w.LPWSTR));adv.ConvertSidToStringSidW.restype=w.BOOL
    adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes=(w.LPCWSTR,w.DWORD,ctypes.POINTER(w.LPVOID),ctypes.POINTER(w.DWORD));adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype=w.BOOL
    adv.SetFileSecurityW.argtypes=(w.LPCWSTR,w.DWORD,w.LPVOID);adv.SetFileSecurityW.restype=w.BOOL
    kernel.GetCurrentProcess.restype=w.HANDLE
    kernel.CloseHandle.argtypes=(w.HANDLE,);kernel.LocalFree.argtypes=(w.LPVOID,);kernel.LocalFree.restype=w.LPVOID
    token=w.HANDLE();sid_text=w.LPWSTR();sd=w.LPVOID()
    def check(ok):
        if not ok:raise ctypes.WinError(ctypes.get_last_error())
    try:
        check(adv.OpenProcessToken(kernel.GetCurrentProcess(),0x0008,ctypes.byref(token)))
        length=w.DWORD()
        adv.GetTokenInformation(token,1,None,0,ctypes.byref(length))
        if not length.value:raise OSError('Cannot determine Windows user SID')
        buf=ctypes.create_string_buffer(length.value)
        check(adv.GetTokenInformation(token,1,buf,length,ctypes.byref(length)))
        sid=ctypes.cast(buf,ctypes.POINTER(w.LPVOID))[0]
        check(adv.ConvertSidToStringSidW(sid,ctypes.byref(sid_text)))
        inherit='OICI' if directory else ''
        sddl='D:P'+''.join('(A;'+inherit+';FA;;;'+principal+')' for principal in (sid_text.value,'SY','BA'))
        check(adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl,1,ctypes.byref(sd),None))
        check(adv.SetFileSecurityW(str(path),0x00000004|0x80000000,sd))
    finally:
        if sd:kernel.LocalFree(sd)
        if sid_text:kernel.LocalFree(ctypes.cast(sid_text,w.LPVOID))
        if token:kernel.CloseHandle(token)


def protect(path, *, directory=False, windows=None):
    path=Path(path).absolute()
    if is_link(path):raise OSError('Private storage cannot be a link or reparse point')
    s=path.stat()
    if directory and not stat.S_ISDIR(s.st_mode):raise OSError('Private directory expected')
    if not directory and not stat.S_ISREG(s.st_mode):raise OSError('Private regular file expected')
    if (os.name=='nt' if windows is None else windows):_windows_dacl(path,directory)
    else:os.chmod(path,0o700 if directory else 0o600,follow_symlinks=False)


def private_directory(path):
    path=Path(path)
    if is_link(path):raise OSError('Private storage cannot be a link or reparse point')
    path.mkdir(parents=True,exist_ok=True,mode=0o700)
    protect(path,directory=True)
    return path


def atomic_private_text(path,text):
    path=Path(path);private_directory(path.parent)
    if is_link(path):raise OSError('Private file cannot be a link or reparse point')
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'-',suffix='.tmp')
    try:
        # Apply ACL BEFORE writing any credential. Python 3.11 Windows has no fchmod.
        protect(tmp)
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            fd=None;f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if fd is not None:os.close(fd)
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_private_json(path,data):
    atomic_private_text(path,json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
