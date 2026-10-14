"""Small explicit Linux/Windows boundary. Native Windows validation is still required.
Modified in Zetalvx Image Lab - Qwen 2.1 Edition 0.1.0.21. Apache-2.0.
"""
from __future__ import annotations
import os, signal, subprocess
from pathlib import Path

def venv_python(root, *, windows=None):
    win = os.name == "nt" if windows is None else windows
    return Path(root) / ("Scripts/python.exe" if win else "bin/python")

def default_data_root():
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()/"AppData/Local"))) / "ImageLabQwen21"
    return Path.home()/".local/share/ImageLabQwen21"

def process_start(pid):
    try:
        if os.name != "nt":
            return Path(f"/proc/{int(pid)}/stat").read_text().split(") ", 1)[1].split()[19]
        # Management Python can be a clean, dependency-free interpreter. Use the
        # OS creation identity directly rather than importing psutil here.
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        kernel.GetProcessTimes.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME))
        kernel.GetProcessTimes.restype=wintypes.BOOL
        handle=kernel.OpenProcess(0x1000,False,int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:return None
        try:
            created,ended,kernel_time,user_time=(wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(ended),ctypes.byref(kernel_time),ctypes.byref(user_time)):
                return None
            ticks=(created.dwHighDateTime<<32)|created.dwLowDateTime
            return 'win-filetime-'+str(ticks)
        finally:kernel.CloseHandle(handle)
    except (OSError, ValueError, IndexError):
        return None
    except Exception:
        # psutil.NoSuchProcess / AccessDenied must never make a stale PID valid.
        return None

def hidden_kwargs(*, windows=None):
    """No console window for helper processes. Does not hide errors or discard logs."""
    win = os.name == "nt" if windows is None else windows
    return {"creationflags": 0x08000000} if win else {}  # CREATE_NO_WINDOW

def detached_kwargs(*, windows=None):
    win = os.name == "nt" if windows is None else windows
    if win:
        # Do not combine with DETACHED_PROCESS: it disables CREATE_NO_WINDOW.
        # This also covers the Windows venv redirector, not only pythonw.exe.
        return {"creationflags": 0x08000000 | 0x00000200}
    return {"start_new_session": True}

def terminate_owned_tree(proc, timeout=4):
    """Only for an owned Popen. Never find or terminate processes by executable name."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        import psutil
        try:
            parent=psutil.Process(proc.pid)
            children=parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return
        for child in reversed(children):
            try: child.terminate()
            except psutil.NoSuchProcess: pass
        try: parent.terminate()
        except psutil.NoSuchProcess: pass
        _, alive=psutil.wait_procs(children+[parent], timeout=timeout)
        for p in alive:
            try: p.kill()
            except psutil.NoSuchProcess: pass
        proc.wait(timeout=timeout)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=timeout)
