"""flock-compatible boundary for the lock patterns used by this application.
POSIX uses the real fcntl. Windows uses byte-zero msvcrt advisory locking;
shared locks are intentionally exclusive there. No cross-host guarantee.
Modified in Zetalvx Image Lab - Qwen 2.1 Edition 0.1.0.21. Apache-2.0.
"""
import os, errno, time
if os.name != "nt":
    from fcntl import flock, LOCK_EX, LOCK_SH, LOCK_UN, LOCK_NB
else:
    import msvcrt
    LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8
    def flock(file, operation):
        fd = file if isinstance(file, int) else file.fileno()
        old = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if operation & LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return
            if not operation & (LOCK_EX | LOCK_SH):
                raise ValueError("Unsupported lock operation")
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    return
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if operation & LOCK_NB:
                        raise BlockingIOError(errno.EAGAIN, "Lock is held") from exc
                    time.sleep(0.05)
        finally:
            os.lseek(fd, old, os.SEEK_SET)
