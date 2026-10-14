"""Small host-side dialogs. No remote endpoint may trigger these helpers."""
import os
import shutil
import subprocess
import sys

def graphical():
    return os.name == 'nt' or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

def dialog(message, *, question=False, title='Zetalvx Image Lab'):
    if os.name == 'nt':
        import ctypes
        # Default answer is No; never turn closing the dialog into consent.
        flags = (0x4 | 0x100 | 0x20) if question else 0x40
        return ctypes.windll.user32.MessageBoxW(None, str(message), title, flags | 0x10000) == (6 if question else 1)
    if graphical():
        if shutil.which('zenity'):
            args=['zenity', '--question' if question else '--info', '--title='+title, '--text='+str(message), '--width=460']
            if question: args += ['--default-cancel', '--ok-label=Yes / Sì', '--cancel-label=No']
        elif shutil.which('kdialog'):
            args=['kdialog','--title', title, '--yesno' if question else '--msgbox', str(message)]
        else: args=None
        if args:
            return subprocess.run(args, check=False).returncode == 0
    if sys.stdin and sys.stdin.isatty():
        if question:
            return input(str(message)+' [y/N]: ').strip().lower() in ('y','yes','s','si','sì')
        print(message); return True
    if sys.stderr: print(message, file=sys.stderr)
    return False
