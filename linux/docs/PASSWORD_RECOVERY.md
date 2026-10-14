# Password reset in a manual source checkout

See the **Local account password recovery** section of `../../docs/MANUAL_SETUP.md`. There is no installer, `imagelab-qwen` command, reset shell script or uninstall tool in this source distribution. The reset is a host-only call to `core.local_password_reset.reset_password` from the manually activated venv.
