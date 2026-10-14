# Security considerations (manual source distribution)

Run Zetalvx Image Lab as a regular, unprivileged user in a manually created Python virtual environment. Do not expose the app or image worker directly to the public Internet. The web service defaults to self-signed local HTTPS on 127.0.0.1:8298; the GPU worker listens on 127.0.0.1:8299 and requires its internal authentication token. Use a trusted private LAN and an appropriate firewall if enabling remote browser access.

Store credentials, first-run codes, TLS keys, model weights, user projects and generated images only in the selected source data directory (QWEN_SOURCE_HOME). Never commit them. Review diagnostics and logs before sharing; they may contain paths or sensitive filenames.

The GitHub source checkout contains **no automatic application updater, installer, rollback service or uninstaller**. Review Git changes and third-party dependency updates yourself, maintain an independent venv, and back up your data before changing your environment. File integrity checks do not amount to signing or a safety certification.

Model weights and plugins are third-party content; read their security and license terms. Optional separately installed ComfyUI is user-controlled and may execute third-party code.
