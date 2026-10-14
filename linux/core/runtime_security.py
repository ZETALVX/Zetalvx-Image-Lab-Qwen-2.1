"""Fail-closed PyTorch baseline for this reviewed candidate, not a CVE certificate.
Modified in Zetalvx Image Lab - Qwen 2.1 Edition 0.1.0.21. Apache-2.0; see LICENSE and NOTICE.
"""
from __future__ import annotations
from importlib import metadata
import re
MIN_TORCH = (2, 14, 0)
ADVISORY = "https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p"

def reviewed_torch_version(version: str) -> bool:
    # Only final releases. Accept CPU/CUDA local build tags, not rc/dev/nightly.
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[a-zA-Z0-9._-]+)?", str(version).strip())
    return bool(m and tuple(map(int, m.groups())) >= MIN_TORCH)

def require_reviewed_torch(version: str | None = None) -> str:
    if version is None:
        try:
            version = metadata.version("torch")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("PyTorch is missing in this runtime. Prepare the separate AI candidate runtime.") from exc
    if not reviewed_torch_version(version):
        raise RuntimeError(
            f"Blocked PyTorch runtime {version!r}: a final release >=2.14.0 is required "
            "for the reviewed checkpoint-loader advisory. Unknown/prerelease versions are refused. "
            "Do not upgrade the old working environment in place; prepare and audit a separate runtime. "
            + ADVISORY)
    return version
