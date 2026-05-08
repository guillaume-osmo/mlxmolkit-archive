# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Method-name → parameter-dict registry for xTB methods.

Mirrors :data:`mlxmolkit.rm1.methods.METHOD_PARAMS`. Currently exposes
GFN0-xTB only; GFN1 and GFN2 will land as Phase B and C.
"""

from __future__ import annotations

from .params_gfn0 import GFN0_PARAMS, GFN0ElementParams


XTB_METHOD_PARAMS: dict[str, dict[int, GFN0ElementParams]] = {
    "GFN0": GFN0_PARAMS,
}


def get_xtb_params(method: str = "GFN0") -> dict[int, GFN0ElementParams]:
    """Look up the per-element parameter dict for an xTB method.

    Args:
        method: ``"GFN0"`` (currently the only supported method).
            Case-insensitive; `"GFN0-xTB"`, `"gfn0"` etc. all map to GFN0.

    Returns:
        Element-keyed parameter dict.

    Raises:
        ValueError: if ``method`` is unknown.
    """
    key = method.upper().replace("-", "").replace("XTB", "").strip("_")
    if key not in XTB_METHOD_PARAMS:
        raise ValueError(
            f"Unknown xTB method {method!r}. Available: "
            f"{sorted(XTB_METHOD_PARAMS.keys())}"
        )
    return XTB_METHOD_PARAMS[key]
