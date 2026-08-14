from __future__ import annotations

from setuptools import Extension, setup

try:
    import numpy as np
except Exception:  # pragma: no cover - setup-time guard
    np = None


ext_modules = []
if np is not None:
    ext_modules.append(
        Extension(
            "mlxmolkit.xtb._multipole_cpp",
            ["mlxmolkit/xtb/_multipole_cpp.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", "-DNDEBUG"],
        )
    )
    ext_modules.append(
        Extension(
            "mlxmolkit.xtb._gxtb_cpp",
            ["mlxmolkit/xtb/_gxtb_cpp.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", "-DNDEBUG"],
        )
    )


setup(ext_modules=ext_modules)
