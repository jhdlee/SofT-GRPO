"""Locate CUDA library headers in the same isolated interpreter as PyTorch."""

from pathlib import Path
import sysconfig


def nvidia_library_paths(site_packages=None):
    # The private NVCC toolkit intentionally contains only compiler, cudart,
    # and CCCL artifacts. ATen's CUDAContextLight.h additionally includes these
    # headers, provided by the complete lock's NVIDIA dependency wheels.
    site = Path(site_packages or sysconfig.get_path("purelib")).resolve(strict=True)
    includes, libraries = [], []
    for package, headers in (
        ("cublas", ("cublas_v2.h", "cublasLt.h")),
        ("cusparse", ("cusparse.h",)),
        ("cusolver", ("cusolverDn.h",)),
    ):
        include = (site / "nvidia" / package / "include").resolve()
        library = (site / "nvidia" / package / "lib").resolve()
        if not include.is_relative_to(site) or not library.is_relative_to(site):
            raise RuntimeError(f"NVIDIA {package} build paths escape the isolated environment")
        for header in headers:
            path = include / header
            if not path.is_file() or not path.resolve().is_relative_to(site):
                raise RuntimeError(f"pinned NVIDIA {package} wheel is missing its local header: {header}")
        payloads = list(library.glob(f"lib{package}*.so*"))
        if (not library.is_dir() or not payloads
                or any(not path.is_file() or not path.resolve().is_relative_to(site) for path in payloads)):
            raise RuntimeError(f"pinned NVIDIA {package} wheel is missing its library directory")
        includes.append(str(include))
        libraries.append(str(library))
    return includes, libraries
