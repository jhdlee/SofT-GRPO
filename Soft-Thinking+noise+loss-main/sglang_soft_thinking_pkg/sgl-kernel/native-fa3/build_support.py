"""Authenticate host-only FA3 repair inputs and isolated CUDA library headers."""

import hashlib
from pathlib import Path
import sysconfig


HOST_BACKWARD_REPAIR = "scheduler_semaphore_and_explicit_gradients_v1"
UPSTREAM_FLASH_API_SHA256 = "35f2f6f5db472886219c7391a8c4d73ef619db4c8ef97ba23a914463ccf27f15"
PATCHED_FLASH_API_SHA256 = "e989b32ee79bb31429c45be2900050e8ce45a5985601360fbc25e9cc418a5767"
_MISSING_SEMAPHORE = (
    b"    // auto tile_count_semaphore = (params.is_causal || params.is_local) ? torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));\n"
    b"    // params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();"
)
_INITIALIZED_SEMAPHORE = (
    b"    // OPD host repair: retain the scheduler workspace through the launch.\n"
    b"    at::Tensor tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));\n"
    b"    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();"
)


def patched_flash_api(source):
    """Restore one missing host allocation in the exact pinned upstream file."""
    if hashlib.sha256(source).hexdigest() != UPSTREAM_FLASH_API_SHA256:
        raise RuntimeError("native FA3 upstream flash_api.cpp hash differs")
    if source.count(_MISSING_SEMAPHORE) != 1:
        raise RuntimeError("native FA3 semaphore repair requires exactly one matching block")
    result = source.replace(_MISSING_SEMAPHORE, _INITIALIZED_SEMAPHORE)
    if hashlib.sha256(result).hexdigest() != PATCHED_FLASH_API_SHA256:
        raise RuntimeError("native FA3 generated flash_api.cpp hash differs")
    return result


def prepare_host_source(attention_root, output_directory):
    """Generate a hash-bound include without editing the pinned source checkout."""
    attention_root = Path(attention_root).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    if output_directory.is_relative_to(attention_root):
        raise RuntimeError("native FA3 generated host source must be outside the upstream checkout")
    upstream = attention_root / "hopper/flash_api.cpp"
    if upstream.is_symlink() or not upstream.is_file():
        raise RuntimeError("native FA3 upstream flash_api.cpp must be a regular file")
    content = patched_flash_api(upstream.read_bytes())
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / "opd_flash_api.cpp"
    if output.is_symlink():
        raise RuntimeError("native FA3 generated host source cannot be a symlink")
    if output.exists():
        if not output.is_file() or output.read_bytes() != content:
            raise RuntimeError("native FA3 generated host source differs from the authenticated repair")
    else:
        with output.open("xb") as stream:
            stream.write(content)
    return output


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
