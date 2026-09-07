"""Build only from authenticated external checkouts and exact compilers.

Use scripts/build-qwen-environment.sh from the parent source snapshot. This
setup never downloads source, changes another environment, or falls back to a
different compiler. A CUDA GPU is not needed to compile SM90a machine code.
"""

import os
from pathlib import Path
import re
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def checked_source(variable, commit):
    root = Path(os.environ[variable]).resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != commit or dirty:
        raise RuntimeError(f"{variable} must be a clean checkout of {commit}")
    return root


attention = checked_source("OPD_FA3_SOURCE_DIR", "f89bc2306632d1ec5f97b014dded4254f5b4a907")
cutlass = checked_source("OPD_CUTLASS_SOURCE_DIR", "7127592069c2fe01b041e174ba4345ef9b279671")
cuda = Path(os.environ["CUDA_HOME"]).resolve(strict=True)
assembler = Path(os.environ["OPD_PTXAS_PATH"]).resolve(strict=True)
for binary, version in ((cuda / "bin/nvcc", "12.6.85"), (assembler, "12.8.93")):
    observed = subprocess.check_output([str(binary), "--version"], text=True)
    if not re.search(r"\bV" + re.escape(version) + r"\b", observed):
        raise RuntimeError(f"wrong compiler {binary}: expected {version}")

# The private toolkit's bin/ptxas is the authenticated 12.8.93 assembler. NVCC
# resolves its companion assembler there; no system toolkit is modified.
if (cuda / "bin/ptxas").resolve() != assembler:
    raise RuntimeError("CUDA_HOME/bin/ptxas must be the pinned assembler")
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
os.environ.setdefault("MAX_JOBS", "4")

defines = [
    "FLASHATTENTION_DISABLE_SM8x", "FLASHATTENTION_DISABLE_FP16", "FLASHATTENTION_DISABLE_FP8",
    "FLASHATTENTION_DISABLE_HDIM64", "FLASHATTENTION_DISABLE_HDIM96",
    "FLASHATTENTION_DISABLE_HDIM192", "FLASHATTENTION_DISABLE_HDIM256",
    "FLASHATTENTION_DISABLE_HDIMDIFF64", "FLASHATTENTION_DISABLE_HDIMDIFF192",
    "FLASHATTENTION_DISABLE_SPLIT", "FLASHATTENTION_DISABLE_SOFTCAP",
    "FLASHATTENTION_DISABLE_LOCAL", "FLASHATTENTION_DISABLE_DROPOUT",
]
hopper = attention / "hopper"
sources = [
    str(Path(__file__).parent / "binding.cpp"),
    str(hopper / "flash_prepare_scheduler.cu"),
    str(hopper / "flash_fwd_combine.cu"),
    *[str(hopper / "instantiations" / name) for name in (
        "flash_fwd_hdim128_bf16_sm90.cu",
        "flash_fwd_hdim128_bf16_packgqa_sm90.cu",
        "flash_fwd_hdim128_bf16_paged_sm90.cu",
        "flash_bwd_hdim128_bf16_sm90.cu",
    )],
]
common = ["-O3", "-std=c++17", *["-D" + value for value in defines]]
setup(
    ext_modules=[CUDAExtension(
        "opd_fa3._C", sources,
        include_dirs=[str(hopper), str(cutlass / "include"), str(cutlass / "tools/util/include")],
        extra_compile_args={
            "cxx": common,
            "nvcc": [*common, "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                     "-DCUTE_USE_PACKED_TUPLE=1", "--threads=2"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
