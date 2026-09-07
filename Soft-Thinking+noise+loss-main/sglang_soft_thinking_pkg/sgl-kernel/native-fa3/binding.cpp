// Thin binding around the exact upstream implementation. No kernel arithmetic
// is copied or changed here. Both actor and rollout call this same forward.
#include <torch/extension.h>
#include "flash_api.cpp"

using Tensor = at::Tensor;
using MaybeTensor = std::optional<Tensor>;

auto opd_fwd(
    Tensor q, Tensor k, Tensor v, MaybeTensor k_new, MaybeTensor v_new,
    MaybeTensor q_v, MaybeTensor out, MaybeTensor cu_q, MaybeTensor cu_k,
    MaybeTensor cu_new, MaybeTensor used_q, MaybeTensor used_k,
    std::optional<int64_t> max_q, std::optional<int64_t> max_k,
    MaybeTensor page_table, MaybeTensor batch_idx, MaybeTensor leftpad,
    MaybeTensor rotary_cos, MaybeTensor rotary_sin, MaybeTensor rotary_seqlens,
    MaybeTensor q_descale, MaybeTensor k_descale, MaybeTensor v_descale,
    double scale, bool causal, int64_t left, int64_t right, double softcap,
    bool rotary_interleaved, MaybeTensor scheduler, int64_t splits,
    std::optional<bool> pack_gqa, int64_t sm_margin) {
    std::optional<const Tensor> sinks = std::nullopt;
    return mha_fwd(
        q, k, v, k_new, v_new, q_v, out, cu_q, cu_k, cu_new, used_q, used_k,
        max_q, max_k, page_table, batch_idx, leftpad, rotary_cos, rotary_sin,
        rotary_seqlens, q_descale, k_descale, v_descale, scale, causal,
        left, right, 0, softcap, rotary_interleaved, scheduler, splits,
        pack_gqa, sm_margin, sinks, std::nullopt, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &opd_fwd);
    m.def("bwd", &mha_bwd);
    m.def("build_contract", []() {
        pybind11::dict result;
        result["sgl_attn_commit"] = "f89bc2306632d1ec5f97b014dded4254f5b4a907";
        result["cutlass_commit"] = "7127592069c2fe01b041e174ba4345ef9b279671";
        result["nvcc_version"] = "12.6.85";
        result["ptxas_version"] = "12.8.93";
        result["architecture"] = "sm_90a";
        result["dtype"] = "bfloat16";
        result["head_dimension"] = 128;
        result["native_backward"] = true;
        return result;
    });
}
