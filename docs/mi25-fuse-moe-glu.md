# Fused MoE decode: MUL_MAT_ID gate + up + GLU in one mat-vec-id dispatch (mi25 fork)

## Why
Qwen3.6 35B-A3B decode on the WX 9100 (gfx900, Windows AMD driver) runs each GPU-resident MoE layer as:
`MUL_MAT_ID gate` (one dispatch per token), `MUL_MAT_ID up` (one per token), `GLU swiglu` (one), then
`MUL_MAT_ID_MUL down` (already fused with the router weights). Per decode token that is 48 expert gate/up dispatches,
24 GLU dispatches, and a graph-dependency barrier between `up` and `GLU` in each of the 24 GPU MoE layers.
The barrier diagnostic (`GGML_VK_DIAG_NO_BARRIERS`) puts the GPU-side cost of the 879 barriers per token at
7–10 ms (8–11 µs each). Fusing gate + up + GLU removes 24 dispatches and 24 barriers per token, and the expert
input is read once instead of twice. Upstream fuses this only in CUDA MMQ (c49edf096, f5222f25c); the Vulkan
fusion list has nothing equivalent.

## What
A fused path for the node sequence `MUL_MAT_ID (gate)`, `MUL_MAT_ID (up)`, `GLU` at `node_idx`, `+1`, `+2`, taken
when **all** of these hold:
- `gate = nodes[i]` and `up = nodes[i+1]` are `GGML_OP_MUL_MAT_ID` with the same `src[1]` (input) and the same
  `src[2]` (ids); their `src[0]` have the same type, the same `ne` and `nb`, and are quantized, F16 or F32;
- `ggml_vk_use_mul_mat_vec_id(gate)` is true, i.e. `ids->ne[1] <= GGML_VK_MMVID_MAX_TOKENS` (default 8);
- `glu = nodes[i+2]` is `GGML_OP_GLU` with op `GGML_GLU_OP_SWIGLU` (required) or `GGML_GLU_OP_GEGLU` (optional),
  split form (`src[0] == gate`, `src[1] == up`), not swapped, F32 in and out, and its output has the shape of `gate`;
- `gate` and `up` have no other consumers (`ggml_can_fuse` / `ggml_vk_can_fuse` style check: the intermediates are
  not graph outputs and are used only by `glu`);
- gate and up weights live in the same `vk_buffer`, and the element offset between them is a whole number of
  blocks (`(up_offset - gate_offset) % ggml_type_size(type) == 0`). Otherwise fall back unfused;
- `device->fuse_moe_glu` is true. It defaults to true when `device->architecture == AMD_GCN` and false elsewhere.
  `GGML_VK_FUSE_MOE_GLU=0` forces it off (A/B tests), and `=1` turns it on for other architectures;
- fusion is not disabled (`device->disable_fusion`).

Everything else (prompt batches above the threshold, non-split GLU, other GLU ops, other devices) is unchanged.

## Design
- **Shader.** A `MUL_MAT_ID_GLU` define in the mat-vec shaders (`mul_mat_vec_base.glsl`, `mul_mat_vec_iface.glsl`, and
  each `mul_mat_vec_*.comp` used by the types below, plus `mul_mat_vecq.comp` when `quantize_y`). The gate weights
  are read through binding 0 at `a_offset` and the up weights through the same binding at `a_offset + p.up_offset`
  (elements of `A_TYPE`), so no type-specific load code changes. Each thread accumulates gate and up sums
  (`temp_g`, `temp_u`), both are reduced, and the writing lane stores
  `D = silu(g) * u` (SWIGLU: `g / (1 + exp(-g)) * u`) or `D = gelu(g) * u` (GEGLU, same formula as `glu.comp`).
  `expert_skip` (id -1) writes 0, as today.
- **Push constants.** `vk_mat_vec_id_push_constants` gets `uint32_t up_offset` and `uint32_t glu_op` (0 none,
  1 swiglu, 2 geglu): 56 bytes, within the 128-byte limit of gfx900.
- **Pipelines.** `pipeline_dequant_mul_mat_vec_id_glu_f32[type]` (and the q8_1 variant when `quantize_y`) for at
  least Q4_K, Q5_K, Q6_K, Q8_0 (the Qwen3.6 Q4_K_M expert types). A type without a fused pipeline falls back unfused.
- **Host.** In the fusion chain of `ggml_backend_vk_graph_compute`, before the existing `MUL_MAT_ID_*` entries, match
  `{ MUL_MAT_ID, MUL_MAT_ID, GLU }` with the conditions above: `num_additional_fused_ops = 2`, fusion name
  `MUL_MAT_ID_GLU`, `op_srcs_fused_elementwise = { false, false, true }`. `ggml_vk_mul_mat_vec_id_q_f16` takes the
  gate node, binds `D` to the GLU output, sets `up_offset` and `glu_op`, and loops over tokens as today (one
  dispatch per token). The Q8_1 quantize of the input happens once, as today.
- **Test hooks** (needed by the tests, part of this change): `ggml_backend_vk_reg_get_proc_address` already returns
  `ggml_backend_vk_last_graph_dispatches` (`uint32_t (*)(ggml_backend_t)`). Add
  `ggml_backend_vk_fuse_moe_glu_enabled` (`bool (*)(ggml_backend_t)`, returns `device->fuse_moe_glu`).

## Acceptance
1. Builds with `build-up.bat`.
2. `test-backend-ops -o MOE_GLU_FUSION -b Vulkan1` passes all cases (Q4_K at n_tokens 1/4/8/16/64; Q5_K, Q6_K, Q8_0
   at 1/8; 256 experts, 8 used, 2048 -> 512, broadcast input), with `GGML_VK_MMVID_MAX_TOKENS` unset and `=64`, and
   with `GGML_VK_FUSE_MOE_GLU=0`. Each case compares with the CPU, and for n_tokens <= the threshold checks that the
   graph ran in at most n_tokens + 2 dispatches (unfused: 2*n_tokens + 2 or more).
3. The existing `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID` and `GLU`-op cases still pass.
4. In-model: v11 flags, temp-0 decode text equal to `GGML_VK_FUSE_MOE_GLU=0`, PPL within ±0.005, and decode ms/token
   A/B (short / 27K, 2 rounds; dual v6 1 round). Expected gain about 0.3–0.6 ms/token (1–1.5%).

## Out of scope
Prompt batches (tiled MUL_MAT_ID), the shared expert (dense MUL_MAT gate/up), the down projection.
