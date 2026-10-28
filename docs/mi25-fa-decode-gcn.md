# GCN flash-attention decode kernel (mi25 fork)

## Why
On the WX 9100 (gfx900, GCN5, wave64, no integer dot product) the scalar FA kernel (`flash_attn.comp`) handles
decode for Qwen3.6 (head size 256, GQA 8) and Qwen3.8-27B (head size 256, GQA 6). With 8 GQA rows it picks
`row_split = 4`, a 256-thread workgroup, `D_split = 8` and a 26 KB occupancy limiter. Measured per layer (one token,
2 KV heads): q8_0 KV 221 µs at 8K, 685 µs at 32K, 1461 µs at 73K. Time barely changes with KV bytes (q4_1 1354 µs vs
f16 1899 µs at 73K), so it is not bandwidth-bound: each of the 4 row groups loads and dequantizes the same K/V
again, the q8_0 scale is reloaded for every vec4, and O is replicated across 8 column groups. Roofline for 73K,
2 KV heads, q8_0: 79 MB of K/V (163 µs at 484 GB/s), 1.2 GFLOP (~100 µs at 12 TFLOPS).

## What
A dedicated decode shader, `vulkan-shaders/flash_attn_decode_gcn.comp`, used only when **all** of these hold:
- the device architecture is `AMD_GCN` and the FA path would be `FA_SCALAR`;
- the GQA path is active (`gqa_ratio > 1`, which means N ≤ 8 query tokens), `gqa_ratio <= 8`;
- `HSK == 256 && HSV == 256`;
- K and V types are each one of Q8_0, Q5_1, Q4_0;
- no sparse mask, no sinks, `max_bias == 0`, `logit_softcap == 0`, no f16 dequant scratch (`use_dequant_kv`), no mask opt;
- env `GGML_VK_FA_DECODE_GCN` is unset or not `0` (`GGML_VK_FA_DECODE_GCN=0` turns the new path off for A/B tests).

Everything else (prompt batches, spec-verify batches of more than 8 tokens, other GPUs, other types, sinks, ALiBi,
softcap) keeps using the existing kernels unchanged.

## Design decisions
- **Same interface as `flash_attn.comp`.** It uses the same 8 bindings, the same `vk_flash_attn_push_constants`, the same spec
  constants (`get_fa_spec_constants`), `flash_attn_base.glsl` (`init_indices`, `gqaStore`, `perElemOpStoreCol0`) and
  `flash_attn_dequant.glsl` (`dequantize4`). It writes the same split_k layout (O, then L and M), so
  `flash_attn_split_k_reduce.comp` and its dispatch are unchanged.
- **Workgroup = one wave of 64 threads.** It needs `requireFullSubgroups` with subgroup size 64. Br = 8 rows (all GQA rows of one
  token and KV head; rows `r >= N` are skipped with a uniform branch, never read). Bc = 64 KV columns per tile.
- **Score phase (QK).** Lane `c` owns KV column `j*Bc + c` and computes the full 256-element dot product for all 8
  rows: `for d4 in 0..63: K4 = dequantize4(K row, 4*d4); for r < N: s[r] += dot(Q[r][d4], K4)`. Q is read from
  binding 0 with an index that is uniform across the wave, so the AMD compiler can use scalar loads (no LDS, no
  shuffles). Q is not pre-scaled: `s[r] = s[r] * p.scale + mask`. An inactive column (beyond KV) gets
  `s = NEG_FLT_MAX_OVER_2`.
- **Mask.** In GQA mode all rows of a token share one mask row: `data_m[m_offset + col]`, with
  `m_offset = gqa_iq1*KV + ((iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * p.nem1 * KV` (the second term only when
  `nem2 != 1 || nem3 != 1`, exactly as in `flash_attn.comp`).
- **Online softmax per tile.** For each row: `rowmax = subgroupMax(s[r])`, `Mnew = max(M[r], rowmax + FATTN_KQ_MAX_OFFSET)`,
  `eM = exp(M[r] - Mnew)`, `p = active ? exp(s[r] - Mnew) : 0`, `L[r] = L[r]*eM + subgroupAdd(p)`, `O[r] *= eM`. Then
  store `p` to `shared float Psh[Br*Bc]` at `[r*Bc + c]`. All lanes hold identical M and L.
- **Value phase (PV).** Lane `t` owns output dims `4t..4t+3` (HSV/4 = 64 = lanes). For each column `c` of the tile
  (uniform loop, bounded by `KV` when `KV_bounds_check`): `V4 = dequantize4(V row j*Bc+c, 4t)`, then
  `O[r] += Psh[r*Bc + c] * V4` for `r < N`. V loads are coalesced (the 64 lanes read one V row), and each V element is
  dequantized once and used for 8 FMAs. `barrier()` goes before and after the PV phase (Psh reuse).
- **Output.** As in `flash_attn.comp` for `gqa_ratio > 1`:
  - with split_k (`p.k_num > 1`), store unnormalized O through `gqaStore`, and from lane 0 store L and M at the split_k L/M offsets;
  - otherwise multiply by `1/L` (0 if `L == 0`) and `gqaStore`.
- **Math is fp32 throughout** (`FLOAT_TYPE float`, `ACC_TYPE float`); there is one SPIR-V module, `flash_attn_decode_gcn_f32`.
- **Split-k.** For the decode path, `split_k = shader_core_count * mult / (workgroups_x*workgroups_y*workgroups_z)`,
  where `mult` is 8 by default (one-wave workgroups need more of them than the 4-wave scalar kernel) and
  can be overridden by env `GGML_VK_FA_DECODE_GCN_SPLIT_MULT`. The existing rounding of `split_kv` to the 64-column
  alignment and the `split_k_size` reservation apply unchanged.

## Host wiring (ggml-vulkan.cpp)
- `FaCodePath` gets `FA_DECODE_GCN`.
- `get_fa_tuning_params_decode_gcn()` returns `path = FA_DECODE_GCN, block_rows = 8, block_cols = 64,
  workgroup_size = 64, subgroup_size = 64, d_split = 1, row_split = 1, shmem_staging = 0, limit_occupancy_shmem = 0`.
- In `ggml_vk_flash_attn`, once `use_sparse` is known and before `get_fa_pipeline_state`, override `tuning_params` when
  the conditions above hold. Force `use_mask_opt = false` for this path.
- In pipeline creation, a loop over `pipeline_flash_attn_f32_f16` entries with `path == FA_DECODE_GCN` creates them from
  `flash_attn_decode_gcn_f32` with 8 bindings, `sizeof(vk_flash_attn_push_constants)`, `wg_denoms {Br,1,1}`,
  `get_fa_spec_constants`, `align = aligned ? Bc : 1`, `disable_robustness = true`, `require_full_subgroups = true` and
  `required_subgroup_size = 64`.
- Every other place that switches on `FaCodePath` keeps its behavior for the existing paths.

## Acceptance
1. It builds (`build-fc9.bat`), and the shader compiles in `vulkan-shaders-gen`.
2. `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan1` passes all cases. That includes the GQA decode block added in
   `make_test_cases_eval` (head size 256, GQA 8 and 6, kv 100 … 73729, nb 1/3/8, q8_0/q8_0, q5_1/q4_0, q4_0/q5_1,
   large logits, plus the sinks/ALiBi/softcap fallbacks), with both `GGML_VK_FA_DECODE_GCN` unset and `=0`.
3. Perf (`test-backend-ops perf -o FLASH_ATTN_EXT`, decode cases): q8_0 at 73K must be clearly faster than 1461
   µs per layer (target 2× or better). It must not be slower at 8K (221 µs).
4. In-model: temp-0 token identity or PPL against `=0`, and TG A/B at 27K/73K on Qwen3.6 v10, dual v5 and 27B v3.

Note on TDD: the existing kernel already passes the correctness cases, so these tests guard against regressions
rather than failing first. The new kernel's effect shows in the perf A/B (`GGML_VK_FA_DECODE_GCN=0` vs unset).

## Known risk
In the score phase each lane reads its own K row (544 B or 1088 B stride between lanes), so its loads are not
coalesced. If perf shows L1 thrash, the follow-up is to stage the raw K tile in LDS. That is not part of this change.
