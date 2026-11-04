# MI25 / WX 9100 llama.cpp fork: optimizations and settings

This is a reference for what we changed in llama.cpp for a Radeon Pro WX 9100 (gfx900, GCN5) on Windows,
why each change helped (or did not), and which runtime settings we tuned.

Everything here comes from three sources: the results log (`mi25-fork-results.md`, rounds in date order),
the git history of this fork, and the shipped launcher scripts. Where something was not recorded, the text says "not measured".

Contents:

1. Summary
2. Code optimizations
3. Runtime settings (one table per server type)
4. Settings tried and rejected
5. Lessons and gotchas

---

## 1. Summary

### Hardware and software

| Part | Detail |
|---|---|
| Main GPU | Radeon Pro WX 9100 (gfx900 / GCN5), 16 GB HBM2, about 484 GB/s, AMD proprietary Vulkan driver (26.5.2 since 2026-10-19) |
| Second GPU (dual setups) | RTX 3070 Ti, 8 GB, Vulkan. It also drives the desktop, so about 7.1 GB is free |
| CPU / RAM | Ryzen 5 5600X, 32 GB DDR4-3200 |
| OS | Windows 11 (TdrDelay is 30 s) |
| No Resizable BAR | The host-visible VRAM window is 256 MiB |
| Fork | `llama.cpp-fc9ce6b`, branch `mi25-up`, rebased on upstream `f5222f25c`. Shipped build: commit `90068a274` |
| Models | Qwen3.6-35B-A3B (MoE, UD-Q4_K_M MTP GGUF; one version with CPU-resident experts requantized to Q3_K) and Qwen3.8-27B (dense, UD-IQ3_XXS) |

### Start and end numbers

All numbers are from the results file. "Start" is the first measurement of that setup. "End" is the last shipped
version. Prompt processing (PP) and token generation (TG) are in tokens per second. Real C++ text was used as the prompt
(synthetic repeated prompts overstate MoE speed; see section 5). Tokens in the column heads are prompt sizes.
Noise between runs is 5-8%, so small differences are not meaningful.

**Qwen3.6-35B-A3B, single GPU (WX 9100 only)**

| | PP 3.4K | PP 8.7K | PP 27K | PP 73K | TG short | TG 27K |
|---|---|---|---|---|---|---|
| Start: bundled LM Studio build (16 CPU expert layers, FA off, 32K ctx) | 204 | 191 | 115 (28.5K) | not measured | 23.7 | 16.0 (28.5K) |
| Fork v1, first fork result | 252 | 211 | 121 (28.5K) | not measured | 23.8 | 15.7 (28.5K) |
| Fork v9, after the fan upgrade (128K ctx) | 325 | 338 | 288 | 91 | 23.3 | 20.7 |
| End: v13 (requantized experts, MTP) | 384 | not measured | 313 | 85.5 (v12, 21 CPU layers) | about 30.5-31.3 on a short prompt (v14 vision test, MTP) | not measured |

Spec-replay decode on real agent turns (lower is better): 36.85 ms/token (v12) -> 35.02 (cpuQ3 model + compact KV) ->
33.91 (plus ngram n-max 12). The 73K number is from v12 at `--n-cpu-moe 21`. The v13/v14 launchers use the same 21.

**Qwen3.6-35B-A3B, dual GPU (WX 9100 + RTX 3070 Ti)**

| | PP 3.4K | PP 8.7K | PP 27K | PP 73K | TG short |
|---|---|---|---|---|---|
| Start: dual v1, layers 0-9 on the 3070 Ti (pre-fan) | 513 (warm) | 450 | 347 | 88 | 26.5 |
| Dual v1, post-fan | 505 | 497 | 398 | 117 | 23.8 |
| Dual v8 (end of the tuning rounds) | 557 | not measured | 446 | not measured | 42-44 on a short prompt with MTP (v9 vision test) |
| Dual v6/v7 (MTP) | 560 | 575 | 447 | 196 | spec replay 26.84 ms/token (37.3 t/s) |

The first dual probe (experts of 16 layers in host memory, split by `-ts`) gave PP 414 / 417 / 283 but TG fell to 14.2.
That was the 3070 Ti sitting in power state P8 during decode. Fixed with the NVIDIA setting "Prefer maximum performance".

**Qwen3.8-27B, single GPU**

| | PP 3.4K / 8.7K / 27K | TG 3.4K / 8.7K / 27K |
|---|---|---|
| Start: v1 (Q3_K_XL, FA, q8 KV, 32K), warm card | 75-77 / 44 / 32 | 5.4-5.9 (10.7 cold) |
| v1 after the fan upgrade | 87 / 75 / 53 | 9.8 / 8.5 / 7.9 |
| End: v4/v5 (IQ3_XXS, rebased build) | 109 / 87 / 56 | 9.5 / 9.0 / 9.3 (noisy) |

**Qwen3.8-27B, dual GPU**

| | PP 3.4K / 8.7K / 27K / 73K | TG 3.4K / 8.7K / 27K / 73K |
|---|---|---|
| Start: dual v1 (Q3_K_XL, -ts 22,42), pre-fan | 121 / 74 / 49 / 27.7 | 9.7 / 10.4 / 9.1 / 2.1 |
| End: dual v2/v3 (IQ3_XXS, -ts 26,38), mean of 2 rounds | 129.5 / 113 / 77 / 42 | 11.7 / 9.3 / 9.9 / 7.9 |

**Hermes agent, first request after a launch (13.4K-token system prompt):** 43 s -> 7.2 s with the prefix cache
(single v2), then 0.36 s once the slot file was saved with the MTP state (single v6).

### Biggest wins

| Win | Effect |
|---|---|
| Keep device buffers out of the 256 MiB BAR window | PP +24% at 8.7K, +53% at 26.9K (Qwen3.6, FA off) |
| Cap scalar flash attention at 8 block rows on GCN | FA from 50-130 to 740-750 GFLOPS; FA on went from a timeout at 27K to PP 122 |
| Chunked matmul attention for prompts | PP +28% at 8.7K, +61% at 27K (128K config) |
| GCN MoE tuning (tiles, rows-per-expert rule, `ggml_cont`) | PP +24% at 3.4K, +10% at 8.7K |
| MTP speculative decoding plus ngram-mod | Spec replay about -21% ms/token (single and dual) |
| Layers on the 3070 Ti (dual) | PP about +60-90% at 3.4K-27K, TG up at long context |
| Per-token mat-vec for batches up to 64 tokens (`GGML_VK_MMVID_MAX_TOKENS=64`) | Spec-decode verify batches -7% ms/token |
| CPU experts requantized to Q3_K plus compact KV | Spec replay -5.0% (single), PP 3.4K 331 -> 384 |
| New fan on the WX 9100 | 27B PP +65-70%, no-spec TG 5.4-5.9 -> 9.8-7.9 (thermal limit removed) |
| Hermes prefix cache | First request 43 s -> 7.2 s -> 0.36 s |
| GCN FA decode kernel | TG +6-8% at 73K context |

---

## 2. Code optimizations

SHAs on the left are the rebased commits on `mi25-up` (upstream `f5222f25c`). The older pre-rebase SHAs from the
results file are given in brackets. Test-only commits are listed at the end of the section.

### 2.1 Shipped

#### 2.1.1 GCN tuning for MoE prompt processing

- **Commit:** `38367ffbd` (pre-rebase `c764a087c` on `mi25-fc9`). **Status:** shipped in every Qwen3.6 version since v1.
- **What changed** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `src/models/delta-net-base.cpp`):
  1. The GCN warptile (taken from the RADV driver path) is now also used on the AMD proprietary driver.
  2. For `MUL_MAT_ID` on GCN, the tile is chosen by the average rows per expert (`nei1*nei0/ne02`) instead of the token count.
     At 32 rows per expert or fewer this picks the small tile, which is what ub512 and ub1024 produce.
  3. In the delta-net conv state, the transposed `qkv_mixed` is made contiguous with `ggml_cont` before the `CONCAT`.
- **Why it helps:**
  - A MoE prompt batch spreads its tokens over 256 experts, so each expert sees few rows. The token count is the wrong
    input for tile choice, and large tiles waste work on mostly empty tiles.
  - The `CONCAT` read a transposed source with a large stride, which is very slow on GCN (commit message).
  - Large tiles on the proprietary driver were tried and ran at about 15 GFLOPS (broken), so they stay off.
- **Measured** (real text, ub1024, `--load-mode none`, same binary with one change at a time):

| Change | PP 3.4K | PP 8.7K |
|---|---|---|
| Base fc9ce6b | 202.6 | 186.6 |
| `ggml_cont` before CONCAT | 206.2 (+2%) | 193.6 (+4%) |
| GCN warptile on the proprietary driver | 216.7 (+7%) | 202.2 (+8%) |
| Small tile via rows-per-expert rule | 223.2 (+10%) | 207.2 (+11%) |

  - Total for the commit: PP +24% at 3.4K and +10% at 8.7K against the bundled LM Studio build (together with `--load-mode none`).
  - `test-backend-ops` on Vulkan1: MUL_MAT_ID 939/939, MUL_MAT 2272/2272, CONCAT 198/198, CPY 254/254, CONT 78/78.

#### 2.1.2 Keep device buffers out of a small BAR window

- **Commit:** `011e86ad3` (pre-rebase `90f97af25`). **Status:** shipped from Qwen3.6 v3 on. The launchers also set
  `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, which gives the same effect explicitly.
- **What changed** (`ggml-vulkan.cpp`, `ggml_vk_get_device`, 14 lines): on GCN, if no DEVICE_LOCAL|HOST_VISIBLE heap is
  larger than 256 MiB, `disable_host_visible_vidmem` is set automatically.
- **Why it helps:** ggml-vulkan allocates device buffers in DEVICE_LOCAL|HOST_VISIBLE memory first. This PC has no
  Resizable BAR, so that memory type is a 256 MiB BAR window and the GPU reads it at about PCIe speed. The attention
  scratch buffers (prealloc_x/y for the KQ and KQV matmuls) landed there whenever they fit, so only the KV-reading
  matmuls slowed down, by about 9x. Other ops in the same graph ran at normal speed (dense q4_K 4.69 TFLOPS in both modes),
  which ruled out GPU clocks. Which buffers fit in the window changes with their size, which explains why ub512 decayed
  gradually from about 3K context, why ub1024 stepped down between n_kv 16384 and 17408, and why one version was
  sometimes fast and sometimes slow at 30K.
- **Measured** (real text, ub1024, FA off, 2 interleaved rounds):

| | Before | Fix built in | Change |
|---|---|---|---|
| PP 3.4K | 250 | 258 | +3% |
| PP 8.7K | 211 | 262 | +24% |
| PP 26.9K | 121 | 185 | +53% |
| PP 30.3K | 173 (103 in another run) | 174 | stable |
| TG, load time, VRAM | | | no regression (VRAM +0.03 GB) |

  - Evidence: KQ ran at 3.47 TFLOPS at KV 16384, then 0.367 at 17408 and beyond. With the BAR window disabled it held about 3.2 TFLOPS up to 28.6K.
  - The driver also reports its device-local heap as 23.1 GiB on a 16 GB card.

#### 2.1.3 Scalar flash attention capped at 8 block rows on GCN

- **Commit:** `82973aa37` (pre-rebase `7c2fcfaa5`). **Status:** shipped from Qwen3.6 v4 on (flash attention on).
- **What changed** (`ggml-vulkan.cpp`, `get_fa_tuning_params_scalar`, plus FA perf cases in `test-backend-ops.cpp`):
  `block_rows` is capped at 8 on GCN instead of 16.
- **Why it helps:** with 16 block rows the scalar FA shader spills registers on gfx900. On the Qwen3.6 prompt shape
  (head size 256, GQA 8) it ran at 50-130 GFLOPS. With 8 rows it runs at 740-750 GFLOPS. A faster-looking setting with
  128-thread workgroups gave wrong results, because a row group has to fill whole 64-wide GCN subgroups, so it was rejected.
- **Measured** (real text, 32K context, ub1024):

| | PP 3.4K | PP 8.7K | PP 26.9K | PP 30.3K |
|---|---|---|---|---|
| FA on, before | 163 | 88 | timeout | not measured |
| FA on, after (TG 23.6 / 23.7 / 20.3 / 19.6) | 245 | 225 | 122 | 104 |
| FA off (reference) | 259 | 261 | 182 | 174 |

  - FA on uses 2 GB less VRAM (13.5 vs 15.4 GB), which is what makes 128K context possible.
  - `test-backend-ops` FLASH_ATTN_EXT: 5321/5321.

#### 2.1.4 Chunked matmul attention for long prompt batches

- **Commits:** `e00119418` and `911fef62d` (pre-rebase `523663f6d`, `e3ea7d17f`, on `mi25-fachunk`).
  The first commit message says "WIP, not validated"; the validation came afterwards and is listed below. The follow-up commit bounds the scratch memory.
- **Status:** shipped from Qwen3.6 v8 on. It is switched on by `LLAMA_MI25_FA_CHUNK=1` in every Qwen3.6 launcher.
- **What changed** (`src/llama-graph.cpp` in `build_attn_mha`, a few lines in `ggml-vulkan.cpp`):
  - For prompt batches of at least `LLAMA_MI25_FA_CHUNK_MIN_TOKENS` tokens (default 512; launchers use 32) and with
    `LLAMA_MI25_FA_CHUNK_MIN_KV` <= n_kv <= `LLAMA_MI25_FA_CHUNK_MAX_KV` (defaults 4096 and 32768), flash attention is replaced by
    KQ `mul_mat`, `soft_max`, then `mul_mat` with V, chunked over query tokens. The scratch budget is `LLAMA_MI25_FA_CHUNK_MIB` (default 128).
  - K is dequantized once per layer (q8_0 -> f32 -> f16) because Vulkan `mul_mat` cannot read the strided q8_0 view.
    V is transposed once per layer. Decode and short batches keep using FA.
  - Memory fixes: when the env var is set, the Vulkan graph optimizer no longer pulls nodes over 32 MiB forward (it had
    interleaved chunks and cost 446 MiB); and V is transposed without a leaf destination tensor, which ggml-alloc never frees (192 MiB at 32K).
    The compute buffer at 32K is 1124 MiB, inside the 1156 MiB the 128K config already reserves.
- **Why it helps:** the scalar FA shader uses only a few percent of what gfx900 can do (about 0.45 TFLOPS in the last
  2048-token batch at 27K; FA was 71.5% of GPU time in a 27K profile). Split-K does not apply to prompt FA, because 2048 queries
  already give about 4096 workgroups, so the limit is per-tile efficiency, not occupancy. Matmul-style attention uses the
  better `mul_mat` kernels and is about 1.5x faster here.
- **Measured** (128K q8_0, same binary off vs on, 2 interleaved rounds):

| Prompt | Off | On |
|---|---|---|
| 3.4K | 319 | 320 |
| 8.7K | 248 | 317 (+28%) |
| 27K | 123 | 198 (+61%) |

  - A 119K prompt ran at 40.6 t/s (previous 38.7), 15.06 GB VRAM, no fault. TG unchanged. PPL at 16K: 1.4095 on vs 1.4093 off (+-0.013).
  - Greedy tokens match on short prompts and diverge on 8K and 27K prompts at near-ties, from the different summation order.
  - Later settings: `LLAMA_MI25_FA_CHUNK_MIN_TOKENS=32` made the path apply to short agent turns too (see 2.1.6).
    Dual setups raise the cap with `LLAMA_MI25_FA_CHUNK_MAX_KV` (48K in dual v2, 64K in dual v3 and later, with compact KV).
    A 128-token-batch PPL check at 16K context: 1.4241 (chunked) vs 1.4254 (normal FA) for q8_0 KV, 1.4253 vs 1.4257 for q5_1/q4_0.
  - The compute buffer grows about 30 KiB per KV token for the chunked path: 1616 MiB at 48K, 2096 MiB at 64K.
  - The single-GPU launchers keep the default 32K cap, because above it the temporaries need extra VRAM that the WX does not have.

#### 2.1.5 Per-token mat-vec for MUL_MAT_ID up to 64 tokens, and the compact multi_add

- **Commit:** `25bbc773e` (pre-rebase `6ff784cf9` on `mi25-tg`). **Status:** shipped. `GGML_VK_MMVID_MAX_TOKENS=64` is in every Qwen3.6 launcher.
- **What changed** (`ggml-vulkan.cpp`, `multi_add.comp`, push-constant headers):
  - New env `GGML_VK_MMVID_MAX_TOKENS` (default 8) sets the largest batch that uses the per-token `MUL_MAT_ID` mat-vec path.
  - `multi_add_small`: fuses up to 4 adds with a smaller push-constant layout, because the full `multi_add` (212 bytes) exceeds gfx900's 128-byte `maxPushConstantsSize`.
  - Tuning overrides `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ`, `GGML_VK_FA_GCN_OCC_KB`.
- **Why the threshold helps:** batches of 9-64 tokens (mostly ngram or MTP verify batches) went to the tiled `matmul_id`
  kernel, which on gfx900 is much slower than running the per-token mat-vec. At the q4_K gate shape: 16 tokens 2010 -> 327 us, 64 tokens 4069 -> 1282 us; the down projection 1931 -> 655 us and 4043 -> 2622 us.
  Above about 110-180 tokens the mat-vec loses, so the threshold stops at 64. Raising it to 128 tokens gave only -3.6% (tried, reverted).
- **Measured** (spec replay on real agent turns, ngram n-max 16, 2 interleaved rounds):

| Config | review-fixes ms/token, MMVID 8 -> 64 | TG, MMVID 8 -> 64 |
|---|---|---|
| single (v9) | 47.8 -> 44.5 (-7%) | 20.71 / 21.13 -> 22.46 / 22.51 |
| dual (v4) | 43.9 -> 40.9 (-7%) | 22.19 / 23.41 -> 24.44 / 24.43 |

  - `test-backend-ops` MUL_MAT_ID passes 939/939 at 8 and at 64. Temp-0 outputs differ slightly from threshold 8 because verify batches now use the single-token kernel; they are deterministic from run to run.
- **`multi_add_small`: correct, no gain.** ADD 103/103, multi_add 113/113, 176 fused groups per token. v9 22.71/22.82 -> 22.70/22.73 t/s. It stays in the binary and is active.
- **Note on 27B:** MMVID 64 was worse on the 27B (round 6, B2: mean +11.45% but worst size -15.97%, versus +13.98% / +3.08% without it). The 27B launchers do not set it.

#### 2.1.6 Short agent turns: chunked attention from 32 tokens and CPU experts for small batches

- **Commits:** none new. This is launcher settings on top of 2.1.4 and the existing `GGML_OP_OFFLOAD_MIN_BATCH` variable.
- **Settings:** `LLAMA_MI25_FA_CHUNK_MIN_TOKENS=32` (all Qwen3.6 launchers) and `GGML_OP_OFFLOAD_MIN_BATCH=512` (single-GPU launchers only).
- **Why it helps:** profiling a 76-token turn at 13.4K context on one GPU showed 902 ms of GPU compute, of which flash attention
  was 399 ms (scalar FA over the whole context) and MoE matmul 313 ms. About 1.2 s more per turn was the per-batch upload of the 16 host-memory expert layers.
  Running small batches' CPU experts on the CPU removes that upload; the chunked path replaces the slow scalar FA.
- **Measured** (replayed Hermes prompts): single, 8 turns, 31.7 -> 26.8 s total (-15.5%), 123-137-token turns 2.3-2.9 -> 1.2-1.4 s.
  Dual, 12 turns, 25.0 -> 22.9 s (-8.5%). Local-agent prompt replay: single v8 -> v9 562 -> 477 s (-15%), dual v3 -> v4 359 -> 319 s (-11%).
  On dual, `GGML_OP_OFFLOAD_MIN_BATCH` adds nothing (1024 was worse for 580-800-token turns), so dual does not set it.

#### 2.1.7 GCN flash-attention decode kernel

- **Commit:** `5bf82b300` (pre-rebase `a2c8367a1` on `mi25-fa-decode`), with test commits `1e5e94ec9` and `90068a274`. **Status:** shipped in the Qwen3.6 roots (single v11, dual v6, Hermes v5) and in every later version built on `mi25-up`.
- **What changed:** a new shader `flash_attn_decode_gcn.comp` (151 lines) plus host wiring in `ggml-vulkan.cpp`. It is used only for GCN, scalar-FA path, GQA ratio 2-8,
  head size 256, K and V in q8_0 / q5_1 / q4_0, and no sinks, softcap or ALiBi. One 64-thread workgroup (one GCN wave) handles all 8 GQA rows of one KV head,
  with 64 KV columns per tile. Each lane owns one KV column for the scores, then one output dimension group for the values. It writes the same split-K layout as the old kernel.
  `GGML_VK_FA_DECODE_GCN=0` turns it off. Full design: `docs/mi25-fa-decode-gcn.md`.
- **Why it helps:** the old scalar kernel's decode time barely tracks the bytes read (q4_1 1354 us vs f16 1899 us at 73K), so it is ALU- and latency-bound,
  not bandwidth-bound: each of its 4 row groups loads and dequantizes the same K/V again and re-reads the q8_0 scale for every vec4.
  The new kernel dequantizes each element once and uses it for 8 FMAs.
- **Measured:**

| Per layer at 73K (us) | q8_0 | q4_0 | q5_1 | q5_1 / q4_0 | 27B shape (GQA 6) |
|---|---|---|---|---|---|
| old -> new | 1461 -> 1157 (-21%) | 1358 -> 1148 (-15%) | 1645 -> 1277 (-22%) | 1499 -> 1248 (-17%) | 2871 -> 1913 (-33%) |

  - At 8K it is about 17% slower per layer (q8_0 221 -> 259 us) and slightly better at 32K. Worth about 1% in-model, so the planned gate to KV >= 16K was not done.
  - In-model TG (plain, no spec, 2 interleaved rounds): v10 at 73K 19.2 / 19.6 -> 20.6 / 20.7 (+6%); dual v5 at 73K 23.0 / 17.7 -> 25.2 / 24.6 (+8% against the cleaner round). Short and 27K contexts are within 1-2%, which is noise.
  - FLASH_ATTN_EXT passes 5487/5487 with the new kernel. Temp-0 identity could not be checked (the profiling script adds a random UUID to each prompt).
  - On the 27B (GQA 6) the rebased binary includes this kernel. The 27B gain was +2% at 27K when tested alone and the effect was not separated from the rest of the rebase gain, so it was not shipped on its own.

#### 2.1.8 Model-side: the rebase itself (27B)

- The 27B v4 uses the same v3 flags on the rebased build (`mi25-up-v1`, commit `90068a274` on upstream `f5222f25c`). PP +30% (84 / 72 / 50 -> 109 / 87 / 56 at 3.4K / 8.7K / 27K).
  TG means 8.4 / 7.2 / 9.0 -> 9.5 / 9.0 / 9.3 (noisy; worst size +3.08%, mean +13.98%). PPL 1.4899 vs 1.4902. Agent tasks 4/4 vs 3/4.
- Which upstream change gave this was not isolated.

#### 2.1.9 Instrumentation (shipped, off by default)

- `60f4dc70a` `GGML_SCHED_TIMING`: per-split copy / issue / wait timing in `ggml-backend.cpp`. It syncs after every split and adds about 1%. Used for the decode profile in section 2.3.
- `25bbc773e` tuning knobs `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ`, `GGML_VK_FA_GCN_OCC_KB` (no effect found).
- `200e980d7`, `500bdf5b9` `GGML_VK_DIAG_NO_BARRIERS` and `GGML_VK_DIAG_COUNT`: timing-only diagnostics. No-barrier output is wrong; never put it in a launcher.

### 2.2 Built and correct, but not enabled in the shipped launchers

#### 2.2.1 Expert prefetch (upload overlap)

- **Commits:** `10596901e`, `1a1b9e87c` (pre-rebase `b6f32515f`, `ad6a8c0e0` on `mi25-overlap`). **Status:** shipped in v5 only (on), then switched off from v6.
  The code is still in the binary, off by default. It is on only when `LLAMA_MI25_PREFETCH` is set. The code tests for presence, so `LLAMA_MI25_PREFETCH=0` also turns it on.
- **What changed** (`src/llama-context.cpp/.h`, `ggml-backend.cpp/.h`): a scheduler "split done" callback and an input-alloc callback.
  While layer k computes, layer k+1's CPU-resident experts upload over the Vulkan transfer queue into one of two VRAM staging slots, and the device copy is pre-bound to its slot so the graph allocator skips it.
  Expert weights never change, so a slot stays valid across ubatches. It needs `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` as well.
- **Why it helps:** the PCIe upload of expert weights for a large batch no longer waits behind compute.
- **Measured** (v4 vs v5, 128K q8_0, 2 interleaved rounds): PP 3.4K 242 -> 261 (+8%), 8.7K 199 -> 239 (+20%), 26.9K 111 -> 120 (+8%), 73K 56.1 -> 58.0 (+3%), 119K 38.6 -> 38.2 (no change). TG unchanged. Temperature-0 tokens identical to v4; PPL 1.5309 +-0.0163 both on and off.
- **Why it was dropped:** v6 moved to `-ub 2048`, which gave PP 3.4K 322 vs 263 (+22%) and 8.7K 273 vs 240 (+14%) at 128K without prefetch.
  With ub2048 plus prefetch the VRAM reaches about 15.45 GB at 128K, and above about 15.4 GB the AMD driver raises a GPU fault. VRAM at 128K with prefetch:
  model 12463 MiB, KV q8 1360, recurrent 63, two prefetch slots 2 x 498, compute 832. Prefetch costs about +0.56 GB net. Where it fits (96K, ncm16, ub2048) it gave +10% at 3.4K and +7% at 8.7K, but that trades away context. Not shipped.

#### 2.2.2 Fused MoE gate / up / GLU decode mat-vec

- **Commits:** `a98c1f09c` on branch `mi25-fuse`, with `5ec03741b`, `96c096fc4`, `0e8ebf0d2` (tests and test hook). **Status:** not shipped.
- **What changed:** `MUL_MAT_ID` gate, `MUL_MAT_ID` up and the SWIGLU/GEGLU `GLU` run as one mat-vec-id dispatch per token (n_tokens <= `GGML_VK_MMVID_MAX_TOKENS`). Default on for GCN, `GGML_VK_FUSE_MOE_GLU=0` turns it off.
  It removes 24 dispatches and 24 barriers per decode token on the GPU-resident MoE layers. Design: `docs/mi25-fuse-moe-glu.md`.
- **Measured:** correct (all tests, temp-0 identical); op level 5-7% faster per layer. In-model spec replay: +0.25% single, +0.20% dual in round 2, and negative in round 1.
  It misses the 1.5% ship rule. A single-pass gate+up variant was estimated at 1-2%, borderline and not built.
- **Why no gain:** the per-layer op savings are too small a share of the 35-40 ms token. Barriers and dispatch count were not the main limit (see 2.3).

#### 2.2.3 Narrower compute-queue barriers (`GGML_VK_BARRIER_MODE`)

- **Commit:** `c8e0998dd`. **Status:** in the shipped branch, default unchanged (mode 0), not used by any launcher.
- **What changed:** `ggml_vk_sync_buffers` emits one global memory barrier with COMPUTE|TRANSFER stages and read+write access on both sides, about 880 per decode token on Qwen3.6.
  Mode 1 narrows the source access to write-only. Mode 2 uses COMPUTE -> COMPUTE and SHADER_WRITE -> SHADER_READ|WRITE, and brackets transfer commands with a full barrier.
- **Why it was tried:** with all barriers skipped (diagnostic only, output wrong) a decode token got about 17-20% faster per the commit, and 7-10 ms per token per the fuse design note.
- **Measured:** correct, no gain: v11 short 42.4 vs 42.2 ms, 27K 48.0 vs 48.1 ms. The driver evidently does not make narrower barriers cheaper. The reason was not investigated further.

### 2.3 Where a decode token goes (profile that guided the work)

`decode_profile.py` with `GGML_SCHED_TIMING` (ms per token, plain decode):

| | v9 short | v9 27K | dual v4 short | dual v4 27K |
|---|---|---|---|---|
| Total | 43.0 | 48.4 | 42.4 | 46.2 |
| Splits per token | 34 (17 GPU + 17 CPU) | 34 | 15 | 15 |
| GPU issue / wait | 6.9 / 25.0 | 4.7 / 32.7 | V0 4.3 / 11.6, V1 5.0 / 24.2 | V0 4.3 / 17.3, V1 2.9 / 26.0 |
| CPU experts compute / copy | 6.5 / 2.2 | 6.7 / 2.9 | 2.5 / 1.0 | 2.6 / 1.0 |

- The GPU runs about 1,760 dispatches per token (about 37 ms). Mat-vecs dominate; FA at 27K is 7-9 ms.
- In-model mat-vecs ran about 2x slower than the same shape in isolation (q4_K 2048x4096: 67.7 vs 27.4 us). A GPU keep-alive test (to rule out clock ramp) gave no gain (27K TG 20.47 with keep-alive vs 20.72 without), so the cause is unexplained.
- gfx900's 128-byte push-constant limit disables the `multi_add` and `rms_norm_mul_rope` fusions.
- Mat-vec bandwidth by format at the 27B shapes (n = 1, GB/s): q3_K 120, q4_K 246, q4_0 230, q5_K 220, q6_K 246, q8_0 267, iq3_s 111, iq3_xxs 154, iq4_xs 173, iq4_nl 198. The I-quants and q3_K are instruction-bound and run well below the 484 GB/s of HBM2.

### 2.4 Tried and rejected code changes

| Change | Result | Reason |
|---|---|---|
| Skip the per-layer expert-ID readback for large batches | +0% with pinned memory, -27% on synthetic prompts | No gain; reverted |
| Smaller GPU submissions (`GGML_VK_SUBMIT_GFLOPS`) | not needed | No TDR with ub1024 up to 28.5K on real text; removed |
| 128-thread FA workgroups | faster but wrong results | A row group must fill whole 64-wide GCN subgroups |
| Large matmul tiles on the proprietary driver | about 15 GFLOPS | Broken on this driver |
| Decode FA split multiplier and tile sweep (Bc, workgroup, row_split) | default 32/4/4 and multiplier 2 were best | 32/8/4 was -4% at 73K but +6% at 8K, under 1% of TG |
| Occupancy knob `GGML_VK_FA_GCN_OCC_KB` (0/8/14/20) | no effect | The old decode FA is ALU/latency-bound |
| Mat-vec shader tuning for the 27B (rows per workgroup 1-8, workgroup 64/256, 36 combinations) | default was best | Latency-bound kernel |
| Rewrite of iq3_s / iq4_xs mat-vec shaders (`mi25-mmv`, unmerged) | 27B TG 10.7 -> 8.8 (hoist only: 9.35) | Holding B in registers costs occupancy, which a latency-bound kernel cannot afford |
| MoE mat-vec up to 128 tokens | -3.6% only | The mat-vec kernel loses above about 110-180 tokens |
| `multi_add_small` fusion | correct, no gain | 22.71 -> 22.70 t/s |
| Fused MoE gate/up/GLU (`mi25-fuse`) | +0.2-0.25% | Misses the 1.5% rule |
| `GGML_VK_BARRIER_MODE` 1 / 2 | no gain | 42.4 vs 42.2 ms short |
| Expert prefetch at 128K with ub2048 | no gain | GPU fault above about 15.4 GB; see 2.2.1 |
| K/V dequant shader to extend chunked attention above 32K (single GPU) | not built | Needs extra VRAM the WX does not have |
| Chunked attention on the 27B dual (22/42, 128K) | PP 27K 61 vs 48-53 | Runtime buffers left the 3070 Ti with 78 MiB free after a 32K prompt; with q8_0/q4_0 KV 112 MiB, with q5_1/q4_0 96 MiB. Moving 4 layers to the WX gave a GPU fault |
| Q4_K_M MTP on dual, experts 10-18 on CPU | faults at 119K (15.54 GB) | The MTP draft context adds 256 MiB KV and 680 MiB compute buffer |

Test coverage added for all of the above: `7db659f40`, `dd45c2a44` (FA perf cases), `f7218e0f3` (small-batch MUL_MAT_ID perf), `a196ce8e3` (27B mat-vec and Qwen3.6 MUL_MAT_ID format cases).

### 2.5 Commit list

Branch `mi25-up`, oldest first:

| SHA | Subject |
|---|---|
| `38367ffbd` | GCN tuning for MoE prompt processing |
| `011e86ad3` | Keep device buffers out of a small BAR window |
| `82973aa37` | Cap scalar flash attention at 8 block rows |
| `10596901e`, `1a1b9e87c` | Expert prefetch (off by default) |
| `7db659f40`, `dd45c2a44` | FA perf test cases |
| `e00119418`, `911fef62d` | Chunked matmul attention |
| `f7218e0f3` | MUL_MAT_ID perf cases |
| `60f4dc70a` | `GGML_SCHED_TIMING` |
| `25bbc773e` | `GGML_VK_MMVID_MAX_TOKENS`, compact multi_add, tuning knobs |
| `a196ce8e3`, `1e5e94ec9` | Test cases and decode-kernel spec |
| `5bf82b300`, `90068a274` | GCN FA decode kernel and review tests |
| `200e980d7`, `500bdf5b9`, `c8e0998dd` | Barrier diagnostics and barrier mode (no gain) |

Branch `mi25-fuse` adds `96c096fc4`, `5ec03741b`, `a98c1f09c`, `0e8ebf0d2`.
Older pre-rebase branches: `mi25-fc9`, `mi25-tg`, `mi25-mmv`, `mi25-overlap`, `mi25-fachunk`, `mi25-fa-decode`.

---

## 3. Runtime settings

The tables use the flags in the latest launchers: single `mi25-server\v14`, dual `mi25-server-dual\v9`, Hermes
`mi25-server-hermes\v9` and `mi25-server-dual-hermes\v9`, 27B `mi25-server-27b\v5` and `mi25-server-dual-27b\v3`.
"Gain" is the recorded measurement for that setting. Many settings were measured as part of a bundle and are marked so.

### 3.1 Qwen3.6 single GPU (`mi25-server\v14`)

| Flag / env | Value | Why | Measured gain |
|---|---|---|---|
| model | `Qwen3.6-35B-A3B-UD-Q4_K_M-cpuQ3.gguf` (MTP GGUF) | Experts of layers 0-20 requantized to Q3_K, so the CPU-resident experts are smaller and the CPU/PCIe path is faster | Spec replay 36.85 -> 35.02 ms/token (-5.0%, 3 rounds, with KV and ngram set as in S2); PP 3.4K 331 -> 384, 27K 292 -> 313; PPL 1.5310 vs 1.5283 (+0.12 to 0.2%) |
| `-dev Vulkan1` | WX 9100 only | Single-GPU setups keep the 3070 Ti free for the desktop and games | not applicable |
| `-ngl 99` | all layers | Everything except the CPU experts is on the GPU | |
| `--n-cpu-moe 21` | experts of the first 21 layers in host RAM | The fit limit: fewer layers device-fault the WX at load (14 and 16 faulted 3/3; 17 loaded 3/3). 73K passes at 14.46 GiB | See the correction in section 5 |
| `-c 131072` | 128K context | q5_1/q4_0 KV makes it fit | 4x the 32K context of v3 |
| `-np 1` | one slot | A single agent stream | |
| `-t 6` | 6 CPU threads | 5600X has 6 cores. Threads 4/5/6 were within noise | none |
| `-fa on` | flash attention | Needed for 128K; saves 2 GB VRAM versus FA off (13.5 vs 15.4 GB) | see 2.1.3 |
| `-ctk q5_1 -ctv q4_0` | KV cache types | Smaller KV leaves room for the CPU-expert layers' headroom. PPL cost is negligible | Spec replay +2.4% (S1), part of the S2 bundle |
| `-b 2048 -ub 2048` | batch sizes | ub1024 vs ub512 was +36% PP at 3.4K and +34% at 8.7K; ub2048 was +22% PP at 3.4K and +14% at 8.7K over ub1024 at 128K. TdrDelay is 30 s, so the old 2 s limit no longer applies | See 3.6 |
| `--load-mode none` | no mmap; CPU experts in pinned Vulkan host memory | Uploads become direct DMA instead of a staging memcpy plus a full sync | PP +10% at 3.4K, +2% at 8.7K (alone). Loading is slower and uses more committed memory |
| `--reasoning off` | no thinking | The agent tasks do not need it; reasoning with a 1024 budget was +18% slower wall time | |
| `--spec-type draft-mtp,ngram-mod` | MTP draft plus ngram-mod | MTP needs an MTP GGUF. Ngram helps copy and rewrite workloads | MTP about -21% ms/token (35.5 vs 45.3 BASE); ngram alone: 22.7 -> 30.4 t/s on a copy-with-fixes prompt |
| `--spec-draft-n-max 3` | MTP draft length | 2 was -3.5 to -4.8%; 4 and 5 gave +1.8 to +2.8% on the older single config but the S2 bundle beat the S3/S4 variants | Draft 3 kept |
| `--spec-ngram-mod-n-max 12` | ngram draft length | Short drafts verify cheaply; 8 and 24-32 were worse | 35.02 -> 33.91 ms/token (-2.3% mean, 3 rounds, no round slower) |
| `--mmproj ...mmproj-F16.gguf --no-mmproj-offload` | vision encoder on the CPU | The WX had GPU faults with the projector on the GPU; the CPU placement uses no GPU memory | Exact transcription of a test screenshot; first image 16.6 s, repeat 0.5 s; text TG unchanged; 73K passes at 14.46 GiB |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | no device buffers in the BAR window | See 2.1.2 (the fork also does it automatically) | PP +24% at 8.7K, +53% at 26.9K |
| `LLAMA_MI25_FA_CHUNK=1` | chunked matmul attention | See 2.1.4 | PP +28% at 8.7K, +61% at 27K |
| `LLAMA_MI25_FA_CHUNK_MIN_TOKENS=32` | chunked path for short turns too (default 512) | Scalar FA over the whole context dominated 76-token turns | Short turns 2.3-2.9 -> 1.2-1.4 s |
| `GGML_VK_MMVID_MAX_TOKENS=64` | per-token expert mat-vec up to 64 tokens (default 8) | Verify batches avoid the slow tiled kernel | -7% ms/token |
| `GGML_OP_OFFLOAD_MIN_BATCH=512` | batches under 512 tokens run the CPU experts on the CPU | Avoids uploading 7.8 GB of experts for small batches | Part of the short-turn result above |
| (default) `LLAMA_MI25_FA_CHUNK_MAX_KV` 32768 | chunk cap | Above 32K the K/V temporaries need VRAM the WX does not have | |

Not set (so off): `LLAMA_MI25_PREFETCH`, `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`, `GGML_VK_BARRIER_MODE`, `GGML_VK_FUSE_MOE_GLU` (the fused path is only in `mi25-fuse`).

### 3.2 Qwen3.6 dual GPU (`mi25-server-dual\v9`)

Same environment as the single setup, with these differences.

| Flag / env | Value | Why | Measured gain |
|---|---|---|---|
| model | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (MTP GGUF, plain Q4_K_M) | Requantizing on the dual layout (cpuQ3 + draft 5, D3) gave only +0.82% and failed the 1% rule | not shipped |
| `-dev Vulkan0,Vulkan1 -ts 10,30` | 3070 Ti then WX, split 1:3 | Gives the 3070 Ti about 6.2-7.0 GiB of layers | Layers 0-9 on the 3070 Ti: PP 3.4K 513 vs 297 (v8), 8.7K 450 vs 302, 27K 347 vs 212, 73K 88 vs 68; TG 26.5 vs 23.6 |
| `-ot "blk\.(1[0-9])\.ffn_(gate\|up\|down)_exps\.=CPU"` | experts of layers 10-19 in host RAM | Needed to fit the MTP draft context (+256 MiB KV, +680 MiB compute) on the WX. 73K passes with WX 15.07 GiB | MTP replay about -21% ms/token |
| `-ctk q5_1 -ctv q4_0` | compact KV | Frees VRAM so the chunked-attention cap can go to 64K | vs q8_0/q8_0: PP 60K 185 vs 143 (+29%), 119K 75.4 vs 59.1 (+28%), TG 60K 23.8 vs 19.1; PPL 1.5337 vs 1.5316 |
| `LLAMA_MI25_FA_CHUNK_MAX_KV=65536` | chunk cap 64K | Fits with the compact KV | 40K prompt +15% at the 48K step; see the v3 numbers above |
| `--spec-draft-n-max 5` | MTP draft length | Dual verify batches are cheaper | D2 -3.5% (3 rounds, 28.91 -> 27.72 ms); agents 4/4 vs 2/4 on v7 |
| `--spec-ngram-mod-n-max 12` | ngram draft length | Better than 16 | 25.47 -> 25.11 ms (-1.4%, 3 rounds) |
| `GGML_OP_OFFLOAD_MIN_BATCH` | not set | Adds nothing on dual | |
| vision | same as single (`--no-mmproj-offload`) | The WX faults with the projector on it; the 3070 Ti is already near its limit | First image 15.2 s, repeat 0.4 s, TG 42-44 on a short prompt, 3070 Ti 7.0 GiB after an image |

Other dual details:

- **One fixed exe.** Every dual launcher copies its `bin\` into `mi25-server-dual\bin` and runs from there, because the NVIDIA "Prefer maximum performance" setting is tied to that path. Without the setting (or from another path) the 3070 Ti stays in P8 during decode: 51.6 / 58.3 ms/token from another path vs 42.4 from the right one.
- **Free-VRAM guard.** The launcher exits if the 3070 Ti has less than 6300 MiB free.
- `-ts 10,30` and `-ot ... =CPU` give the 3070 Ti about 7.0 GiB at 73K (peak 7191 MiB = 7.02 GiB; limit used 7.3 GiB).

### 3.3 Hermes roots (`mi25-server-hermes\v9`, `mi25-server-dual-hermes\v9`)

Settings match the single and dual Qwen3.6 servers above, with these additions. The Hermes roots inherited the measured gains of the shared settings; they only had launch tests, not the full combined check.

| Setting | Value | Why | Measured gain |
|---|---|---|---|
| `--slot-save-path <vN>\slots` | slot save directory | Lets the server save and restore the KV of the Hermes system prompt | See below |
| Background job in the launcher | On first launch: send `hermes-warm-request.json` (the real Hermes system prompt and tools, max_tokens 1), then `POST /slots/0?action=save` with `hermes_prefix.bin`. On later launches: `POST /slots/0?action=restore` (0.5 s) | The Hermes prompt is about 13.4K tokens, stable across sessions and folders | First request after launch: 43 s -> 7.2 s (single v2), 25.8 -> 4.9 s (dual v2). With the MTP build: 13,410 tokens restored, 0.36 s single and 0.34 s dual |
| Bad-file guard | The launcher deletes a failed restore file, or a saved file with fewer than 1,000 tokens | One auto-build wrote a bad 0.3 MB file | The next launch rebuilds it |
| Dashboard | Starts `hermes dashboard` on port 9119 (localhost unless a password is set) | Convenience, not performance | |

Facts about the prefix cache:

- The slot file is now 437.5 MB (single) and 381.7 MB (dual), versus 3.2 MB and 8.6 MB for the pre-MTP build. The save appends 3 context checkpoints (188.4 MiB) and the MTP draft state (26.5 MiB). Restore was verified on both roots.
- Delete `vN\slots\hermes_prefix.bin` after a Hermes update so the next launch rebuilds it.
- Each follow-up turn processes only 200-750 new tokens, which still costs 1.3-2.2 s before the short-turn fixes in 2.1.6.
- A title-style request evicts the main prompt from the single slot, but the next main turn restores it from the server's host prompt cache (30 new tokens, 0.59 s), so `-np 2` is not needed.
- Hermes reads `n_ctx` 131072 from `/props` and compresses at 0.5 (about 64K), which matches the dual 64K chunked-attention cap. The single 32K cap would double the compressions, so it was left unchanged.

### 3.4 Qwen3.8-27B single GPU (`mi25-server-27b\v5`)

| Flag / env | Value | Why | Measured gain |
|---|---|---|---|
| model | `Qwen3.8-27B-UD-IQ3_XXS.gguf` | Smaller than Q3_K_XL (12.0 vs 13.9 GB) | TG +9-12% without spec (10.65 vs 9.75 at 3.4K), PP equal, PPL 1.4902 vs 1.4800 (within error) |
| `-dev Vulkan1` | WX only | 3070 Ti stays free | |
| `-c 32768 -fa on -ctk q8_0 -ctv q8_0` | 32K, q8_0 KV | 128K does not fit with a dense 27B on 16 GB | v1: 2x the LM Studio context (16K, FA off) at the same speed |
| `-b 2048 -ub 512` | batch sizes | ub1024 gave no improvement on the 27B dual (3.4K worse) and costs 0.7 GB | |
| `--spec-type draft-mtp,ngram-mod` | MTP plus ngram (no `--spec-draft-n-max`, so the default) | The 27B GGUF has an MTP layer. Output is identical and acceptance is about 0.99 on code copies | TG on code copy 11.9 vs about 6.0 (v2, old config); PP -25% at 3.4K and +1 GB VRAM in v2 |
| `--spec-ngram-mod-n-max 16` | ngram draft length | Replayed leftovers on the dual: MTP n3 + ngram 10.04, n2 10.07, n4 10.03, MTP alone 10.11: already at the optimum | |
| `--mmproj mmproj-Qwen3.8-27B-BF16.gguf --no-mmproj-offload` | vision on the CPU | The WX has no BF16 | Exact transcription; first image 34.0 s, repeat 1.2 s; TG 12.7; 27K passes with WX 13.18 GiB |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | as above | | |
| `--load-mode none` | not set | Dense model, no CPU-resident experts | |
| `LLAMA_MI25_FA_CHUNK`, `GGML_VK_MMVID_MAX_TOKENS` | not set | Chunked attention is off on the single 27B; MMVID 64 and the FA chunk were not better (B2 mean +11.45% but worst size -15.97%; B3 mean -7.58%) | |

### 3.5 Qwen3.8-27B dual GPU (`mi25-server-dual-27b\v3`)

| Flag / env | Value | Why | Measured gain |
|---|---|---|---|
| `-dev Vulkan0,Vulkan1 -ts 26,38` | layers split 26 : 38 | The 3070 Ti holds about 26 layers (about 161 MB of weights per layer plus 17 MB of KV per 32K). Splits 28,36 and 30,34 pushed the 3070 Ti peak to 7.6-7.9 GiB, over the 7.3 GiB limit, and did not win | IQ3_XXS 26,38 vs Q3_K_XL 22,42: PP +6-8%, TG at 27K +15-20%, WX VRAM 12.87 vs 14.17 GiB |
| `-c 131072 -ctk q8_0 -ctv q8_0` | 128K | The split leaves room on the WX | |
| `-b 2048 -ub 512` | | ub1024: PP 114 / 73 / 52.5 vs ub512 133 / 76 / 51.5, 0.7 GB more VRAM | no gain |
| `--load-mode none` | set | | |
| `--spec-type draft-mtp,ngram-mod --spec-ngram-mod-n-max 16` | | | Dual vs single v2: TG 9.7 / 10.4 / 9.1 vs 5.4 / 4.9 / 4.3 (pre-fan) |
| vision | BF16 mmproj on the CPU | | First image 27.1 s, repeat 1.0 s, TG 15.6; 73K passes with WX 12.87 GiB, 3070 Ti 6.75 GiB |
| Guard | needs 6200 MiB free on the 3070 Ti; uses the shared exe `mi25-server-dual\bin` | | |

### 3.6 Batch size history

| Step | Config | Result |
|---|---|---|
| ub512 -> ub1024 (v1) | real text | PP +36% at 3.4K, +34% at 8.7K |
| ub1024 -> ub2048 (v6) | 128K q8_0 | PP 3.4K 322 vs 263 (+22%), 8.7K 273 vs 240 (+14%), 27K 133 vs 130; no fault up to 119K; VRAM 14.8-14.86 GB |

---

## 4. Settings tried and rejected

| Setting | Config | Result | Reason |
|---|---|---|---|
| ngram defaults (simple / map-k / k4v / mod / cache, 3-token drafts) | Qwen3.6 early | 13.3-20.8 t/s vs 22.7 none on copy | Each verify pass costs nearly a full step; acceptance about 25% |
| `--spec-ngram-mod-n-max 32` | Qwen3.6 | 18.5 vs 30.4 for n-max 16 on a copy prompt; also 24.6 / 23.8 in the agent replay | Longer drafts waste verify work. n-max 24 stalled mid-generation in one replay (pre-existing issue) |
| ngram n-max 8 | single S2 / dual D2 | single +0.75% (fail); dual -1.48% (fail) | |
| Draft-model speculation (`unsloth/Qwen3.5-0.8B-GGUF` Q8_0) | single, dual | 12.3 t/s vs 22.1 BASE single; 6.1 vs 28.5 dual; draft 16 stalled or timed out | Slower on every config and placement; dual draft on the CPU ran at about 4 tokens/s |
| IQ4_XS MTP GGUF | single | 15-18% slower decode; PP at 27K 7-10% lower; needs ncm17 | The iq4_xs mat-vec is instruction-bound on gfx900 |
| Q4_K_M MTP on the old dual layout | dual | Layout faulted; TG better but agents 0/3 on review-fixes | Not shipped at the time. MTP was shipped later on the rebased build (single v12, dual v7) after retuning |
| `--spec-draft-n-max` 2 / 4 / 5, MTP alone | single v12, dual v7 | single n4 +2.2 / +1.8, n5 +2.4 / +2.8, alone +0.2 / 0.0; dual n5 +4.3 / +4.1 (later shipped as D2) | Under the old 3% bar for single. On dual n5 shipped under the 1% rule |
| Fewer CPU expert layers (ncm15 / ncm14 at v4, 128K) | single | +2-7% PP for +0.45 / +0.85 GB VRAM | Clashes with VRAM headroom |
| `--n-cpu-moe` 10 or 14 on single | single | Standalone loads device-faulted the WX | See section 5; the labelled tests had actually run at 21 |
| CPU threads (`-t 4`, `-t 5`, `-t 6 -tb 12`, `-t 4 -tb 12`), poll, priority, strict CPU | single, dual | All within noise (best -0.3% to -1.0%) | |
| Qwen3.6 Q3_K_XL (17.2 GB) | single, dual | single -0.5 / -0.9% (n-cpu-moe 16, replay), PP +14 to +21%; dual zero-CPU splits fail | Replay speed did not improve; dual needs 4 CPU layers |
| IQ4_XS / IQ3_XXS / q3_K / q6_K etc. as a new Qwen3.6 GGUF | MUL_MAT_ID microbench | No format reached 1.3x Q4_K | No GGUF proposal made |
| Requantizing CPU experts on the dual layout (D3) | dual | +0.82% mean | Fails the 1% rule |
| `GGML_OP_OFFLOAD_MIN_BATCH=1024` | dual | no gain; 580-800-token turns worse | 512 on single only |
| Reasoning on with `--reasoning-budget 1024` | dual Hermes | 6/6 pass, but wall 101.6 s vs 86.3 s | The tasks do not need thinking |
| KV types for decode (q4_0, q4_1, q5_1, f16) | single | q4_1 only 7% faster than q8_0 at 73K | Not worth shipping; f16 and q4_1 are not handled by the new decode kernel |
| 27B Q3_K_XL -> Q4_K_M on dual | dual 27B | With q8_0 KV faults at load; with compact KV 73K TG falls to 5.25 | Does not fit / worse at long context |
| Dual 27B splits 28,36 and 30,34 | dual 27B | Mixed per-size TG, 3070 Ti peaks at 7.6-7.9 GiB | Over the 7.3 GiB limit; noise 15-30% round to round |
| Dual 27B KV q8_0 / q4_0 or q5_1 / q4_0 to fit chunked attention | dual 27B | 3070 Ti 96-112 MiB free after a 32K prompt | Runtime buffers grow into whatever is freed |
| Dual 27B `-ub 1024` | dual 27B | no gain, +0.7 GB | |
| `-sm tensor` (tensor split) | dual Qwen3.6 | Server crashes on the first prompt (connection reset); `llama_params_fit` and backend sampling are unsupported with it | Unsupported |
| Row split (`-sm row`) | dual | Not run. A code read showed that it fails at model load ("does not support split buffers"): the row path needs a backend split buffer type, and neither ggml-vulkan nor ggml-cuda exports one in this tree | Unsupported |
| mmproj on the GPU (WX or 3070 Ti) | single, dual | Device lost / GPU fault at load | The projector goes on the CPU (`--no-mmproj-offload`). The single-GPU control without any projector also faulted at that time, so the WX state was suspect |
| ub2048 + prefetch at 128K | single | PP 3.4K about 340 but VRAM 15.45 GB | GPU fault above about 15.4 GB |
| 27B dual draft/other: `--spec-draft-n-max` 2 / 4 | dual 27B | 10.07 / 10.03 vs 10.04 | Already at the optimum |
| GPU keep-alive to avoid clock ramp | single | no gain | Clock-ramp theory not supported |
| Agent-side changes: `search` tool, trimmed prompt, loop guard | local agent | No measurable gain on the 2-task suite | Not part of the server; reverted |
| IQ4_XS MTP on the dual layout | dual | Mixed, worse at 27K (PP 327-339 vs 347-416) | |
| Hermes: compression threshold change, `-np 2` or an auxiliary model | Hermes | Not needed | Prefix restore from the host prompt cache works |
| 27B vision on the GPU | 27B | Not possible | The WX has no BF16 |

---

## 5. Lessons and gotchas

1. **Thermals change the answer.** An under-cooled WX lost about 50% matmul speed and about 3x flash-attention speed when hot
   (27B TG 10.7 cold vs 7.5-8 hot). Several "dead ends" (prompt FA investigation, 27B at 73K TG 2.1) were thermal.
   The new fan recovered most of it: 27B PP at 8.7K / 27K +70% / +65%; the 27B dual 73K TG went from 2.1 to 7.9. Use a cool-down (120-300 s) between benchmark arms.
2. **Synthetic prompts mislead.** A repeated 10-word prompt used only a few MoE experts, so only those uploaded.
   With real text almost every expert is uploaded: PP at 3.4K is about 203 t/s, not 326. All numbers in this document use real C++ text.
3. **The driver faults at about 15.4 GB.** The WX has 16 GB of HBM2, but the AMD driver raises a GPU fault / device lost above about 15.4 GB
   (16 GB minus the Windows share). It cannot be raised. This is why 128K f16 KV, ub2048 + prefetch at 128K, MTP at `--n-cpu-moe` 14-16, and several 27B layouts failed.
   Check peak VRAM after a long prompt (73K or 119K), not just after load.
4. **Repeated `--n-cpu-moe` flags add up.** `--n-cpu-moe N` does not replace an earlier value; each flag adds overrides for the first N layers (`common/arg.cpp`), so the larger value wins.
   The test harness passed `--n-cpu-moe 21` and then each arm's own value, so single-GPU arms labelled 10 / 14 / 16 / 17 / 18 actually ran at 21.
   Standalone loads with only 14 or 16 device-faulted the WX (3/3 each); 17 loaded. The measured gains stand for 21. The launchers use 21.
5. **Single-GPU servers leave the 3070 Ti free.** The 3070 Ti drives the desktop and games. Single-GPU launchers use `-dev Vulkan1` only, and the vision work never put the encoder on it.
   Dual launchers check for the 3070 Ti and for 6200-6300 MiB free before starting.
6. **Dual servers must run from one exe path.** NVIDIA's per-program "Prefer maximum performance" setting is tied to `mi25-server-dual\bin\llama-server.exe`.
   Without it the 3070 Ti sits in P8 (210 MHz core, PCIe gen1) during decode, because decode bursts are too short to clock it up, and TG drops from 23 to about 14.
7. **Noise is 5-8% between rounds.** Short-context TG on the dual 27B swings 15-30% round to round. Use interleaved rounds, a cool-down, and a margin of at least the rule you set (3%, 5%, later 1% with gains that must show in every round).
   Several apparent wins were an outlier in the baseline (for example 28.83 and 29.45 ms in the v7 base).
8. **Row split is unsupported on Vulkan, and tensor split crashed.** Layer split with `-ts` is the working multi-GPU mode.
9. **Environment variable names.** `LLAMA_MI25_PREFETCH` is tested for presence, so `=0` turns it on. `--draft-max` / `--draft-min` were replaced by `--spec-draft-n-max` / `--spec-draft-n-min` in this build.
10. **The Hermes prefix cache is a file, not a flag.** It needs `--slot-save-path` and the launcher job. Regenerate it after a Hermes update or a build change. The slot file size depends on the build (3 MB without MTP, 380-440 MB with it).
11. **Leftover processes cause false alarms.** The early "GPU fault on real text" report came from a leftover benchmark script that had loaded a second model onto the GPU. Check for stray `llama-server` and bench processes before measuring.
12. **Upstream drift.** The original base `f74e2fa` was 3% slower at 4K and 7% slower at 12K than `fc9ce6b` (not bisected). The fork later rebased onto `f5222f25c`; the 27B gained PP +30% from that rebase.
13. **Agent task results are noisy.** review-fixes passed 3 of 5 on one dual config and 1 of 2 on others. Treat a single run as a trajectory, not a quality signal.
14. **Some things were never measured.** Not measured: 27B vision on a live Hermes task, IQ3_XXS on the dual 27B root at the time of the Q3_K_XL comparison (it was done in the next round), the cause of the ~2x in-model vs isolated mat-vec gap,
    why 30,34 dual 27B is slow when it runs first in a round, Windows reserve behaviour on the WX, a WX temperature or clock reader (none installed).
    The FA decode kernel gate to KV >= 16K and a single-pass gate+up fusion were estimated, not built.

### Open levers

- Rewrite or accelerate the small-batch `MUL_MAT_ID` kernel (the tiled kernel reads expert weights at about 37 GB/s at 64-256 tokens). Not attempted; invasive.
- A chunked-attention extension above 32K on a single GPU (needs a q8_0 -> f16 copy shader). Not done.
- A GPU power or clock setting for the in-model mat-vec gap. It needs a driver setting and the user's decision.
