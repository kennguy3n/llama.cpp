# Phase 1c: Design brief — AVX2 / AVX-VNNI repack for `Q1_0_g128`

**Status**: Design only. **No source files outside `docs/` will be modified until this brief is signed off.**
**Audience**: Ken (reviewer / sign-off).
**Goal**: Concrete, file-level, line-level proposal for a runtime AVX2 + AVX-VNNI repack of `block_q1_0_g128`, mirroring the existing `Q4_0` template documented in `01-survey.md` and instantiated against the `Q1_0_g128` layout documented in `02-q1-layout.md`.

## Decisions taken (per `kennguy3n/cv-guard#15` discussion)

| # | Decision                                                                                                                                                              | Source                                       |
|--:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| 1 | The on-disk GGUF format is **unchanged**. Existing `Bonsai-1.7B.gguf` (and any future `Q1_0_g128` model) is binary-compatible with this work.                          | mainline pattern, `01-survey.md §1, §5`      |
| 2 | The fork is private; this work is implemented directly (no human-author requirement). It's still committed to a feature branch and reviewed before merging to `prism`. | Ken's answer to question 1                  |
| 3 | Two kernel branches are written: AVX2-only (Zen 3, Skylake-X-no-VNNI, etc.) **and** AVX-VNNI (Alder Lake P-cores, Zen 4).                                              | Ken's answer to question 2                   |
| 4 | The packed in-memory block may grow to include precomputed `sum(qy)`-friendly fields if it accelerates the inner loop. On-disk size unaffected.                       | Ken's answer to question 3                   |
| 5 | The new packed type is **runtime-only** — it never appears in `ggml.h`'s `ggml_type` enum, never appears in any GGUF, and never crosses an ABI boundary.                 | mainline pattern; Q4_0_4_4 etc. work the same way |

## Layout: `block_q1_0_g128x4`

```cpp
// repack.h (additions)

// Four block_q1_0_g128 weight rows interleaved for 4-way GEMV on AVX2.
// Total size: 4 deltas (8 B) + 4 × 16 quant-bytes (64 B) = 72 B,
// matching exactly 4 × sizeof(block_q1_0_g128) = 4 × 18 = 72 B.
// On-disk size of the underlying GGUF is therefore unchanged; this is
// a pure in-memory permutation.
struct block_q1_0_g128x4 {
    ggml_half d[4];   // deltas for the 4 interleaved q1_0_g128 rows
    uint8_t   qs[64]; // 4 × 16 packed weight bytes, interleaved at 4-byte granularity
};
static_assert(sizeof(block_q1_0_g128x4) ==
              4 * sizeof(ggml_half) + 4 * (QK1_0_g128 / 8),
              "wrong q1_0_g128x4 block size/padding");
```

### Quant byte interleave granularity

Recall that the AVX2 inner loop on the un-repacked path consumes the 16 quant bytes of a single row in **four 4-byte chunks** (one chunk per `q8_0` sub-block of 32 elements; `for (int k = 0; k < 4; ++k)`). The natural interleave granularity is therefore **4 bytes** — exactly one `uint32_t qbits32` per row per sub-block.

With four interleaved rows the quant section is laid out:

```
Offset   Content
─────────────────────────────────────────────────────────
 0..3   row0 sub-block0 (32-bit qbits)
 4..7   row1 sub-block0
 8..11  row2 sub-block0
12..15  row3 sub-block0
16..19  row0 sub-block1
20..23  row1 sub-block1
24..27  row2 sub-block1
28..31  row3 sub-block1
…
```

This packs neatly into one **128-bit broadcast load** per sub-block: `_mm_loadu_si128((__m128i *)&block.qs[16*k])` produces a 16-byte register holding the four `qbits32` values for sub-block `k` of all four rows.

### Why no per-block `sum(qy)` field after all

Decision #4 left the door open for adding a precomputed `sum(qy)` to the packed block. After working through the kernel (§4), it's clear `sum(qy)` is per-**activation**-block, not per-**weight**-block — adding it to `block_q1_0_g128x4` would duplicate it across every weight row that consumes the same activations.

The right place for `sum(qy)` is alongside the activation, in `block_q8_0` itself. The mainline `block_q8_K` struct already carries a `bsums` field for exactly this purpose (`repack.h:78`). Two follow-on options:

- **Option A**: extend `block_q8_0` with a `sum_qs` field. Cheap (4 B per 34 B activation block, ~12 % bandwidth growth). Disadvantage: every existing kernel that consumes `block_q8_0` has to be ABI-checked.
- **Option B**: introduce a new `block_q8_0_sum` activation type, used only when the matched weight is `Q1_0_g128`-repacked. Mirrors how `block_q8_K` exists alongside `block_q8_0`.

For Phase 3 the first kernel I'll write uses **neither** — `sum(qy)` is computed on the fly using `_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), qy), _mm256_set1_epi16(1))` and **hoisted out of the per-row loop**, since the 4 rows share the same activation block. That alone captures the `2·sum(masked) − sum(total)` win without any block-format changes.

If microbenchmarks (Phase 2) show meaningful upside from caching `sum(qy)` across activation tokens, we revisit Option B. Until then, **the activation block is also unchanged**.

## Repack helper

```cpp
// repack.cpp (additions)

static block_q1_0_g128x4 make_block_q1_0_g128x4(block_q1_0_g128 * in) {
    block_q1_0_g128x4 out;
    for (int r = 0; r < 4; r++) {
        out.d[r] = in[r].d;
    }
    // Interleave at 4-byte granularity: one sub-block of one row at a time.
    // Total 4 sub-blocks × 4 rows × 4 bytes = 64 bytes.
    for (int k = 0; k < 4; k++) {        // sub-block index
        for (int r = 0; r < 4; r++) {    // row index
            memcpy(&out.qs[k*16 + r*4],
                   &in[r].qs[k*4],
                   4);
        }
    }
    return out;
}

static int repack_q1_0_g128_to_q1_0_g128_4_bl(struct ggml_tensor * t,
                                              const void * GGML_RESTRICT data,
                                              size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0_g128);
    GGML_ASSERT(t->ne[1] % 4 == 0);
    /* exact mirror of repack_q4_0_to_q4_0_4_bl, but reading 4 block_q1_0_g128
       rows at a time instead of 4 block_q4_0 rows. */
    const block_q1_0_g128 * src = (const block_q1_0_g128 *)data;
    block_q1_0_g128x4 *     dst = (block_q1_0_g128x4 *)t->data;

    const int64_t nrows  = t->ne[1];
    const int64_t nblocks_per_row = t->ne[0] / QK1_0_g128;

    for (int64_t r = 0; r < nrows; r += 4) {
        for (int64_t b = 0; b < nblocks_per_row; b++) {
            block_q1_0_g128 group[4];
            for (int rr = 0; rr < 4; rr++) {
                group[rr] = src[(r + rr) * nblocks_per_row + b];
            }
            *dst++ = make_block_q1_0_g128x4(group);
        }
    }
    return 0;
}
```

## GEMV kernel A — AVX2-only

The kernel computes one matrix-vector product per call: 4 interleaved weight rows × 1 activation column. The key invariants:

- One `_mm_loadu_si128` of the 16-byte interleaved-qbits broadcast per sub-block (`k`), giving us all 4 row-masks at once.
- One `_mm256_loadu_si256` of `qy` per sub-block, **shared** across the 4 rows.
- One `sum(qy)` partial-sum vector per sub-block (`sum_qy_v`, kept as `__m256i`), **shared** across the 4 rows.
- Per-row inside the sub-block: one `_mm256_blendv_epi8` (mask-driven select between zero and qy) + one `maddubs+madd` to a per-lane partial-sum vector + one fused vector `2·partial − sum_qy` + one `fmadd` into the row accumulator. **No horizontal sums in the inner loop**: partial sums stay vectorised end-to-end and are only reduced once per row at write-out time.

```cpp
// arch/x86/repack.cpp (additions)

void ggml_gemv_q1_0_g128_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                                  const void * GGML_RESTRICT vx,
                                  const void * GGML_RESTRICT vy,
                                  int nr, int nc) {
#if defined(__AVX2__) && !defined(__AVX512F__)
    {
        const block_q1_0_g128x4 * x = (const block_q1_0_g128x4 *)vx;
        const block_q8_0 *        y = (const block_q8_0 *)vy;

        const int nb = n / QK1_0_g128;
        // Lookup table for "expand 32 bits → 32 bytes of 0x00/0xFF" via shuffle+and+cmpeq.
        const __m256i shuffle_mask = _mm256_set_epi8(
            3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,
            1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
        const __m256i bit_mask = /* same per-byte bit pattern as quants.c */;

        for (int col = 0; col < nc; col++) {
            __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(),
                              _mm256_setzero_ps(), _mm256_setzero_ps() };

            for (int ib = 0; ib < nb; ib++) {
                const block_q1_0_g128x4 & xb = x[col * nb + ib];

                // 4 row deltas in a single 64-bit half-vector.
                const __m128 d0 = _mm_cvtph_ps(_mm_loadu_si64((const __m128i *)&xb.d[0]));

                for (int k = 0; k < 4; k++) {
                    const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib*4 + k].qs);
                    const float   d_y = GGML_CPU_FP16_TO_FP32(y[ib*4 + k].d);

                    // sum(qy) once per sub-block (shared across 4 rows). Stays a per-lane
                    // partial-sum vector — no horizontal-sum in the inner loop.
                    const __m256i sum_qy_v = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_set1_epi8(1), qy),
                        _mm256_set1_epi16(1));

                    // 4 row qbits32, broadcast-loaded as a single 16-byte register.
                    const __m128i qbits_4rows = _mm_loadu_si128((const __m128i *)&xb.qs[k*16]);

                    for (int r = 0; r < 4; r++) {
                        // Lane-extract this row's 32 bits.
                        const uint32_t qbits32 =
                            (uint32_t)_mm_extract_epi32(qbits_4rows, r);

                        // Bit-expansion to 0xFF mask (same 5-op sequence as quants.c).
                        const __m128i qbits_128 = _mm_set1_epi32(qbits32);
                        const __m256i qbits_256 = _mm256_broadcastsi128_si256(qbits_128);
                        const __m256i shuffled  = _mm256_shuffle_epi8(qbits_256, shuffle_mask);
                        const __m256i tested    = _mm256_and_si256(shuffled, bit_mask);
                        const __m256i mask_ff   = _mm256_cmpeq_epi8(tested, bit_mask);

                        // Mask-select qy into "qy where bit=1 else 0", reduce to per-lane
                        // partial-sum vector (no horizontal-sum yet).
                        const __m256i masked    = _mm256_blendv_epi8(_mm256_setzero_si256(), qy, mask_ff);
                        const __m256i partial_v = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(_mm256_set1_epi8(1), masked),
                            _mm256_set1_epi16(1));

                        // dot(±1, qy) = 2·partial_v − sum_qy_v, kept per-lane.
                        // Lanes carry distinct partial sums (madd_epi16 reduces 16-bit pairs);
                        // the final hsum_float_8 at write-out time combines them correctly.
                        const __m256i row_dot_v = _mm256_sub_epi32(
                            _mm256_slli_epi32(partial_v, 1),
                            sum_qy_v);
                        const __m256  row_dot_f = _mm256_cvtepi32_ps(row_dot_v);
                        const float   d_row     = ((float *)&d0)[r];

                        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d_row * d_y),
                                                 row_dot_f,
                                                 acc[r]);
                    }
                }
            }

            // Write 4 row outputs.
            for (int r = 0; r < 4; r++) {
                s[col * nr + r] = hsum_float_8(acc[r]);
            }
        }
        return;
    }
#endif

    ggml_gemv_q1_0_g128_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
```

Inner-loop op count per 32 weights, per row (after hoisting `sum_qy_v`, `qy`, and `qbits_4rows` out of the per-row loop; horizontal sums removed from the hot path):

| Op                                    | Count |
|---------------------------------------|------:|
| Bit expansion (broadcast+shuffle+and+cmpeq, 5 ops) | 5 |
| `blendv_epi8`                         | 1     |
| `maddubs_epi16 + madd_epi16` (vector partial sum) | 2 |
| `slli_epi32 + sub_epi32` (vector `2·partial − sum_qy`) | 2 |
| `cvtepi32_ps + fmadd_ps` (accumulate into row vector) | 2 |
| **Total**                             | **~12** |

Because `qy`, `sum_qy_v`, and the 4-row `qbits` block are loaded **once** per sub-block but reused across 4 rows, the **amortised** cost per 32-weights-per-row is `~10/4 (per-row body) + 0.5·(shared loads + sum_qy) ≈ 3.5–4 ops` — close to the AVX-512 path's 3 ops. The single horizontal-sum (`hsum_float_8`) per row is paid once at the end of the column, **outside** the `nb`-deep inner loop, so it amortises to zero across the prefill.

**Expected speedup over the un-repacked AVX2 single-row path: ~2.5×, i.e. ~50–60 t/s prefill on the EPYC 7763 demo VM.**

## GEMV kernel B — AVX-VNNI

Same skeleton as kernel A, with two changes:

1. Bit-expansion stays the same (AVX-VNNI doesn't add mask registers; it's an AVX-encoding of `vpdpbusd` and friends).
2. The signed-dot-product epilogue collapses to a single `_mm256_dpbusd_avx_epi32`:

```cpp
#elif defined(__AVXVNNI__) && !defined(__AVX512F__)
    /* same outer structure, but inside the per-row block: */
    const __m256i neg_qy   = _mm256_sub_epi8(_mm256_setzero_si256(), qy);
    const __m256i signed_qy = _mm256_blendv_epi8(neg_qy, qy, mask_ff);
    const __m256i int_acc   = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(),
                                                      _mm256_set1_epi8(1),
                                                      signed_qy);
    acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d_row * d_y),
                             _mm256_cvtepi32_ps(int_acc),
                             acc[r]);
```

Inner-loop op count per 32 weights drops from ~10 to ~7 (saves the maddubs+madd+hsum, replaces with one dpbusd). On Alder Lake P-cores / Zen 4, **expected throughput is within ~15 % of the AVX-512 fast path**.

## GEMM kernel

Identical structure to GEMV but processes multiple activation columns per call. Mainline `Q4_0` ships both a GEMV (`nr=1`) and GEMM (`nr>1`) variant; we should match that for batched prefill. The GEMM kernel can hoist the `sum_qy` / `qbits_4rows` loads even further (across both rows and columns).

For this brief I'm stubbing the GEMM kernel as `ggml_gemm_q1_0_g128_4x4_q8_0` and committing to writing it in Phase 3 alongside the GEMV. Test parity (output = un-repacked vec_dot for the same inputs) is the gating criterion.

## Dispatcher branch

`ggml/src/ggml-cpu/repack.cpp:3392`, after the existing `Q4_0` block:

```cpp
} else if (cur->type == GGML_TYPE_Q1_0_g128) {
    // AVX-VNNI branch — fires on Alder Lake P-cores / Zen 4 (no AVX-512).
    if (ggml_cpu_has_avx_vnni() && !ggml_cpu_has_avx512()) {
        if (cur->ne[1] % 4 == 0) {
            return &q1_0_g128_4x4_q8_0_avxvnni;
        }
    }
    // AVX2-only branch — fires on Zen 3 / Skylake-X-no-VNNI / older EPYC.
    if (ggml_cpu_has_avx2() && !ggml_cpu_has_avx512() && !ggml_cpu_has_avx_vnni()) {
        if (cur->ne[1] % 4 == 0) {
            return &q1_0_g128_4x4_q8_0_avx2;
        }
    }
    // AVX-512 hosts fall through; the existing fast path in
    // arch/x86/quants.c:855 outperforms any GEMV we can write.
}
```

Two `tensor_traits` instances are added at the top of the function (mirroring the existing `q4_0_*` entries):

```cpp
static const tensor_traits<block_q1_0_g128, 4, 4, GGML_TYPE_Q8_0, /*kernel=*/AVX2>
    q1_0_g128_4x4_q8_0_avx2;
static const tensor_traits<block_q1_0_g128, 4, 4, GGML_TYPE_Q8_0, /*kernel=*/AVX_VNNI>
    q1_0_g128_4x4_q8_0_avxvnni;
```

(The `kernel=` template parameter is a small additive change to the existing `tensor_traits` template so we can dispatch to two kernels of the same shape. This is a five-line edit in `traits.h` and is the only piece of plumbing this design adds outside the new files.)

## Test plan

Three test categories, all runnable on the EPYC 7763 demo VM:

1. **Bit-identical output vs. un-repacked vec_dot.** A new test in `tests/test-quantize.cpp` (or the existing repack tests) repacks a synthetic `Q1_0_g128` tensor, runs both `ggml_gemv_q1_0_g128_4x4_q8_0` and `ggml_vec_dot_q1_0_g128_q8_0` on the same input, and asserts every output float matches within `1e-5`. Must pass on AVX2-only **and** AVX-VNNI hosts.
2. **End-to-end with Bonsai-1.7B.** Load `models/slm/Bonsai-1.7B.gguf` through `llama-cli`, run a fixed prompt, capture log-probs of the first 32 tokens. Repeat with the AVX-512 build. Diff the log-prob arrays — any deviation > `1e-3` per token is a kernel bug.
3. **Cross-arch fallback testing.** Run the tests under `GGML_CPU_GENERIC=1`, on a NEON Arm64 host (CI), and (if available) on an AVX-512 box. Confirm none of those branches see the new code path, and confirm the new generic fallback kernel gives bit-identical results to the un-repacked vec_dot.

## Microbenchmark plan (Phase 2)

A standalone C program at `tools/microbench/q1_g128_repack.cpp`:

- Allocates 1024 weight rows × 1024 elements (8 KiB worth of `Q1_0_g128` blocks, fits in L1).
- Runs each kernel variant 10⁵ times over the same buffer.
- Reports cycles/element using `__rdtsc`, plus instruction count via `perf stat`.
- Variants: (a) un-repacked AVX-512, (b) un-repacked AVX2 single-row, (c) repacked AVX2 4×4, (d) repacked AVX-VNNI 4×4. Variants a & b are baselines; c & d are the new code.

Phase 2 deliverable is the harness + a baseline-only run on this VM (just a & b), so we have ground-truth numbers before any new kernel exists.

## Implementation phases (after sign-off)

| Phase | Deliverable                                                                                           | Touches                                                                |
|-------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| 2     | Microbenchmark harness + baseline a & b numbers on Zen 3                                              | `tools/microbench/q1_g128_repack.cpp` (new); `CMakeLists.txt` (1 line) |
| 3a    | `block_q1_0_g128x4` + `make_block_q1_0_g128x4` + `repack_q1_0_g128_to_q1_0_g128_4_bl`                  | `repack.h`, `repack.cpp`                                               |
| 3b    | `ggml_gemv_q1_0_g128_4x4_q8_0` (AVX2 + AVX-VNNI) and `_generic` fallback                              | `arch/x86/repack.cpp` (AVX2 + AVX-VNNI); `repack.cpp` (generic)        |
| 3c    | Dispatcher branch in `ggml_repack_get_optimal_repack_type`                                            | `repack.cpp:3442` insertion                                            |
| 3d    | `tensor_traits` template kernel-tag plumbing                                                          | `traits.h` (5-line addition)                                           |
| 4     | GEMM variant (`ggml_gemm_q1_0_g128_4x4_q8_0`) and dispatcher hookup                                   | same files as above                                                    |
| 5     | Tests (parity, e2e, cross-arch) and microbenchmark numbers                                            | `tests/`, `docs/avx2-repack-design/04-results.md`                      |
| 6     | cv-guard PR follow-up: no source changes; just rerun the benchmark and update `slm-benchmark-results.md` | `kennguy3n/cv-guard` repo only                                         |

Total estimated effort with the design ratified: **5–7 working days**, roughly half on the GEMV kernel and half on the GEMM + tests + microbench harness.

## Sign-off checklist for Ken

Tick yes / no:

- [ ] **Layout of `block_q1_0_g128x4`** as proposed in §"Layout" (4 deltas + 4 × 16 quant bytes, interleaved at 4-byte granularity, 72 B total)?
- [ ] **`sum(qy)` is computed on the fly inside the kernel and not added to the activation block** (deferred to Option B in a later phase if microbenchmarks justify it)?
- [ ] **Two kernel branches**, AVX2-only **and** AVX-VNNI, sharing the same packed-block layout, dispatched by `ggml_cpu_has_avx_vnni()` and `ggml_cpu_has_avx2()`?
- [ ] **AVX-512 hosts continue to use the existing `arch/x86/quants.c:855` fast path** (no GEMV repack on AVX-512)?
- [ ] **No on-disk GGUF format changes**, no `ggml_type` enum changes, no `ggml.h` ABI changes?
- [ ] **Test plan** as proposed in §"Test plan" (parity vs. existing vec_dot + e2e log-prob diff + generic fallback)?
- [ ] **Microbenchmark harness lands first (Phase 2)** before any new kernel is written, so we have honest baseline numbers?

If all seven boxes can be ticked I'll move to Phase 2 immediately. If any deviate, I'll revise the brief and resubmit.
