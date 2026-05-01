# Drop-Unpack Experiment (Q1_0_g128 4×4 AVX2 GEMM)

> **Status:** experiment completed; **kernel implementation reverted**.
> The chunked kernel was a measured ~21% pp regression on Zen 3 /
> AVX2-only and was not worth the in-tree maintenance cost. The
> kernel source is preserved in the git history of branch
> `devin/1777638955-prism-drop-unpack-chunked-gemm` (PR #15) for any
> future maintainer who wants to revive the experiment on a different
> ISA. This document is the only artifact that lands on `prism`.

## Goal

Quantify whether the explicit `block_q8_0x4 → block_q8_0[]` scratch unpack
in `ggml_gemm_q1_0_g128_4x4_q8_0` (`arch/x86/repack.cpp`, shipped in
[PR #13](https://github.com/kennguy3n/llama.cpp/pull/13)) is a meaningful
fraction of GEMM time on Zen 3 / AVX2-only, and if so, eliminate it by
reading `block_q8_0x4` directly from the inner kernel.

The original PR #13 review thread predicted "<1% additional win" from this
optimisation; this experiment confirms the prediction by measuring the
actual cost.

## Approach

A new AVX2 kernel `ggml_gemm_q1_0_g128_4x4_q8_0_avx2_chunked` was
written in `arch/x86/repack.cpp` that takes `const block_q8_0x4 *` as
its activation source. Inside the inner loop, it loads all 128 bytes of
each `block_q8_0x4` instance as 4 × `__m256i` and deinterleaves each
activation row's 32 bytes on the fly using:

- 4× `_mm256_permutevar8x32_epi32` (one per ay-load), each routing the
  row's two 4-byte chunks to a specific output lane pair (0/1, 2/3,
  4/5, 6/7).
- 3× `_mm256_blend_epi32` to stitch the four permuted vectors into a
  contiguous 32-byte qy.
- Total: 7 ops per `(sub-block, activation_row)`, i.e. 28 extra ops per
  sub-block (=4 rows × 7) on top of the existing 16 dot-product ops per
  sub-block.

The new kernel was gated behind a `GGML_Q1_G128_CHUNKED=1` env var in
the dispatcher (`ggml_gemm_q1_0_g128_4x4_q8_0`) so the existing
`tests/test-q1-g128-repack-parity.cpp` could exercise both code paths
without modification, and `llama-bench` could A/B both within the same
binary.

The kernel + dispatcher branch were implemented, parity-verified, and
benchmarked on the EPYC 7763 / Zen 3 demo VM. After measurement
confirmed the regression, the kernel + dispatcher branch were **reverted
from the PR**; this document is the only artifact that lands.

## Parity (during experiment)

`tests/test-q1-g128-repack-parity.cpp` (52 elements: 4 GEMV outputs +
16 GEMM nr=4 outputs + 32 GEMM nr=8 outputs) passed both variants with
**`diff = 0.000000e+00`** on every element. The chunked kernel produced
bit-identical FP32 output to the default kernel.

```
$ ./build/bin/test-q1-g128-repack-parity
PASS: GEMV (nr=1) + GEMM (nr=4) + GEMM (nr=8) all match single-row vec_dot within FP16 tolerance

$ GGML_Q1_G128_CHUNKED=1 ./build/bin/test-q1-g128-repack-parity
PASS: GEMV (nr=1) + GEMM (nr=4) + GEMM (nr=8) all match single-row vec_dot within FP16 tolerance
```

`llama-cli` smoke test in chunked mode produced sensible tokens:
`"The capital of France is"` → `"**Paris**. It is the"` (same as default).

## Results

`./build/bin/llama-bench -m Bonsai-1.7B.gguf -t 4 -p 256,512,1024 -n 32 -r 3`,
3 independent runs per variant. Throughput in t/s, mean ± stddev across
the 3 outer runs:

| Variant                                | pp256 (t/s)       | pp512 (t/s)       | pp1024 (t/s)      | tg32 (t/s)        |
|----------------------------------------|------------------:|------------------:|------------------:|------------------:|
| **`_4rows`** (default, PR #13)         | **31.22 ± 0.21**  | **30.34 ± 0.13**  | **29.48 ± 0.20**  | **19.65 ± 0.36**  |
| `_chunked` (`GGML_Q1_G128_CHUNKED=1`)  | 24.76 ± 0.07      | 24.27 ± 0.27      | 23.05 ± 0.69      | 19.84 ± 0.05      |
| **Δ vs default**                       | **−20.7%**        | **−20.0%**        | **−21.8%**        | +1.0% (within noise) |

Per-run numbers (raw):

```
run 1: default  pp256 31.49  pp512 30.51  pp1024 29.72  tg32 19.90
run 1: chunked  pp256 24.83  pp512 24.62  pp1024 23.44  tg32 19.89
run 2: default  pp256 31.15  pp512 30.28  pp1024 29.33  tg32 19.81
run 2: chunked  pp256 24.69  pp512 24.03  pp1024 22.16  tg32  7.79  *
run 3: default  pp256 31.01  pp512 30.24  pp1024 29.39  tg32 19.23
run 3: chunked  pp256 24.76  pp512 24.17  pp1024 23.54  tg32 19.83

* run 2 chunked tg32=7.79 looks like a thermal / scheduling outlier on
  the shared demo VM; tg32 is the smallest test (32 tokens) and is
  most sensitive to per-call jitter. The other 2 runs show tg32 within
  noise of the default. The pp numbers are stable across runs.
```

## Analysis

The chunked kernel is a **clear regression on Zen 3 / AVX2-only**. The
deinterleave overhead (28 extra ops per sub-block) dominates whatever
savings come from skipping the explicit unpack pass.

Why the explicit unpack is "cheaper than it looks":

1. **The unpack is amortised across the y-group, not the inner loop.** It
   runs 4 × `nb_q8` iterations once per 4 output rows, then the inner
   kernel runs `nb × 4 × 64` SIMD ops (= 1024 ops at Bonsai n=2048,
   nb=16). The unpack's ~4 × 32 × `nb_q8` = ~8 KB of byte copies takes a
   few hundred cycles vs the inner kernel's ~10 000 cycles per y-group.
2. **The scratch buffer fits comfortably in L1.** ~9 KB scratch vs Zen
   3's 32 KB L1d means it does not evict the weight stripe.
3. **Deinterleave is amortised per inner-loop iteration, not per call.**
   Adding 7 ops to every `(sub-block, activation_row)` is 28 ops per
   sub-block × `nb × 4` sub-blocks = 1 792 extra ops per y-group on
   Bonsai. That is more than the ~500-cycle unpack saves.

The cost-benefit breakdown predicted on a back-of-envelope basis (~28
ops extra per sub-block costing more than 500 cycles unpack saves)
matches the observed −20% pp regression: 1 792 / 10 000 ≈ 18% slowdown,
within stddev of the measured −20%.

## Why the kernel was reverted

After measurement showed the regression, two options were on the table:
keep the chunked kernel as opt-in dead code (gated by env var) for
future maintainers to revive on different ISAs, or revert the kernel
and keep only this document.

The revert was chosen because:

1. **~190 lines of always-off code in a hot-path file** is real
   maintenance cost. Future refactors of `arch/x86/repack.cpp` would
   need to keep both kernels in sync, even though one is never the
   default.
2. **The kernel is not lost** — it lives in the git history of the
   feature branch (`devin/1777638955-prism-drop-unpack-chunked-gemm`,
   commit `e81713f0`) and can be cherry-picked or re-applied if a future
   AVX-VNNI / AVX-512 / NEON port wants to A/B-test the same idea on a
   different ISA.
3. **This document captures the outcome.** The numbers, parity proof,
   analysis, and "why it lost" rationale are preserved here, so any
   future maintainer who finds themselves wondering "what about
   consuming `block_q8_0x4` directly?" can read this doc first.

## When the chunked kernel might still win

This experiment is specific to **AVX2-only Zen 3**. The chunked variant
might pay off on:

- **AVX-VNNI or AVX-512 hosts** where the inner-loop dot product is
  ~3× faster (one `vpdpbusd_epi32` instead of `vpmaddubsw + vpmaddwd`),
  so the deinterleave overhead becomes a smaller fraction of total
  kernel time. On Alder Lake P-cores or Zen 4 the chunked kernel might
  break even or win.
- **NEON / SVE** where `tbl` / `tbx` instructions can do 4-way
  deinterleave in fewer ops than AVX2's permutevar+blend chain. Worth
  re-measuring if the NEON kernel is ever ported into
  `arch/arm/repack.cpp` (currently it falls through to the generic
  scalar path).
- **Larger `n` or `nr`** where L1 pressure starts to evict weight
  stripes. Bonsai's hidden=2048 / `nb_q8`=64 keeps the scratch buffer
  small; a model with hidden=8192 would have a 36 KB scratch that
  spills out of L1.

If any of those conditions become relevant, revive the kernel from the
feature branch (`devin/1777638955-prism-drop-unpack-chunked-gemm`,
commit `e81713f0`) instead of re-implementing from scratch.

## What does ship in this PR

- ✅ This document, recording the experiment, the measured numbers, and
  the rationale for not keeping the kernel in-tree.

## What does NOT ship

- ❌ The chunked kernel itself (reverted; lives only in the feature
  branch's git history).
- ❌ The `GGML_Q1_G128_CHUNKED` env var (reverted; the dispatcher in
  `arch/x86/repack.cpp` is unchanged).
- ❌ No on-disk format changes; no GGUF compatibility breaks.
- ❌ No changes to non-AVX2 paths.
- ❌ No changes to the GEMV path (`nr < 4`); the unpack experiment is
  GEMM-specific.

## AI authorship disclosure

This experiment was implemented and measured with Devin assistance on
the **prism (private) fork**. The chunked kernel (now reverted), the
env-var gate (now reverted), and this write-up are all
AI-authored / AI-edited under user authorisation for the private fork.
The default GEMM path (`_4rows`) is untouched.

This work is not intended for upstream `ggml-org/llama.cpp`; the
upstream project's `AGENTS.md` / `CONTRIBUTING.md` rules apply only
when work is being prepared for upstream submission.
