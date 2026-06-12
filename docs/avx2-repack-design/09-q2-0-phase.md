# Q2_0 phase: 4-row interleaved repack + AVX2 / AVX-VNNI kernels

## Status

**Implemented and measured.** This phase extends the proven Q1_0_g128
4-row runtime-repack design (Phases 1–4, see docs 01–08) to the 2-bit
ternary `Q2_0` type used by the `prism-ml/Ternary-Bonsai-8B` model. The
on-disk GGUF format is unchanged; the new layout is produced in memory at
tensor-init time and only fires on AVX2-only hosts.

## Motivation

`Q2_0` shipped with a single-row GEMV only (`ggml_vec_dot_q2_0_q8_0`,
`nrc == 1`). At inference time this reloads the activation for every output
row and re-runs the full 2-bit unpack (shuffle + 3 shifts + 4 ands + 3 ors)
per sub-block per row. The Q1_0_g128 work already showed that a 4-row
interleaved repack — loading each activation once and reusing it across 4
weight rows — recovers a large fraction of the prompt-processing throughput
left on the table by the single-row path. `Q2_0` has the same structural
opportunity.

Baseline `llama-bench` on this VM (AMD EPYC 7763, Zen 3, AVX2/FMA/F16C, no
AVX-512), `Ternary-Bonsai-8B-Q2_0.gguf`:

| threads | test  | t/s         |
| ------: | ----- | ----------: |
|       8 | pp128 | 7.52 ± 0.19 |
|       8 | tg64  | 5.67 ± 0.45 |

## Layout: `block_q2_0x4`

`Q2_0` packs 128 weights per block: a single FP16 delta and 32 bytes of
2-bit ternary codes (4 codes/byte, code `{0,1,2}` → value `(code-1)·d` ∈
`{-1,0,+1}·d`; see `quantize_row_q2_0_ref` in `ggml-quants.c`). Four such
blocks (one per row) interleave into a `block_q2_0x4` (136 B, identical to
`4 × sizeof(block_q2_0)`):

```
[d0 d1 d2 d3]                        // 8 B  : 4 FP16 deltas
[r0_sb0 r1_sb0 r2_sb0 r3_sb0]        // 32 B : sub-block 0, 8 bytes/row
[r0_sb1 r1_sb1 r2_sb1 r3_sb1]        // 32 B : sub-block 1
[r0_sb2 r1_sb2 r2_sb2 r3_sb2]        // 32 B : sub-block 2
[r0_sb3 r1_sb3 r2_sb3 r3_sb3]        // 32 B : sub-block 3
```

Each sub-block `k` holds 32 ternary codes per row (8 packed bytes/row),
aligned 1:1 with the four `block_q8_0` activation sub-blocks that cover the
same 128 weights. This is the Q1_0_g128 layout with the per-row stride
widened from 4 bytes (16 codes, 1 bit) to 8 bytes (32 codes, 2 bits).

- `repack.h` — `struct block_q2_0x4` + size asserts.
- `repack.cpp` — `make_block_q2_0x4`, `repack_q2_0_to_q2_0_4_bl`, the
  `repack/gemv/gemm<block_q2_0,4,4,GGML_TYPE_Q8_0>` template instantiations,
  the `tensor_traits` instance, and the dispatcher gate.

## Kernels

`arch/x86/repack.cpp` adds AVX2 and AVX-VNNI GEMV and GEMM kernels:

- **Byte unpack** — the 8-packed-bytes → 32 signed `{-1,0,+1}` expansion is
  the same `broadcast → shuffle-replicate → 4 shifted/masked streams →
  position-blend → sub 1` sequence as the single-row AVX2 kernel in
  `arch/x86/quants.c:ggml_vec_dot_q2_0_q8_0`.
- **Dot** — `|qx|` (unsigned, via `sign(qx,qx)`) and `sign(qy,qx)` feed
  either `maddubs`+`madd` (AVX2) or a single `dpbusd` (AVX-VNNI). Both
  reduce to the same 8 int32 lane-partials, so the AVX-VNNI path is
  bit-identical to the AVX2 path; only the instruction count differs.
- **GEMV** (`nr==1`) — loads each activation sub-block once and dots it
  against the 4 interleaved weight rows, accumulating into 4 `__m256`.
- **GEMM** (`nr%4==0`) — true 4×4 inner kernel: unpacks each weight row's
  ternary codes once per sub-block and reuses them across the 4 activation
  rows, holding 16 FP32 accumulators in registers. The activation side is
  the `block_q8_0x4` buffer produced by `ggml_quantize_mat_q8_0_4x4`,
  unpacked with the shared `gemm_q1_0_g128_unpack_row_from_q8_0x4` helper.

A scalar/SIMD-free `*_generic` GEMV/GEMM is also provided and aliased for
non-x86 builds in `arch-fallback.h`, mirroring Q1_0_g128.

## Dispatcher gate

`ggml_repack_get_optimal_repack_type` opts `Q2_0` into the 4-row repack only
when `ggml_cpu_has_avx2() && !ggml_cpu_has_avx512()` and `ne[1] % 4 == 0`.
The rationale matches Q1_0_g128: `Q2_0` has a hand-tuned single-row AVX-512
(BW+VL+VNNI) path in `arch/x86/quants.c` that already beats the generic
repack on AVX-512 hosts, so the repack is reserved for AVX2-only hosts where
no such fast path exists. AVX-512 hosts and non-x86 architectures keep their
existing single-row path with no behavioural change.

## Correctness

`tests/test-q2-0-repack-parity.cpp` (mirrors
`test-q1-g128-repack-parity.cpp`) quantises 4 weight rows of
`4 × QK2_0 = 512` weights as `Q2_0`, repacks them into `block_q2_0x4`, and
compares the production GEMV (`nr=1`) and GEMM (`nr=4`, `nr=8`) against the
mainline single-row `vec_dot`.

- **AVX2 (native, Zen 3):** all rows match with diff `0.000000e+00` (exact).
- **AVX-VNNI (`-mavxvnni` build, Intel SDE `-adl`):** PASS, bit-identical.

`test-quantize-fns` continues to pass for `q2_0` (the repack is transparent
to dequant).

## Benchmarks (this VM)

`llama-bench`, `Ternary-Bonsai-8B-Q2_0.gguf`, `-p 128 -n 64 -r 3`, AVX2
path. Baseline = same build with the Q2_0 gate removed (single-row GEMV).

| threads | test  | baseline (t/s) | repack (t/s) | speedup |
| ------: | ----- | -------------: | -----------: | ------: |
|       8 | pp128 |    7.52 ± 0.19 | 13.24 ± 0.02 |  1.76×  |
|       8 | tg64  |    5.67 ± 0.45 |  6.35 ± 0.05 |  1.12×  |
|      16 | pp128 |    7.27 ± 0.09 | 11.34 ± 0.13 |  1.56×  |
|      16 | tg64  |    2.94 ± 0.08 |  3.09 ± 0.04 |  1.05×  |

Prompt processing (GEMM-bound) sees the bulk of the win, as expected: it is
compute-bound on the weight unpack, which the 4-row kernel amortises across
activation rows. Token generation (GEMV-bound, `nr=1`) is closer to
memory-bandwidth-bound, so the gain is modest.

## Honest limits / caveats

- **Measured on one host.** All throughput numbers are this Zen 3 VM only.
  The AVX-VNNI kernel is validated for **correctness** under Intel SDE
  (Alder Lake), not for wall-clock throughput — SDE is a functional emulator,
  so its timings are not representative.
- **AVX2-only opt-in.** The gate deliberately excludes AVX-512 hosts, where
  the existing single-row VNNI path is faster. This was not re-measured here;
  it inherits the Q1_0_g128 rationale.
- **`llama-bench` label.** The model prints as `qwen3 8B Q1_0_g128`; this is
  a pre-existing cosmetic ftype-name bug in the label, not a sign the wrong
  type is loaded. The tensors are genuinely `Q2_0` (type id 42).
- **Not upstream-ready.** This is the `prism` private fork; the work is
  AI-assisted (disclosed per `AGENTS.md`) and has not had the cross-arch
  hardware testing upstream would require.
