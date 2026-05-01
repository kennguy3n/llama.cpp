# `microbench-q1-g128-repack`

Phase 2 microbenchmark harness for the proposed AVX2 / AVX-VNNI runtime
repack of `Q1_0_g128`, designed in
[`docs/avx2-repack-design/03-design-brief.md`](../../docs/avx2-repack-design/03-design-brief.md)
(merged on `prism` via PR #10).

## Why a separate harness?

A standalone harness lets us iterate on SIMD intrinsics in seconds
instead of minutes. The design-brief kernel candidates (Kernel A
"AVX2-only 4×4 GEMV", Kernel B "AVX-VNNI 4×4 GEMV") live entirely
inside `q1_g128_repack.cpp` and never touch
`ggml/src/ggml-cpu/arch/x86/`. That means:

* No need to rebuild ggml + llama on every kernel tweak.
* Easy to run on a clean Zen 3 / Alder Lake / Zen 4 VM without dragging
  in CUDA/Metal/Vulkan toolchains.
* Bit-identical-output parity is verified against the scalar reference
  implementation **and** the existing AVX2 `vec_dot` path inside the
  same binary, so layout bugs surface before any production-side
  integration work begins (Phase 3+).

When the harness numbers are good, the validated kernel gets ported
into `ggml/src/ggml-cpu/arch/x86/repack.cpp` (Phase 3) along with the
matching `tensor_traits` + dispatcher hooks (Phase 4).

## Build

```bash
cmake -S . -B build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build build -j$(nproc) --target microbench-q1-g128-repack
```

By default the harness compiles with `-mavx2 -mfma -mf16c`. Enable the
AVX-VNNI kernel with:

```bash
cmake -S . -B build -DLLAMA_MICROBENCH_AVX_VNNI=ON
cmake --build build -j$(nproc) --target microbench-q1-g128-repack
```

The AVX-VNNI binary will only execute correctly on Alder Lake P-cores,
Zen 4, or any `vpdpbusd_avx`-capable host. On Zen 3 it will issue the
`#UD` exception at runtime — leave `LLAMA_MICROBENCH_AVX_VNNI=OFF` on
the demo VM.

We deliberately do **not** enable `-mavx512vnni` here. Phase 1's design
brief targeted AVX2-only and AVX-VNNI consumer hosts; AVX-512 is
already covered by the existing kernel at
`ggml/src/ggml-cpu/arch/x86/quants.c:855`.

## Run

```bash
./build/bin/microbench-q1-g128-repack [--blocks N] [--iters M] [--seed S]
```

| Flag | Default | Meaning |
|---|---:|---|
| `--blocks N` | `4096` | Number of `Q1_0_g128` weight blocks per row (1 block = 128 weights). For a Bonsai-1.7B-scale workload use `13312` (~1.7M weights). |
| `--iters M`  | `1024` | Timed iterations. Median of `M` reported. |
| `--seed S`   | `0xC0FFEE` | RNG seed for synthesis. Determines exact bit pattern of weights and activations. |

Output is plain text on stdout. Sample baseline numbers from the
EPYC 7763 / Zen 3 demo VM are committed at
[`baseline-zen3-epyc-7763.txt`](baseline-zen3-epyc-7763.txt) so
reviewers can see results without rebuilding.

## What the harness measures

Each iteration computes **four output rows** of dot products against a
single column of activations. The four kernels timed (under the same
inputs from the deterministic seed):

| # | Kernel | Layout | Op count / 32-weight sub-block / row | Compiled when |
|---|---|---|---:|---|
| 1 | `vec_dot_scalar` | `block_q1_0_g128` | ~256 (1 op per bit) | always |
| 2 | `vec_dot_avx2` | `block_q1_0_g128` | ~10 | `__AVX2__` |
| 3 | `gemv_q1g128_4x4_avx2` (NEW) | `block_q1_0_g128x4` | ~10 (per-row) + ~1 amortised shared | `__AVX2__` |
| 4 | `gemv_q1g128_4x4_avx_vnni` (NEW) | `block_q1_0_g128x4` | ~7 (per-row) + ~1 shared | `__AVXVNNI__` |

**Parity gate**: kernels 2, 3, and 4 must produce FP32 outputs that
match the scalar reference within `1e-3` relative tolerance per row.
Tolerance is generous because FP16 deltas accumulate roundoff
differently in scalar vs vectorised reductions; the asserted bound is
well below the user-visible noise floor of FP16-quantised activations.
A failed parity check exits with status `2` *before* timing runs.

**Timing methodology**: `clock_gettime(CLOCK_MONOTONIC_RAW)`-backed
`std::chrono::steady_clock::now`, median of `M` iterations after a
32-iteration warm-up. Single-threaded by design — multi-threaded
prefill perf is a separate question covered by `tools/llama-bench`.

## Interpreting the numbers

Same workload, same inputs, all four kernels produce four row dots,
so the ns/4-rows column is directly comparable. Convert to "ns/row" by
dividing by 4. Convert to "ns per weight" by dividing further by the
weight count `nb * 128`.

For comparison against `tools/llama-bench` numbers:

* `llama-bench` reports tokens/s. One token of prefill traverses the
  full ~1.7B weight matrix, so weights/s = `tokens/s × 1.7e9`. Convert
  to ns/weight via `1e9 / weights_per_s`. Then divide by the number
  of physical cores being used (the harness is single-threaded; real
  prefill uses 4 cores on the demo VM).
* The per-element numbers from this harness should match
  `llama-bench` to within ~10% after the thread-count divide. A larger
  divergence usually means a different cache footprint (Bonsai-1.7B's
  full weight matrix is ~250 MB and won't fit in L2; tune `--blocks`
  to roughly match if you need apples-to-apples).

## What "good" looks like

The Phase-1c brief estimated a 2.5× win over `vec_dot_avx2` on AVX2-only
hosts. The actual measured win on Zen 3 is **1.28×**
(see baseline file). The design brief over-estimated because it
assumed bit expansion would amortise across 4 rows, but each row has
different `qbits` and the bit expansion is per-row work that cannot
be shared.

The gain is real and reproducible, but smaller than predicted. The
realistic projection for Bonsai-1.7B prefill on EPYC 7763 / Zen 3:

* Current `Q1_0_g128` AVX2: 23 t/s prefill (measured by `llama-bench`).
* With 4×4 GEMV repack: ~28–30 t/s (1.28× × 23).
* With Q3_K_M (existing alternative, no kernel work): 41 t/s.

In other words, the AVX2 4×4 repack is a ~30% improvement but doesn't
catch K-quants for free. **The real prize is AVX-VNNI**: Kernel B
collapses the per-row dot to a single `vpdpbusd_avx` and is expected
to push throughput close to the AVX-512 fast-path (~70 t/s
projected). A follow-up run on an Alder Lake P-core or Zen 4 host
will quantify that gain.

## Phase gate

This harness produces the data that gates Phase 3. The Phase-1c
checklist box "the kernel intrinsics produce ≥2× over the existing
AVX2 path on Zen 3" should be revised based on these numbers. See
the PR thread on `kennguy3n/llama.cpp` for the conversation.
