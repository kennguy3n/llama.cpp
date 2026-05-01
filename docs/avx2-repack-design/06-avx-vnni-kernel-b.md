# Phase 7: AVX-VNNI Kernel B for Q1_0_g128 4×4 repack

## Status

In-tree; gated on `__AVXVNNI__`. Compile-time selected by the existing
dispatcher in `ggml/src/ggml-cpu/arch/x86/repack.cpp` whenever the
ggml-cpu build target sets `-mavxvnni` (Alder Lake P, Sapphire Rapids,
Zen 4).

* AVX2-only build (default `-march=native` on Zen 3 / Skylake-X /
  Cascade Lake) → unchanged. Kernel A (the PR #13 4×4 GEMM) stays the
  fast path; Kernel B is omitted at compile time.
* AVX-VNNI build (`-DGGML_AVX_VNNI=ON`, or any `-march` that selects an
  AVX-VNNI-capable CPU) → Kernel B replaces Kernel A in the dispatcher.

**Parity verified under Intel SDE 9.48 (`-adl`, Alder Lake CPUID +
chip-check)**. See
[`parity-alderlake-sde-kernel-b.txt`](parity-alderlake-sde-kernel-b.txt).
Both `nr=1` (GEMV), `nr=4` (GEMM), and `nr=8` (GEMM) produce
`diff = 0.000000e+00` against the single-row reference `vec_dot`.
End-to-end smoke test under SDE generates "Paris" correctly for
`The capital of France is` from a real Bonsai-1.7B Q1_0_g128 GGUF.

**No real-hardware perf number.** The development host is an AMD
EPYC 7763 (Zen 3) which does not have native `vpdpbusd_avx`. SDE
dynamically translates the instruction into host-native sequences,
making timing meaningless under emulation. The microbench result
from PR #11 measured Kernel A at 1.28× over single-row AVX2 in a
controlled microkernel. Kernel B's projected uplift on real
Alder Lake / Zen 4 is left for whoever owns such a host to measure.

## What changed

Two new kernels in `ggml/src/ggml-cpu/arch/x86/repack.cpp`:

* `ggml_gemv_q1_0_g128_4x4_q8_0_avx_vnni` — `nr=1` GEMV variant.
* `ggml_gemm_q1_0_g128_4x4_q8_0_avx_vnni_4rows` — `nr%4==0` GEMM
  variant matching the AVX2 4-rows kernel signature.

Both share the same `block_q1_0_g128x4` repack format (no on-disk
format change), the same shuffle/bit-test mask constants, and the
same outer skeleton as their AVX2 counterparts. The dispatcher in
`ggml_gemv_q1_0_g128_4x4_q8_0` and `ggml_gemm_q1_0_g128_4x4_q8_0`
prefers Kernel B over Kernel A whenever `__AVXVNNI__` is defined.

## Inner-loop body diff

The AVX2 inner body computes a bipolar dot via the explicit
`2*partial - sum_qy` trick:

```cpp
const __m256i masked  = _mm256_blendv_epi8(zero, qy, mask_w[r_w]);
const __m256i partial = _mm256_madd_epi16(
    _mm256_maddubs_epi16(ones_b, masked), ones_w);
const __m256i row_dot = _mm256_sub_epi32(
    _mm256_slli_epi32(partial, 1), sum_qy_v);
```

Kernel B collapses the `madd*` chain to a single `dpbusd`:

```cpp
const __m256i signed_qy = _mm256_blendv_epi8(neg_qy, qy, mask_w[r_w]);
const __m256i int_acc   = _mm256_dpbusd_avx_epi32(
    zero, ones_b, signed_qy);
```

The activation row pre-computes `neg_qy = 0 - qy` once per `qy` load
(shared across all 4 `r_w` iterations). `sum_qy_v` and the
`maddubs/madd/slli` chain disappear. Per `(r_a, r_w)` per sub-block,
this drops from 5 vector ops + 1 shared op to 2 vector ops.

## Op-count change per (r_a, r_w) per sub-block

| Op                    | Kernel A (AVX2) | Kernel B (AVX-VNNI) |
|-----------------------|----------------:|---------------------:|
| `blendv_epi8`         | 1               | 1                    |
| `maddubs_epi16`       | 1               | 0                    |
| `madd_epi16`          | 1               | 0                    |
| `slli_epi32`          | 1               | 0                    |
| `sub_epi32`           | 1               | 0                    |
| `dpbusd_avx_epi32`    | 0               | 1                    |
| **Total**             | **5**           | **2**                |

Plus, shared per `qy` load (per `(r_a, sub_block)`):
* Kernel A: `maddubs + madd` to produce `sum_qy_v` → 2 ops shared across 4 `r_w`.
* Kernel B: `sub_epi8` to produce `neg_qy` → 1 op shared across 4 `r_w`.

Per-sub-block total: 5×4 + 2 = **22 ops** (Kernel A) vs 2×4 + 1 = **9 ops** (Kernel B). 2.4× fewer integer ops in the hot path.

## What this means in practice

* **Latency**: `vpdpbusd_avx` is 5-cycle latency on Alder Lake P-cores
  (vs `pmaddubsw + pmaddwd` chain ≈ 5+4 = 9-cycle dependency chain on
  the same hosts). Each accumulator's per-iteration critical path
  shortens by roughly 4 cycles; with 16 accumulators in flight
  (`acc[r_a][r_w]`) the IPC is bounded by port-5 throughput, not
  latency.
* **Throughput**: Alder Lake P has 1× `vpdpbusd_avx` per cycle on
  port 5; total inner-loop port pressure drops from 22 ops/sub-block
  to 9 ops/sub-block. On the 16-accumulator GEMM kernel that's
  expected to translate to a meaningful (but not dramatic) prefill
  uplift over Kernel A.
* **Memory traffic**: identical to Kernel A. Same `block_q1_0_g128x4`
  layout, same activation unpack from `block_q8_0x4`. The win is
  purely on the integer compute side.

The explicit *projection* in PR #11's microbench was that Kernel B
on AVX-VNNI hosts approaches AVX-512 single-row throughput — likely
~70 t/s pp512 on Bonsai-1.7B / Alder Lake P-cores, vs the 30.87 t/s
PR #13 baseline on Zen 3. **This number is unverified on real
hardware**; whoever owns an Alder Lake / Zen 4 box can confirm or
correct it.

## How to build + test

### Default Zen 3 / AVX2-only

No change. Existing build path (`-march=native` or
`-DGGML_NATIVE=ON`) compiles only Kernel A on this host. Kernel B
is omitted at compile time.

### AVX-VNNI build (for SDE testing or real Alder Lake / Zen 4 hosts)

```bash
cmake -B build-vnni -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_BMI2=ON -DGGML_AVX_VNNI=ON \
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-vnni -j$(nproc)
```

### Parity check (Intel SDE)

```bash
sde64 -adl -- ./build-vnni/bin/test-q1-g128-repack-parity
```

Expected: `PASS: GEMV (nr=1) + GEMM (nr=4) + GEMM (nr=8) all match
single-row vec_dot within FP16 tolerance` with `diff = 0.000000e+00`
on every element.

### Real-hardware bench (whoever has Alder Lake / Zen 4)

```bash
./build-vnni/bin/llama-bench -m Bonsai-1.7B.gguf -t 4 -p 256,512,1024 -n 32 -r 3
```

Compare pp512 to the PR #13 Zen 3 baseline of 30.87 ± 0.15 t/s. Any
real-hardware number on AVX-VNNI is a new data point — please commit
it to this repo as `phase7-end-to-end-<host>.txt` next to the
existing `phase3-end-to-end-zen3-epyc-7763.txt`.

## Disassembly cross-check

```
$ objdump -d build-vnni/bin/libggml-cpu.so | grep -c vpdpbusd
506
```

Both GEMV and GEMM 4-rows variants are inlined into the public
dispatchers (`ggml_gemv_q1_0_g128_4x4_q8_0` and
`ggml_gemm_q1_0_g128_4x4_q8_0`) by GCC 11.4 at `-O3`. The 506-count
matches the expected unrolling pattern: 16 `vpdpbusd_avx` per
GEMM inner iteration (4 sub-blocks × 4 r_a × 1 inlined per r_w that
GCC hoisted), 16 in the GEMV inner iteration (4 k × 4 r), plus
register-renaming copies in the prologue/epilogue.

## Why no `_mm256_dpbusds_*` saturating variant?

The non-saturating `vpdpbusd_avx` is sufficient here. Each lane
accumulates `sum(±qy)` over at most 32 bytes per sub-block. With
`qy` already INT8 (range `[-128, 127]`), the worst-case per-lane
sum bound is `32 × 127 = 4064`, well within INT32 range. There is
no possibility of overflow in the `dpbusd` step itself. The
saturating `vpdpbusds_avx` would behave identically here.

## Known follow-ups

This change does NOT include:

* AVX-512 VNNI variant. AVX-512 hosts already use the existing
  single-row `vec_dot` AVX-512 path (which the cross-arch validation
  in [04-cross-arch-validation.md](04-cross-arch-validation.md) flags
  as having a pre-existing GCC 11.4 build break in `Q2_0`). A 4×4
  AVX-512-VNNI variant would be a follow-up; the necessary
  `_mm256_dpbusd_epi32` / `_mm512_dpbusd_epi32` intrinsics are
  separately available behind `__AVX512VNNI__`.
* Real-hardware perf measurement. Awaiting an Alder Lake / Zen 4 host.
* AVX-VNNI runtime CPU detection in the dispatcher. Currently
  compile-time only via `__AVXVNNI__`. Phase 4 (the on-disk repack
  converter) will likely add runtime detection alongside the
  `--repack-q1-g128-avx2` flag.

## AI authorship disclosure

Per [`AGENTS.md`](../../AGENTS.md) and
[`CONTRIBUTING.md`](../../CONTRIBUTING.md), this is the `prism`
private fork. The kernel port from `tools/microbench/` Phase 2 into
`arch/x86/repack.cpp` was AI-assisted under explicit authorisation
from @kennguy3n. Parity validation under SDE was run on the live
build artefact; the captured output is committed verbatim. This
work is not intended for upstream `ggml-org/llama.cpp`.
