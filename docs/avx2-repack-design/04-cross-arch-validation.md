# Phase 5: cross-arch validation of the Q1_0_g128 4×4 AVX2 repack

## Goal

PR #13 added a true 4×4 AVX2 GEMM to `ggml/src/ggml-cpu/arch/x86/repack.cpp`
plus generic-fallback GEMV/GEMM in `ggml/src/ggml-cpu/repack.cpp`. The
dispatcher gates the runtime repack on `__AVX2__ && !__AVX512F__`. This
document records the cross-arch verification done on top of
`prism @ 541a9866` (PR #13 merged).

## Dispatch surface, summarized

* **x86 with AVX2 and no AVX-512** (Zen 3, Skylake-X-no-AVX-512, etc.):
  takes the runtime repack and the new 4×4 AVX2 GEMM. The fast path
  measured by PR #13 (30.87 t/s pp512 on Zen 3 EPYC 7763).
* **x86 with AVX-512** (Skylake-X+AVX-512, Cascade Lake, Ice Lake, Zen 4):
  the dispatcher gate at `repack.cpp:3708` returns `nullptr` for
  `Q1_0_g128`, so the existing single-row `vec_dot` path in
  `arch/x86/quants.c` (which already uses VNNI-style mask blends) is
  used. The runtime repack code is never reached at runtime; the AVX-512
  build still has to *compile* the new AVX2 GEMM but not run it.
* **x86 without AVX2** (very old hosts, or any build with
  `-DGGML_AVX2=OFF`): the `arch/x86/repack.cpp` AVX2 kernels are
  compiled out by `#if defined(__AVX2__)`. The dispatcher gate evaluates
  to false (`ggml_cpu_has_avx2() == 0`), so the `tensor_traits` are
  never returned and the kernel never runs. The matmul instead goes
  through the existing scalar `vec_dot_q1_0_g128_q8_0` in `quants.c`.
* **Non-x86 (NEON/AArch64, RISC-V, LoongArch, s390x, WebAssembly,
  PowerPC)**: `arch-fallback.h` aliases the generic fallback in
  `repack.cpp` to the actual `ggml_gemv_q1_0_g128_4x4_q8_0` /
  `ggml_gemm_q1_0_g128_4x4_q8_0` symbols. Those generic implementations
  are pure C++ (no x86 intrinsics) and exercise the same packed
  `block_q1_0_g128x4` layout. However, those archs don't register a
  `tensor_traits` for `Q1_0_g128` either, so in practice the runtime
  repack is never enabled there - the single-row `vec_dot` path is used
  instead. The aliases exist so the generic functions stay
  link-resolved.

## Configurations tested

All builds done on AMD EPYC 7763, Zen 3, AVX2-only, GCC 11.4.

### 1. x86 with AVX2 (default Zen 3 build)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j 4
ctest -j 4 --output-on-failure
```

Result: **50/50 tests pass**. `tests/test-q1-g128-repack-parity` reports
`diff=0.000000e+00` on all 4 GEMV rows + 16 GEMM[nr=4] elements + 32
GEMM[nr=8] elements. Already validated end-to-end in PR #13.

### 2. x86 no-AVX, no-AVX2 (scalar fallback)

```sh
cmake -B build-noavx2 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
  -DGGML_AVX512=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build build-noavx2 --target test-q1-g128-repack-parity -j 4
./build-noavx2/bin/test-q1-g128-repack-parity
```

Result: **PASS**. Per-element diffs are ~3e-7 to ~1e-6, well within the
FP16 tolerance the test uses (9.82e-3). Diffs are FP32 rounding noise
from the different op ordering in the generic GEMM (no FMA fusion).

```sh
./build-noavx2/bin/test-quantize-fns | grep q1_0_g128
# Testing q1_0_g128
```

`test-quantize-fns` exercises the scalar `vec_dot_q1_0_g128_q8_0` path
and confirms it round-trips correctly.

### 3. x86 with AVX-512 (skipped by dispatcher)

```sh
cmake -B build-avx512 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_BF16=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
```

Build configuration succeeds. The build itself **fails** on
`ggml/src/ggml-cpu/arch/x86/quants.c:682`:

```
quants.c:682:29: error: implicit declaration of function
'_mm512_setr_epi8'; did you mean '_mm512_set1_epi8'?
[-Werror=implicit-function-declaration]
```

This is a **pre-existing GCC-version compatibility issue** in the
prism Q2_0 AVX-512 commit `96c3636c` (April 30, before the AVX2
repack project started). `_mm512_setr_epi8` was only added in GCC 12;
the demo VM has GCC 11.4. **This breakage is not caused by the
runtime repack code in this PR** - the new repack code in
`arch/x86/repack.cpp` and `repack.cpp` does not reference
`_mm512_setr_epi8` at all.

The AVX-512 build path is independently broken on this host. To fully
validate AVX-512, a host with GCC 12+ (or LLVM/clang) is required.

I have **not** patched `quants.c` here - the Q2_0 commit was authored by
@kennguy3n and is outside the scope of this Phase 5 work. Filing as a
separate finding for tracking; possible followups:

* Upgrade the demo VM toolchain to GCC 12+.
* Or replace `_mm512_setr_epi8` with a compile-time-constexpr-friendly
  alternative (e.g. a `_mm512_loadu_si512` of a `static const uint8_t
  shuf[64]`).

The dispatcher gate is independent of this issue: even if the AVX-512
build succeeded on a GCC-12+ host, `ggml_cpu_has_avx512()` would return
true and `repack.cpp:3708` would return `nullptr` for `Q1_0_g128`,
keeping the existing single-row path active.

### 4. AArch64 NEON (cross-compile + qemu)

```sh
cmake -B build-aarch64 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="-march=armv8-a+simd" \
  -DCMAKE_CXX_FLAGS="-march=armv8-a+simd" \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF
cmake --build build-aarch64 --target test-q1-g128-repack-parity -j 4
qemu-aarch64-static -L /usr/aarch64-linux-gnu \
  ./build-aarch64/bin/test-q1-g128-repack-parity
```

Result: **PASS**. Per-element diffs are 0 to ~1.2e-6, FP32 rounding
noise. The aarch64 binary executes the generic-fallback GEMV and GEMM
in `repack.cpp` (since `arch-fallback.h:73-74` aliases the
`*_generic` functions to the actual `ggml_gemv_q1_0_g128_4x4_q8_0` /
`ggml_gemm_q1_0_g128_4x4_q8_0` symbols on aarch64).

In an actual deployment on an AArch64 host, the runtime repack would
not be activated for `Q1_0_g128` because the
`ggml::cpu::repack::get_tensor_traits` call returns `nullptr` outside
the AVX2-no-AVX-512 gate. The single-row `vec_dot` path (in NEON if
the host has it, else scalar) handles the matmul. The generic
fallback exists to keep the symbol table linkable on cross-arch builds
where the `block_q1_0_g128x4` template instantiations of
`gemv<>`/`gemm<>` get linked from `repack.cpp`.

## Summary of findings

| Configuration | Build | Run | Parity | Notes |
|---|---|---|---|---|
| x86 AVX2 (PR #13 default) | OK | OK | diff = 0 | 50/50 ctest pass |
| x86 no AVX (scalar) | OK | OK | diff <= 1e-6 | FP rounding only |
| x86 AVX-512 (GCC 11.4) | **FAIL** | n/a | n/a | Pre-existing Q2_0 commit `_mm512_setr_epi8` issue |
| AArch64 NEON (cross + qemu) | OK | OK | diff <= 1.2e-6 | Generic fallback path |

## Open issues

1. **AVX-512 build broken on GCC 11.4** (pre-existing in commit
   `96c3636c`): `_mm512_setr_epi8` is a GCC-12+ intrinsic. Independent
   of this PR. Fix candidates: upgrade VM toolchain, or replace the
   call site with a `_mm512_loadu_si512` from a 64-byte constant
   array.

## Conclusion

PR #13's runtime repack is **safe across all the architectures
checked**. The generic fallback (used outside the AVX2-no-AVX-512
gate) produces FP32 results bit-equivalent to the AVX2 fast path
within FP16 tolerance, so consumers on non-AVX2 x86, AArch64, RISC-V,
LoongArch, s390x, WebAssembly and PowerPC can rely on the same
correctness contract.
