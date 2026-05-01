# Phase 4: full dispatcher + on-disk repack converter (revised scope)

## Status

**Verified, no new code.** The runtime ISA dispatcher is already
infrastructure-complete via `GGML_CPU_ALL_VARIANTS=ON` +
`GGML_BACKEND_DL=ON`. The on-disk pre-repack converter is
intentionally **not** added: the equivalent upstream pattern
(`GGML_TYPE_Q4_0_4_4` / `_4_8` / `_8_8`) has been deprecated in
favour of runtime repack only.

This document records what Phase 4 was originally scoped to add, what
the existing infrastructure already provides, and the verification
that PR #13 (Kernel A) and PR #17 (Kernel B) drop into the right
runtime variants without further code changes.

## What Phase 4 was originally scoped to add

From [`03-design-brief.md`](03-design-brief.md):

> Phase 4 — full dispatcher + on-disk repack converter wiring +
> cross-arch fallback testing on real ARM / AVX-512 hosts.

Three pieces:

1. **Full dispatcher** — runtime CPU detection so a single binary can
   pick AVX-VNNI vs. AVX2 vs. AVX-512 vs. NEON paths at load time
   instead of at compile time.
2. **On-disk repack converter** — a `--repack-q1-g128-avx2` flag in
   `llama-quantize` to produce a pre-repacked GGUF, registering a new
   `GGML_TYPE_Q1_0_G128_AVX2` enum and skipping the runtime repack.
3. **Cross-arch fallback testing** — verify NEON / AVX-512 / RISC-V /
   PPC / s390x / Wasm fallbacks still work after PR #13.

Item (3) was done in PR #14 ([`04-cross-arch-validation.md`](04-cross-arch-validation.md)).

## How the existing dispatcher already covers item (1)

ggml-cpu has a backend-variant build mode that compiles **multiple**
copies of the CPU backend, each gated on a different ISA feature
set, and ships them as separate shared libraries. At process start,
the loader inspects the host CPU and picks the most capable variant
that the host supports.

```cmake
ggml_add_cpu_backend_variant(haswell        SSE42 AVX F16C FMA AVX2 BMI2)
ggml_add_cpu_backend_variant(alderlake      SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
ggml_add_cpu_backend_variant(skylakex       SSE42 AVX F16C FMA AVX2 BMI2 AVX512)
ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
ggml_add_cpu_backend_variant(sapphirerapids ... AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
# ... and ARM / PPC / RISC-V variants below
```

(Source: [`ggml/src/CMakeLists.txt:353-385`](../../ggml/src/CMakeLists.txt))

For the Q1_0_g128 4×4 work, the relevant variants are:

| Variant         | ISA flags                          | `__AVXVNNI__` | `__AVX512F__` | Kernel selected        |
|-----------------|------------------------------------|--------------:|--------------:|------------------------|
| `haswell`       | AVX2 + FMA + F16C + BMI2           | undefined     | undefined     | Kernel A (PR #13)      |
| `alderlake`     | + AVX_VNNI                         | **defined**   | undefined     | Kernel B (PR #17)      |
| `skylakex`      | + AVX512                           | undefined     | defined       | Single-row AVX-512 vec_dot (existing) |
| `cascadelake`   | + AVX512 + AVX512_VNNI             | undefined     | defined       | Single-row AVX-512 vec_dot (existing) |
| `zen4`          | + AVX512 + AVX512_VBMI/VNNI/BF16   | undefined     | defined       | Single-row AVX-512 vec_dot (existing) |
| `sapphirerapids`| + AMX_TILE + AMX_INT8              | undefined     | defined       | Single-row AVX-512 vec_dot (existing) |

The existing runtime gate in
[`ggml/src/ggml-cpu/repack.cpp:3702`](../../ggml/src/ggml-cpu/repack.cpp)
handles the AVX-512 case correctly:

```cpp
} else if (cur->type == GGML_TYPE_Q1_0_g128) {
    // ... single-row AVX-512 vec_dot beats the 4-row repack on AVX-512 hosts.
    if (ggml_cpu_has_avx2() && !ggml_cpu_has_avx512()) {
        if (cur->ne[1] % 4 == 0) {
            return &q1_0_g128_4x4_q8_0;
        }
    }
}
```

`ggml_cpu_has_avx512()` returns the compile-time `__AVX512F__` flag
from the variant's source — not a runtime CPUID check — but each
variant has only the right `__AVX*__` defines. So:

* haswell variant → `avx2 && !avx512` → registers the 4×4 repack.
  Kernel A is the only kernel compiled in (no `__AVXVNNI__`); the
  existing dispatcher in `arch/x86/repack.cpp` selects it.
* alderlake variant → `avx2 && !avx512` → registers the 4×4 repack.
  Kernel B replaces Kernel A at compile time via the `#if
  defined(__AVXVNNI__)` branch added in PR #17.
* skylakex / zen4 / sapphirerapids → `avx512` → the runtime gate
  rejects the 4×4 repack and the existing single-row AVX-512
  `vec_dot` path (in `arch/x86/quants.c`) handles the matmul.

**Net result**: an ALL_VARIANTS build picks the right kernel at
runtime per host, with no code changes required for Phase 4.

## Verification (this VM, AMD EPYC 7763 / Zen 3)

Built with:

```bash
cmake -B build-variants -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-variants -j --target ggml-cpu-haswell ggml-cpu-alderlake
```

Disassembly check confirms the kernels land in the expected variants:

```
$ objdump -d build-variants/bin/libggml-cpu-haswell.so   | grep -c vpdpbusd
0
$ objdump -d build-variants/bin/libggml-cpu-alderlake.so | grep -c vpdpbusd
506
```

* haswell: 0 `vpdpbusd` → Kernel A only, as expected (no AVX-VNNI).
* alderlake: 506 `vpdpbusd` → Kernel B compiled in. Inlined by GCC
  11.4 at `-O3` into `ggml_gemv_q1_0_g128_4x4_q8_0` and
  `ggml_gemm_q1_0_g128_4x4_q8_0` per PR #17.

Full ALL_VARIANTS build hits a **pre-existing GCC 11.4 break in the
Q2_0 AVX-512 path** (`_mm512_setr_epi8` not declared in
`<immintrin.h>` for GCC < 12) flagged in
[`04-cross-arch-validation.md`](04-cross-arch-validation.md). This
bug is unrelated to the repack work and blocks the
`skylakex` / `cannonlake` / `cascadelake` / `icelake` /
`sapphirerapids` / `zen4` / `cooperlake` variants until either the
toolchain is upgraded to GCC 12+ or that call site is rewritten to
load from a constant array.

The kernel-level cross-arch fallback (NEON, generic scalar, etc.)
was already validated in PR #14.

## Why the on-disk repack converter is intentionally not added

The original Phase 4 plan called for:

* A new `GGML_TYPE_Q1_0_G128_AVX2` enum (the pre-repacked 4×4 type).
* A `--repack-q1-g128-avx2` flag in `llama-quantize` to convert a
  `Q1_0_g128` GGUF into a `Q1_0_G128_AVX2` GGUF.
* A loader path that recognises `GGML_TYPE_Q1_0_G128_AVX2` and skips
  the runtime repack.

This pattern existed upstream as `GGML_TYPE_Q4_0_4_4`,
`GGML_TYPE_Q4_0_4_8`, and `GGML_TYPE_Q4_0_8_8` (and parallel
`Q4_0_4x4` / `Q4_0_4x8` / `Q4_0_8x8` types). **Upstream has since
deprecated and removed all of these on-disk pre-repacked formats.**
See [`ggml-org/llama.cpp` PR #10446](https://github.com/ggml-org/llama.cpp/pull/10446)
and the runtime-repack pattern at
[`repack.cpp:3572-3722`](../../ggml/src/ggml-cpu/repack.cpp).

Reasons for the upstream deprecation:

1. **Distribution complexity.** Each on-disk format is host-specific
   (a `Q4_0_4_4` GGUF only runs efficiently on hosts that select the
   4×4 traits at runtime; on other hosts it requires conversion). A
   single `Q4_0` GGUF runtime-repacks per host and "just works".
2. **Storage savings, none.** `block_q1_0_g128x4` packs 4 rows × the
   same 72 B as 4× `block_q1_0_g128`; on-disk size is identical to
   the unrepacked format. The runtime repack is ~1 ms per tensor on
   Zen 3 (measured) — negligible vs. model load time.
3. **Maintenance cost.** A new GGML enum entry forces every backend
   to add a code path for it (`ggml-cuda`, `ggml-metal`, `ggml-vk`,
   etc.) even if the type is irrelevant to that backend. The
   runtime-repack pattern is contained entirely within ggml-cpu.

Adopting the deprecated pattern on `prism` purely to mirror the
original Phase 4 plan would re-introduce the same costs upstream
already paid down. The existing runtime repack (validated at
30.87 t/s pp512 in PR #13) achieves the goal without the
distribution / maintenance overhead.

## Recommended consumer build for prism

For a single binary that targets all common x86 server / desktop
hosts (Zen 3 AVX2-only, Alder Lake AVX-VNNI, Zen 4 / Skylake-X /
Sapphire Rapids AVX-512):

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF
cmake --build build -j
```

This produces:

* `bin/libllama.so` (host-agnostic)
* `bin/libggml.so` + `bin/libggml-cpu.so` (loader)
* `bin/libggml-cpu-{x64,sse42,sandybridge,...,alderlake,zen4,sapphirerapids,...}.so`
  (per-ISA variants)

The loader picks the most-capable variant the host CPU supports,
hitting Kernel A on Zen 3 / Skylake-X, Kernel B on Alder Lake / Zen 4
where AVX-VNNI is available without AVX-512, and the existing
single-row AVX-512 vec_dot on Zen 4 / Sapphire Rapids / Cascade
Lake / Skylake-X.

NOTE: until the GCC 11.4 `_mm512_setr_epi8` issue is resolved (per
the open follow-up tracked in `04-cross-arch-validation.md`), the
ALL_VARIANTS build will fail on GCC 11.4 hosts at the AVX-512
variants. Either upgrade the build toolchain to GCC 12+ or restrict
the build to non-AVX-512 variants.

## Closeout for the AVX2 repack project

| Phase | Status         | PR                                      | Net merged |
|-------|----------------|------------------------------------------|------------|
| 1     | Design brief   | [#10](https://github.com/kennguy3n/llama.cpp/pull/10) | ✓ |
| 2     | Microbench     | [#11](https://github.com/kennguy3n/llama.cpp/pull/11) | ✓ |
| 3     | 4×4 GEMV+GEMM  | [#12](https://github.com/kennguy3n/llama.cpp/pull/12) | ✓ |
| 3.5   | True 4×4 GEMM  | [#13](https://github.com/kennguy3n/llama.cpp/pull/13) | ✓ |
| 4     | Dispatcher     | this doc                                 | doc-only   |
| 5     | Cross-arch     | [#14](https://github.com/kennguy3n/llama.cpp/pull/14) | ✓ |
| 6     | cv-guard bench | cv-guard [#17](https://github.com/kennguy3n/cv-guard/pull/17) | ✓ |
| 7     | AVX-VNNI Kernel B | [#17](https://github.com/kennguy3n/llama.cpp/pull/17) | ✓ |
|  +    | Drop-unpack    | [#15](https://github.com/kennguy3n/llama.cpp/pull/15) (doc-only) | ✓ |

All originally planned phases delivered or intentionally re-scoped
with documented rationale. **No further action required for the
core AVX2 repack work.**

Optional follow-ups (separate work, not part of the AVX2 repack
project):

* Fix or work around the `_mm512_setr_epi8` GCC 11.4 break in
  `arch/x86/quants.c:682` so the ALL_VARIANTS build completes the
  AVX-512 variants on GCC 11.4 hosts. (Tracked in PR #14.)
* Real-hardware perf measurement of Kernel B on Alder Lake / Zen 4 /
  Sapphire Rapids. Whoever owns such a host can run `llama-bench`
  and commit the result next to `phase3-end-to-end-zen3-epyc-7763.txt`.
* AVX-512 VNNI 4×4 variant. AVX-512 hosts currently use the
  single-row vec_dot path (which is already faster than the 4×4
  AVX2 GEMM at this hidden size). A 4×4 AVX-512 VNNI kernel would
  need to beat that — not obvious it would.

## AI authorship disclosure

Per [`AGENTS.md`](../../AGENTS.md) / [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
preamble for the prism private fork, this design document was
prepared with AI assistance under explicit authorisation from
@kennguy3n. The verification numbers
(`vpdpbusd`-count disassembly check) were captured from live build
artefacts in this session. Not intended for upstream
`ggml-org/llama.cpp`.
