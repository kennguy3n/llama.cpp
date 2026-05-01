# Phase 1a: Survey — How mainline ggml repacks `Q4_0` for AVX2 / NEON / SVE

**Status**: Research only. No code changes.
**Audience**: Reviewer of the planned `Q1_0_g128` AVX2 / AVX-VNNI repack.
**Goal**: Document the existing mainline mechanism so the new `Q1_0_g128` repack can mirror it exactly. Anything that deviates from this template is a deliberate design choice that must be justified in the design brief (`03-design-brief.md`).

## TL;DR

Mainline ggml does **not** introduce new on-disk GGUF types when it ships an AVX2-/NEON-/SVE-optimised packing of an existing quantization. The on-disk format stays the same. At **tensor-allocation time**, when the tensor lives in a special "CPU repack buffer", the dispatcher rewrites the in-memory layout into a row-interleaved variant that lets a wide GEMV / GEMM kernel issue one activation load per N output rows.

Concretely for `Q4_0`:

| Surface                        | What it is                                                                                                                       |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **GGUF type**                  | `GGML_TYPE_Q4_0` — unchanged on disk                                                                                            |
| **In-memory packed types**     | `block_q4_0x4`, `block_q4_0x8` (4- or 8-row interleaved)                                                                         |
| **Repack helpers**             | `make_block_q4_0x4`, `make_block_q4_0x8`, `repack_q4_0_to_q4_0_4_bl`, `repack_q4_0_to_q4_0_8_bl`                                |
| **GEMV kernels**               | `ggml_gemv_q4_0_4x4_q8_0`, `ggml_gemv_q4_0_4x8_q8_0`, `ggml_gemv_q4_0_8x8_q8_0`                                                  |
| **GEMM kernels**               | `ggml_gemm_q4_0_4x4_q8_0`, `ggml_gemm_q4_0_4x8_q8_0`, `ggml_gemm_q4_0_8x8_q8_0`                                                  |
| **Dispatch**                   | `ggml_repack_get_optimal_repack_type()` in `ggml/src/ggml-cpu/repack.cpp:3392`                                                  |
| **Backend buffer**             | `ggml_backend_cpu_repack_buffer_type()` in `ggml/src/ggml-cpu/repack.cpp:3619`                                                  |
| **Init hook (where it fires)** | `ggml_backend_cpu_repack_buffer_init_tensor` in `ggml/src/ggml-cpu/repack.cpp:3524`                                              |

That's the entire surface area. Sections 1–4 below walk through each piece.

## 1. The packed in-memory block layout

`ggml/src/ggml-cpu/repack.h:13–37` defines a generic interleave block:

```cpp
template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
```

The two template parameters are:
- **K**: bits per quant (here, 4 for Q4_0; 8 for Q8_0). Drives the `qs` byte count.
- **N**: rows interleaved. The deltas array has N entries because each interleaved row has its own scale.

So `block_q4_0x4` packs four consecutive rows of `block_q4_0` (each covering 32 elements) into a single struct: 4 deltas (8 B), then 64 B of nibbles, all sized to a single 128-bit (NEON) or 256-bit (AVX2) register load.

## 2. Repack helper: producing a packed block from N consecutive un-packed blocks

`ggml/src/ggml-cpu/repack.cpp:2016–2059` shows the layout transform for `make_block_q4_0x4`:

```cpp
static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x4 out;
    // 4 deltas first
    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }
    // Quants: interleave at granularity blck_size_interleave (4, 8, or full block)
    const int end = QK4_0 * 2 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_offset = (i / 4) * blck_size_interleave;
        int src_id     = i % 4;
        src_offset += (i % 4) * blck_size_interleave;   // deliberately re-using src_id math
        memcpy(&out.qs[i * blck_size_interleave],
               &in[src_id].qs[src_offset / 4],
               blck_size_interleave);
    }
    return out;
}
```

The key idea: **chunks of `blck_size_interleave` bytes are taken in round-robin from the four input rows**. The four rows are striped together so that one SIMD load gives the kernel the same byte-offset-N slice from all four rows simultaneously.

Two interleave granularities are used:
- `blck_size_interleave = 4` → produces `q4_0_4x4` (NEON `dotprod` target; one 128-bit lane spans 4 bytes from each of 4 rows).
- `blck_size_interleave = 8` → produces `q4_0_4x8` (NEON `i8mm` / matmul-int8 target; one 128-bit lane spans 8 bytes from each of 4 rows).

For 8-row interleave the same idea generalises to `make_block_q4_0x8` (one 256-bit AVX2 load covers 8 bytes from each of 8 rows).

## 3. The actual SIMD kernel for the packed layout

`ggml/src/ggml-cpu/arch/x86/repack.cpp:1448–1462` is the AVX2/AVX-512 entry point for `q4_0_8x8`:

```cpp
void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                             const void * vx, const void * vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        __m256i signextendlut = _mm256_castsi128_si256(
            _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                          7,  6,  5,  4,  3,  2,  1,  0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
        gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);
        return;
    }
#endif
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
```

Notice three structural facts that are repeated across every repacked kernel:

1. The **public signature is `gemv` / `gemm`, not `vec_dot`**. The whole point of repacking is to amortise the activation load across multiple output rows, which is impossible inside a single-row `vec_dot`.
2. The architecture-specific path is gated on a build-time `#if defined(__AVX2__) || defined(__AVX512F__)`, with a guaranteed fallback to a `_generic` C reference. This is enforced by the `arch-fallback.h` macro layer.
3. The actual SIMD work is a templated helper (`gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>`) so multiple block formats with the same interleave shape can share the inner loop.

## 4. Dispatch — how a tensor "decides" to use the repacked path

`ggml/src/ggml-cpu/repack.cpp:3392` is the central decision point:

```cpp
static const ggml::cpu::tensor_traits *
ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {
    static const tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    static const tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    static const tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;
    /* … other types … */

    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)
            || (ggml_cpu_has_riscv_v() && (ggml_cpu_get_rvv_vlen() >= QK4_0))) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
    }
    /* … falls through to nullptr → no repack, use plain ggml_vec_dot … */
}
```

Two important properties:

- **Capability gating is fine-grained**. AVX2, NEON+i8mm, NEON+dotprod, SVE+i8mm, RVV are all separate branches. A target that doesn't meet any branch falls through and the tensor uses the plain `ggml_vec_dot_q4_0_q8_0` kernel.
- **Row-count gating is required**. Each interleave shape needs `ne[1] % N == 0`. Tensors that don't divide evenly fall through to the next branch (or to the un-repacked path).

The tensor traits selected here is stored in `tensor->extra` at `ggml_backend_cpu_repack_buffer_init_tensor` (`repack.cpp:3524`), which fires once when the tensor is allocated to the repack buffer:

```cpp
static enum ggml_status ggml_backend_cpu_repack_buffer_init_tensor(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(
        ggml_repack_get_optimal_repack_type(tensor));
    return GGML_STATUS_SUCCESS;
}
```

The actual byte-level repack happens later, when weights are written into the tensor (the traits class implements a `repack` overload that calls `repack_q4_0_to_q4_0_8_bl` and friends — see `repack.cpp:2335` and `:2489`).

## 5. Why this design is the right template for `Q1_0_g128`

Three properties of the existing pattern that match what we want for `Q1_0_g128` AVX2:

1. **No GGUF format change**. `Bonsai-1.7B.gguf` already on disk — and any future `Q1_0_g128` model — will keep working. Hosts that can use the new packing will repack at load; hosts that can't (CUDA / Metal / Vulkan / NEON / AVX-512) will use their existing fast paths unchanged.
2. **Zero converter changes**. We don't need to touch `convert_hf_to_gguf.py`, `llama-quantize`, or any GGUF tooling. The on-disk schema is unchanged.
3. **Surgical dispatch**. The fix is one new branch in `ggml_repack_get_optimal_repack_type` plus the kernel(s). All existing code paths are untouched.

The cost is real but bounded: we have to write
- one new packed block type (`block_q1_0_g128x4` for AVX2, possibly `_x8` for AVX-VNNI)
- one new repack helper (`make_block_q1_0_g128x4`, `repack_q1_0_g128_to_q1_0_g128_4_bl`)
- one or two new GEMV/GEMM kernels (AVX2-only, plus an AVX-VNNI variant)
- one new branch in the dispatcher
- one test / microbenchmark per kernel

Phases 1b and 1c (the layout doc and the design brief) describe what each of those should look like for `Q1_0_g128`.

## 6. Pointers — where to look in the prism fork

| What                               | Path                                                          |
|------------------------------------|---------------------------------------------------------------|
| Generic packed-block template      | `ggml/src/ggml-cpu/repack.h:13–37`                            |
| Repack helpers (Q4_0)              | `ggml/src/ggml-cpu/repack.cpp:2016`, `:2061`                  |
| Repack converters (Q4_0)           | `ggml/src/ggml-cpu/repack.cpp:2335`, `:2489`                  |
| AVX2/AVX-512 GEMV (Q4_0)           | `ggml/src/ggml-cpu/arch/x86/repack.cpp:1448`                  |
| Dispatch (`get_optimal_repack`)    | `ggml/src/ggml-cpu/repack.cpp:3392`                           |
| Backend buffer registration        | `ggml/src/ggml-cpu/repack.cpp:3619`                           |
| Init hook (per-tensor repack fire) | `ggml/src/ggml-cpu/repack.cpp:3524`                           |
| CPU capability getters             | `ggml/src/ggml-cpu/ggml-cpu.c` — `ggml_cpu_has_avx2`, `…_avx_vnni`, `…_avx512`, `…_neon`, `…_dotprod`, `…_matmul_int8` |
| Arch-fallback rename layer         | `ggml/src/ggml-cpu/arch-fallback.h`                           |
| Existing AVX-512/AVX2 vec_dot      | `ggml/src/ggml-cpu/arch/x86/quants.c:838` (`ggml_vec_dot_q1_0_g128_q8_0`) |
