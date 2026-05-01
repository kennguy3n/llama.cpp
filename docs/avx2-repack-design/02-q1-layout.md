# Phase 1b: `block_q1_0_g128` layout, hot path, and where AVX2 leaves perf on the table

**Status**: Research only. No code changes.
**Audience**: Reviewer of the planned `Q1_0_g128` AVX2 / AVX-VNNI repack.
**Goal**: Inventory the existing prism `Q1_0_g128` representation and SIMD code paths so the design brief (`03-design-brief.md`) can describe a layout transformation that preserves correctness on every existing backend.

## 1. Block format and total bit budget

`ggml/src/ggml-common.h:184–191` defines the on-disk block:

```c
#define QK1_0_g128 128
typedef struct {
    ggml_half d;                  // f16 row scale
    uint8_t   qs[QK1_0_g128 / 8]; // 16 bytes = 128 packed weight bits
} block_q1_0_g128;
static_assert(sizeof(block_q1_0_g128) == sizeof(ggml_half) + QK1_0_g128 / 8,
              "wrong q1_0_g128 block size/padding");
```

So one block:

| Field   | Bytes | Purpose                                                              |
|---------|------:|----------------------------------------------------------------------|
| `d`     | 2     | f16 scale shared by all 128 weights (group size 128)                |
| `qs`    | 16    | 128 packed weight bits — bit 1 ⇒ weight = +1, bit 0 ⇒ weight = -1   |
| total   | 18    |                                                                      |

The activation block `block_q8_0` is the standard ggml 32-element int8 group (`d:2 + qs:32 = 34` bytes). One `q1_0_g128` block always pairs with **four** consecutive `q8_0` blocks (`128 / 32 = 4`) — see the inner `for (int k = 0; k < 4; ++k)` loop in every Q1 vec_dot implementation.

Memory cost per output row of Bonsai-1.7B (1.7 B params, ~96 % of weights are these blocks): `1.7e9 / 128 × 18 ≈ 240 MiB` total → matches the 248 MB GGUF on disk.

## 2. The dispatch layer (single-row vec_dot today)

In ggml today, `Q1_0_g128` only has a single-row `ggml_vec_dot_*` kernel; no GEMV / GEMM specialisation. That means every output row of a matmul re-issues all the activation loads, which is exactly the hot spot the repack pattern is designed to fix.

| Backend         | Kernel location                                                                | Instruction count per 32 weights |
|-----------------|--------------------------------------------------------------------------------|---------------------------------:|
| CUDA            | `ggml/src/ggml-cuda/{convert.cu,common.cuh}` (`dequantize_q1_0_g128`)          | n/a (warp-parallel)              |
| Metal           | `ggml/src/ggml-metal/ggml-metal-{ops,device}.cpp`                              | n/a (SIMD-group)                 |
| AVX-512 + VNNI  | `ggml/src/ggml-cpu/arch/x86/quants.c:855` (the fast path I'm not changing)     | **3** (sub + mask_blend + dpbusd) |
| AVX2 (no VNNI)  | `ggml/src/ggml-cpu/arch/x86/quants.c:893`                                      | **10** (see §3 below)            |
| ARM NEON        | `ggml/src/ggml-cpu/arch/arm/quants.c` — present, optimised for `vmlal`         | ~4–5                             |
| RISC-V V        | `ggml/src/ggml-cpu/arch/riscv/quants.c` — present, vsetvli-driven              | ~6                               |
| Scalar fallback | `ggml/src/ggml-cpu/arch/x86/quants.c:944` and per-arch                          | ~32 (one ALU op per bit)         |

**The repack work is a strictly additive change against the AVX2-no-VNNI path**: I only need to introduce a faster path that fires when `cpu_has_avx2 && !cpu_has_avx512`, leaving every other backend untouched. AVX-VNNI is a separate, also-additive branch.

## 3. The current AVX2 inner loop, instruction by instruction

`ggml/src/ggml-cpu/arch/x86/quants.c:893–941`:

```c
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();

    const __m256i shuffle_mask = _mm256_set_epi8(
        3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
    const __m256i bit_mask = _mm256_set_epi8(
        (char)0x80,0x40,0x20,0x10,0x08,0x04,0x02,0x01, /* repeat 4× */ ...);
    const __m256i ones = _mm256_set1_epi8(1);

    for (; ib < nb; ++ib) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        for (int k = 0; k < 4; ++k) {
            const float d1 = GGML_CPU_FP16_TO_FP32(y[ib*4 + k].d);
            const __m256 d  = _mm256_set1_ps(d0 * d1);

            uint32_t qbits32;
            memcpy(&qbits32, x[ib].qs + k*4, sizeof(qbits32));
            const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib*4+k].qs);

            // ── Bit-expansion: 32 bits → 32 bytes of ±1 (six SIMD ops) ──
            const __m128i qbits_128   = _mm_set1_epi32(qbits32);              // 1
            const __m256i qbits_256   = _mm256_broadcastsi128_si256(qbits_128); // 2
            const __m256i qbits_shuf  = _mm256_shuffle_epi8(qbits_256, shuffle_mask); // 3
            const __m256i bit_test    = _mm256_and_si256(qbits_shuf, bit_mask);   // 4
            const __m256i is_set      = _mm256_cmpeq_epi8(bit_test, bit_mask);    // 5
            const __m256i bit_value   = _mm256_and_si256(is_set, ones);          // 6
            const __m256i bit_doubled = _mm256_add_epi8(bit_value, bit_value);   // 7
            const __m256i qx          = _mm256_sub_epi8(bit_doubled, ones);     // 8

            // ── Dot product: 32×i8 × 32×i8 → 8×i32 (~3 ops via maddubs+madd+add) ──
            acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc); // 9-11+
        }
    }
```

For each 32-element sub-block we issue ~10 256-bit SIMD ops, vs. ~3 on the AVX-512 path. On Zen 3 (4 vector ports, 1c throughput on most ALU ops), best-case throughput is `~3.2 GHz × 4 ports / 10 ops × 32 weights/iter ≈ 4.1 G weights/sec/thread`. With 1.7 B weights/token of prefill that's ~410 ms/token/thread, ~100 ms/token on 4 threads — matches the measured 23 t/s prefill almost exactly. **The AVX2 ceiling is real and the kernel is not leaving cycles on the floor through naivete; the bottleneck is the bit-expansion sequence (ops 1–8) plus the 3-op signed dot-product epilogue.**

## 4. Three places perf could come from, ordered by ease

The repack design (`03-design-brief.md`) will combine some subset of these. Listed here for reference; numbers are estimates I'll validate in Phase 2 with a microbenchmark.

### 4a. Replace the ±1 expansion with a direct masked sum

Mathematical identity: `dot(±1_vector, qy) = 2·sum(qy where bit=1) − sum(qy)`.

`sum(qy)` is a per-`q8_0`-block constant (no weight involvement). `sum(qy where bit=1)` reduces to one `_mm256_blendv_epi8(zero, qy, mask)` followed by horizontal-sum. The mask is `_mm256_cmpeq_epi8(bit_test, bit_mask)` — already computed today (op 5 above).

Inner loop drops from ops 1–11 to **ops 1–5 (mask construction) + blendv + horizontal-sum (~3) ≈ 8 ops** — modest 20 % win on its own, but it eliminates the dependency chain through `vpmaddubsw / vpmaddwd`, freeing ports for the next sub-block. Combined with row interleave (4b) the win is much larger.

### 4b. Row interleave — the core lever

Today, each output row issues an independent `_mm256_loadu_si256` of `qy`. With 4-row interleave (the `block_q1_0_g128x4` layout):

- One `qy` load serves 4 output rows in parallel (saves 3 of every 4 activation loads).
- The 4 weight masks for the 4 rows can come from a single 128-bit broadcast load if we lay out `qbits32` × 4 rows as a 16-byte block.
- Inner loop accumulates 4 partial sums in 4 register accumulators.

This is the same structural win that took mainline `Q4_0` from ~30 t/s to ~50–60 t/s on AVX2-only hosts. Realistic upside on `Q1_0_g128`: 23 t/s → 35–45 t/s (~1.5–2× over the current single-row path).

### 4c. AVX-VNNI fast path on consumer-AVX2 CPUs

AVX-VNNI is the AVX2-encoded subset of VNNI's `vpdpbusd` family. It exists on Intel Alder Lake P-cores, Raptor Lake, and Zen 4 — none of which have full AVX-512 enabled in consumer parts. Adding this branch:

```c
#elif defined(__AVXVNNI__)
    /* exactly the AVX-512 fast path, minus the __mmask32 blend.
       The bit-expansion still has to use blendv_epi8 (no mask regs in AVX),
       but the dot product collapses from 3 ops to 1: vpdpbusd. */
```

Inner loop drops from ~10 ops to ~5 ops. On Alder Lake P-cores / Zen 4, expected throughput is roughly the AVX-512 path minus a small per-iter overhead (no mask register blend) — call it 0.85× the AVX-512 ceiling.

**Important: this branch does nothing for the Zen 3 / EPYC 7763 demo VM**, which has neither AVX-512 nor AVX-VNNI. It's a separate win for a different class of host. Ken's selection (`AVX2-only + AVX-VNNI consumer`) deliberately covers both.

## 5. Cross-arch invariants the repack must preserve

The new repack only fires on x86-64 hosts that satisfy `cpu_has_avx2 && !cpu_has_avx512` (or the AVX-VNNI subset). Every other backend must be unaffected:

- **CUDA**: `ggml-cuda/convert.cu:677,738` and `common.cuh:935` already dispatch on `GGML_TYPE_Q1_0_g128`. We don't touch them.
- **Metal**: `ggml-metal-device.cpp:705,932` already handles the type. We don't touch it.
- **Vulkan**: same — dispatch already in place in the fork.
- **ARM NEON**: present in `arch/arm/quants.c`; tensors are not allocated to the CPU repack buffer on ARM hosts because the dispatcher in `repack.cpp:3392` won't return a tensor_traits for our new entry on `!ggml_cpu_has_avx2()`.
- **AVX-512** and **AVX-512 VNNI** hosts: dispatcher must explicitly skip the new entry when AVX-512 is present, since the existing `ggml_vec_dot_q1_0_g128_q8_0` AVX-512 fast path (`x86/quants.c:855`) is faster than any AVX2 GEMV we can write.

## 6. Concrete starting point for Phase 1c

The design brief should answer:

1. **Block layout for `block_q1_0_g128x4`** — exact byte ordering of the four interleaved deltas + 64 packed bits per row.
2. **Repack helper signature** — `make_block_q1_0_g128x4(block_q1_0_g128 * in)` and `repack_q1_0_g128_to_q1_0_g128_4_bl(...)`.
3. **GEMV signature(s)** — `ggml_gemv_q1_0_g128_4x4_q8_0` (AVX2) and optionally `ggml_gemv_q1_0_g128_8x8_q8_0` (AVX-VNNI).
4. **Dispatcher branch** — exact `if` block to add to `ggml_repack_get_optimal_repack_type`, gated on capability + `ne[1] % 4 == 0`.
5. **AVX-VNNI variant** — same family, separate kernel, separate dispatcher branch ordered before the AVX2-only one.
6. **Test plan** — round-trip the Bonsai-1.7B GGUF through the repacked kernel and assert bit-identical outputs vs. the existing single-row vec_dot.
7. **Microbenchmark plan** — Phase 2 deliverable; standalone harness comparing inner-loop cycles between the four variants on this VM.

That brief follows in `03-design-brief.md`.
