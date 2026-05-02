# Phase 7 follow-up #3 — AVX-512 VNNI Kernel C investigation (doc-only)

> **Status (2026-05-01):** Microbench-only experiment. Kernel C lives in
> `tools/microbench/q1_g128_repack.cpp` behind `LLAMA_MICROBENCH_AVX512_VNNI=ON`.
> The in-tree dispatcher (`ggml/src/ggml-cpu/repack.cpp:3702-3712`) is
> intentionally unchanged — Kernel C is not wired into `arch/x86/repack.cpp`
> and is not reachable at runtime from a normal build.
>
> The decision to defer the dispatcher change is documented in §Decision
> below. The kernel and its evidence are preserved here so that whoever
> next has access to a Cascade Lake / Ice Lake / Sapphire Rapids / Zen 4
> host can pick up the experiment without re-writing it from scratch.

> **AI authorship disclosure** (per `AGENTS.md` / `CONTRIBUTING.md`
> preamble for the prism private fork): the microbench code, this
> document, and the supporting verification commands were drafted with
> AI assistance and reviewed before commit.

## Background

PR #12 (Phase 3, the original Q1_0_g128 4×4 AVX2 runtime repack) added
this dispatcher gate at `ggml/src/ggml-cpu/repack.cpp:3702-3712`:

```c++
} else if (cur->type == GGML_TYPE_Q1_0_g128) {
    // Q1_0_g128 already has a hand-tuned single-row AVX-512 path in
    // ggml/src/ggml-cpu/arch/x86/quants.c that beats the 4-row repack
    // wherever AVX-512BW is available. Only opt into the runtime
    // repack on hosts where that fast path is unavailable, i.e. AVX2
    // (with or without AVX-VNNI) but no AVX-512.
    if (ggml_cpu_has_avx2() && !ggml_cpu_has_avx512()) {
        if (cur->ne[1] % 4 == 0) {
            return &q1_0_g128_4x4_q8_0;
        }
    }
}
```

The "single-row AVX-512 path beats the 4-row repack" claim was an
**unmeasured assumption** when PR #12 was authored — at the time the
4-row repack consisted of the original "GEMV in a loop, on-the-fly
unpack" kernel from PR #12, which we now know was 25.3 t/s on Zen 3.
PR #13 replaced that with a true 4×4 GEMM (Kernel A: 30.9 t/s, 1.36×
over the original single-row AVX2 baseline). PR #17 added the AVX-VNNI
variant (Kernel B; SDE-validated, no real-hardware perf yet).

Follow-up #3 asks: with the modern 4×4 GEMM design from PR #13, would
an AVX-512 VNNI variant — call it Kernel C — also beat the existing
single-row AVX-512 path? If yes, the dispatcher gate should be widened
to also opt into the repack on AVX-512+VNNI hosts.

## Kernel C design

Same outer skeleton as Kernel B (4 weight rows × 4 sub-blocks, partial
sums kept in `__m256` accumulators across the inner loop, FP32 scaling
folded in via `fmadd`). Two AVX-512BW+VL features that AVX-VNNI alone
does not have are exploited:

1. **`_mm256_mask_blend_epi8`**: the 32-bit `qbits` value is consumed
   directly as a `__mmask32`. This eliminates the 5-instruction
   `set1`+`broadcast`+`shuffle`+`and`+`cmpeq` sequence that Kernel B
   needs to expand `qbits32` to a 256-bit byte mask before `blendv`.
   Net: −4 ops per `(sb, r_w)` for the mask path.

2. **EVEX-encoded `_mm256_dpbusd_epi32`** replaces Kernel B's VEX
   `_mm256_dpbusd_avx_epi32`. Same per-instruction throughput on hosts
   that have both encodings (Sapphire Rapids, Zen 4, Tiger Lake), but
   Kernel C is also reachable on **Cascade Lake / Ice Lake**, which
   have AVX-512+VNNI but not AVX-VNNI — those are the hosts where the
   existing single-row AVX-512 vec_dot would otherwise be the only
   non-scalar path.

The kernel is intentionally 256-bit-only, matching the existing
`vec_dot_q1_0_g128_q8_0` AVX-512 path's
`// stay in 256-bit to avoid lane-crossing overhead` comment. A
512-bit variant was considered (see §Alternatives) but the structural
gain is small enough that it is not worth the added register
pressure or the licensed-mode downclock penalty on Skylake-X / Cascade
Lake silicon.

Source: `tools/microbench/q1_g128_repack.cpp` §`Kernel (f)`,
behind `#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__)`.

## Parity validation

Build:

```bash
cmake -B build -DLLAMA_MICROBENCH_AVX512_VNNI=ON
cmake --build build --target microbench-q1-g128-repack -j
```

Run under SDE Ice Lake (AVX-512F+BW+VL+VNNI):

```bash
sde64 -icx -- ./build/bin/microbench-q1-g128-repack --blocks 64 --iters 8
```

Result (`tools/microbench/parity-icx-sde-kernel-c.txt`):

```
Parity check (4 rows, scalar baseline):
  row 0  scalar=    -40.244610  avx2_single=    -40.244576  gemv_avx2=    -40.244576
                                  avx512_single=    -40.244576  gemv_avx512=    -40.244576  OK
  row 1  scalar=    -68.837616  avx2_single=    -68.837593  gemv_avx2=    -68.837593
                                  avx512_single=    -68.837593  gemv_avx512=    -68.837593  OK
  row 2  scalar=     64.643204  avx2_single=     64.643234  gemv_avx2=     64.643234
                                  avx512_single=     64.643234  gemv_avx512=     64.643234  OK
  row 3  scalar=     -0.807468  avx2_single=     -0.807454  gemv_avx2=     -0.807454
                                  avx512_single=     -0.807454  gemv_avx512=     -0.807454  OK
Parity: OK
```

Five-way agreement: scalar reference = single-row AVX2 vec_dot =
single-row AVX-512 VNNI vec_dot = Kernel A 4×4 GEMV = Kernel C 4×4
GEMV. All AVX paths are bit-identical to one another (the small
delta vs scalar is FP16 roundoff in the order of summation).

The same parity also passes under SDE Cascade Lake (`sde64 -clx`) —
the canonical host class for Kernel C, where Kernel B (AVX-VNNI)
cannot compile. See the bottom of
`tools/microbench/parity-icx-sde-kernel-c.txt`.

## Structural-perf measurement

Without real AVX-512 hardware, wall-clock numbers from SDE are
emulator timings, not CPU timings, and are not informative (the file
above shows `vec_dot_avx512_vnni` running ~10× slower than
`vec_dot_avx2` under SDE — that is purely emulation overhead, the
opposite of what real silicon would do).

The structural-perf proxy used here is **dynamic instructions
executed**, captured by `sde64 -icx -mix`. Each function processed
bit-for-bit identical input and produced bit-for-bit identical
output (parity proven above), so the ratio of dynamic ICOUNTs is a
pure-structural comparison.

Workload: `--blocks 4096 --iters 4`, i.e. 4 timing iterations over
524 288-weight columns, plus the parity-test pass and warmup. Both
the single-row and 4×4 paths see the same total weight-row data
(81 920 single-row calls × 1 row each ≡ 20 480 4×4 calls × 4 rows
each).

| Function                              | Dynamic ICOUNT | Ratio vs single-row AVX-512 |
|---------------------------------------|---------------:|----------------------------:|
| `vec_dot_avx2` (single-row, AVX2)     |      7,209,420 | 1.572×                      |
| `vec_dot_avx512_vnni` (single-row)    |      4,587,920 | **1.000× (the floor)**      |
| `gemv_4x4_avx2` (Kernel A, in-tree)   |      5,898,600 | 1.286×                      |
| `gemv_4x4_avx512_vnni` (Kernel C)     |  **3,768,655** | **0.821× → 1.22× speedup**  |

Headline:

* **Kernel C executes 17.9 % fewer instructions than the existing
  single-row AVX-512 vec_dot** for the same workload — a 1.22×
  structural speedup over the floor we are trying to beat.
* **Kernel C executes 36.1 % fewer instructions than Kernel A** (the
  in-tree AVX2 4×4 GEMM that PR #13 ships) — a 1.56× structural
  speedup that comes entirely from the two AVX-512 features described
  in §Kernel C design.
* AVX-512 alone (single-row) is 1.57× ahead of AVX2 alone
  (single-row), confirming that the existing in-tree AVX-512 path is
  already a solid baseline.

## Decision: do not (yet) widen the dispatcher gate

The structural numbers favour Kernel C, but two factors argue
against landing it as a default in-tree path right now:

1. **Decode-side memory amplification.** The dispatcher's
   `tensor_traits` repack happens at allocation time, transforming
   the tensor's storage layout from `block_q1_0_g128` (18 B/block) to
   `block_q1_0_g128x4` (72 B/block, 4 rows interleaved). Once the
   tensor is repacked, the single-row vec_dot path is no longer
   reachable: every dot product, even at decode time (`nrows = 1`),
   goes through the 4×4 GEMV.

   At decode the 4×4 GEMV reads 4× more weight bytes than the
   single-row vec_dot needs (computes 4 output rows in parallel,
   only 1 is consumed). The compute amortises away on prefill
   (`nrows ≥ 4`, every output is used), but at decode it is wasted
   bandwidth. For Bonsai-1.7B-Q1_0_g128 the decode token-rate is
   the user-perceived latency, so a regression there would be felt
   directly.

2. **No real-hardware data.** All numbers above are SDE
   instruction-counts, which model neither port pressure
   (`vpdpbusd` issue rate, `vpblendmb` mask-uop cost) nor cache /
   memory bandwidth. The in-tree decision to enable repack on AVX-512
   hosts should be backed by an actual prefill-vs-decode wall-clock
   measurement on at least one of {Cascade Lake, Ice Lake, Sapphire
   Rapids, Zen 4}.

The doc-only PR therefore lands the kernel in the microbench, the
parity proof, and the structural-perf numbers — but does **not**
flip the dispatcher gate. The follow-up to flip the gate is left
for whoever has hardware access; their workflow is roughly:

```text
  # On the AVX-512+VNNI host:
  cmake -B build -DLLAMA_MICROBENCH_AVX512_VNNI=ON \
        -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON
  cmake --build build --target microbench-q1-g128-repack llama-bench llama-cli -j

  # 1. Microbench: confirm Kernel C >= single-row AVX-512 in wall-clock.
  ./build/bin/microbench-q1-g128-repack --blocks 4096 --iters 1024

  # 2. End-to-end: bench Bonsai-1.7B-Q1_0_g128 with the dispatcher gate
  #    *unchanged* to capture the AVX-512 single-row baseline.
  ./build/bin/llama-bench -m Bonsai-1.7B.gguf -p 512 -n 64 -t <real_cores>

  # 3. Patch the dispatcher gate (one-line change in repack.cpp:3708) to
  #    also enable on AVX-512+VNNI hosts:
  #      if (ggml_cpu_has_avx2() &&
  #          (!ggml_cpu_has_avx512() ||
  #           (ggml_cpu_has_avx512_bw() && ggml_cpu_has_avx512_vnni())))
  #    Rebuild and re-run llama-bench.

  # 4. Compare prefill (pp512) AND decode (tg64) numbers separately.
  #    Land the gate change only if (a) prefill is +>=10 % faster, AND
  #    (b) decode is no worse than -2 % slower. Otherwise the existing
  #    single-row path stays the default.
```

## Alternatives considered

### 512-bit Kernel C (rejected for now)

Concept: use `_mm512_dpbusd_epi32` (16 int32 lanes per dpbusd) and
`_mm512_mask_blend_epi8` (`__mmask64` from concatenated 64-bit qbits)
to fold 2 sub-blocks worth of qy into one dpbusd. Each sub-block has
its own `d_y` so the int32 result has to be split per-sub-block
before scaling, adding 1 extract per `(r_a, sb_pair, r_w)`.

Per-sub-block op estimate: ~21 ops / sb (vs ~23 ops / sb for 256-bit
Kernel C in the GEMV with 4 r_a). Marginal win on paper, but:

* Adds `vextracti64x4` to each (r_a, r_w) chain, adding latency.
* Skylake-X / Cascade Lake silicon down-clocks when 512-bit code is
  executed (Intel licensed-mode penalty); on those hosts the
  256-bit variant likely dominates.
* Modern Intel (Ice Lake+) and AMD (Zen 4) don't have the
  licensed-mode penalty, but `vpdpbusd` is single-uop and
  port-bound on the same execution port whether you encode it
  256-bit or 512-bit, so the wall-clock advantage is small.

Defer to whoever benches Kernel C on real hardware first; if
256-bit Kernel C clearly wins, 512-bit can be a follow-up.

### Skip the whole experiment (rejected)

The argument here is "the existing single-row AVX-512 path is already
hand-tuned, why touch it?". The answer: the dispatcher gate that
prevents repack on AVX-512 hosts was added in PR #12 based on an
**unmeasured** assumption about a kernel that didn't exist yet
(Kernel A wasn't in-tree until PR #13). The structural-perf evidence
above shows the assumption is questionable — Kernel C does
measurably less work than the existing AVX-512 path. Documenting
that finding has independent value even if no one ports the kernel
into the dispatcher.

## Reproducing the numbers

```bash
# In the prism repo root, on any x86_64 host with Intel SDE 9.48+:
cmake -B build-avx512-microbench \
    -DGGML_NATIVE=OFF \
    -DLLAMA_MICROBENCH_AVX512_VNNI=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-avx512-microbench --target microbench-q1-g128-repack -j

# Parity (Ice Lake):
sde64 -icx -- ./build-avx512-microbench/bin/microbench-q1-g128-repack \
    --blocks 64 --iters 8

# Parity (Cascade Lake — the host class without AVX-VNNI):
sde64 -clx -- ./build-avx512-microbench/bin/microbench-q1-g128-repack \
    --blocks 64 --iters 8

# Structural-perf (Ice Lake -mix). Output goes to sde-mix-icx.txt; per-
# function aggregate is in the EMIT_GLOBAL_DYNAMIC_STATS section.
sde64 -icx -mix -omix sde-mix-icx.txt -- \
    ./build-avx512-microbench/bin/microbench-q1-g128-repack \
    --blocks 4096 --iters 4 > /dev/null
grep -E "vec_dot|gemv_q1g128" sde-mix-icx.txt | grep -E "^ +[0-9]+:"
```

## Companion files

* `tools/microbench/q1_g128_repack.cpp` — Kernel C source
  (search for `// Kernel (f)` and `gemv_q1g128_4x4_avx512_vnni`).
* `tools/microbench/CMakeLists.txt` — adds the `LLAMA_MICROBENCH_AVX512_VNNI`
  CMake option that turns on `-mavx512f -mavx512bw -mavx512vl -mavx512vnni`
  for the microbench TU.
* `tools/microbench/parity-icx-sde-kernel-c.txt` — verbatim SDE
  parity outputs (Ice Lake and Cascade Lake) plus the
  structural-perf summary table.
