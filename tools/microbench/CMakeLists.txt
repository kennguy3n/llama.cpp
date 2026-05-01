set(TARGET microbench-q1-g128-repack)
add_executable(${TARGET} q1_g128_repack.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This harness is a stand-alone CLI tool that does NOT link against
# llama.cpp or ggml — it carries its own copies of the relevant block
# layouts so future kernel experiments don't have to rebuild the rest
# of the tree. The only requirement is x86 SIMD intrinsic headers.
#
# We always enable AVX2 + FMA + F16C because Phase 2's whole purpose
# is to compare against the existing AVX2 path. AVX-VNNI is opt-in
# via -DLLAMA_AVX_VNNI=ON; AVX-512 is intentionally NOT enabled here
# because we want this binary to land on AVX2-only hosts (Zen 3,
# Skylake-X-no-VNNI) without the AVX-512 inner loop "winning" the
# microbenchmark by accident.

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|x86")
    target_compile_options(${TARGET} PRIVATE -mavx2 -mfma -mf16c)

    option(LLAMA_MICROBENCH_AVX_VNNI
        "Enable AVX-VNNI in the q1_g128 microbench (Alder Lake P-cores, Zen 4)"
        OFF)
    if (LLAMA_MICROBENCH_AVX_VNNI)
        target_compile_options(${TARGET} PRIVATE -mavxvnni)
    endif()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
