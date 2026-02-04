// Comments in English as requested.
//
// IMPORTANT:
// This file defines the CUDA kernel entrypoint `vdf_prove` with a fixed ABI.
// The Rust side expects:
//   - input:  `challenges` as jobs * 8 u32 words (32 bytes per job)
//   - output: `out` as jobs * 50 u32 words (200 bytes per job)
//       - first 25 words (100 bytes): y
//       - second 25 words (100 bytes): witness
//
// This implementation is a *shape-correct scaffold* intended to be replaced by the
// real class-group VDF proof computation. It performs deterministic mixing to
// populate y and witness, which is sufficient to validate the GPU batch plumbing.

#include <stdint.h>

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// A tiny deterministic mixer (not cryptographic).
__device__ __forceinline__ uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
    a ^= rotl32(b, 5);
    a += 0x9E3779B9u;
    a ^= rotl32(c, 13);
    a *= 0x85EBCA6Bu;
    a ^= a >> 16;
    return a;
}

extern "C" __global__ void vdf_prove(const uint32_t* challenges, uint32_t* out, uint32_t jobs) {
    // One thread writes one u32 of the output buffer.
    uint32_t idx = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t total = jobs * 50u;
    if (idx >= total) return;

    uint32_t job = idx / 50u;
    uint32_t lane = idx - job * 50u;

    // Load the 8-word challenge for this job.
    const uint32_t* ch = challenges + job * 8u;
    uint32_t c0 = ch[0];
    uint32_t c1 = ch[1];
    uint32_t c2 = ch[2];
    uint32_t c3 = ch[3];
    uint32_t c4 = ch[4];
    uint32_t c5 = ch[5];
    uint32_t c6 = ch[6];
    uint32_t c7 = ch[7];

    // Deterministic per-lane state.
    uint32_t s = (job * 0xA5A5A5A5u) ^ (lane * 0x3C6EF372u);
    s = mix(s, c0, c7);
    s = mix(s, c1, c6);
    s = mix(s, c2, c5);
    s = mix(s, c3, c4);

    // Fill y (first 25 words) and witness (last 25 words).
    // This preserves the "witness is second half" contract.
    uint32_t v;
    if (lane < 25u) {
        // y
        v = mix(s ^ 0x11111111u, ch[lane & 7u], lane);
    } else {
        // witness
        uint32_t wl = lane - 25u;
        v = mix(s ^ 0x22222222u, ch[wl & 7u], wl ^ 0x5A5Au);
    }

    out[idx] = v;
}
