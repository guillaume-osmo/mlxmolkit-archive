#include <metal_stdlib>
using namespace metal;

/// Per-row popcount of uint32 packed fingerprints.
/// grid: (N, 1, 1), one thread per row.
kernel void popcount_rows(
    device const uint  *a       [[buffer(0)]],
    device       uint  *cnt     [[buffer(1)]],
    constant     uint  &N       [[buffer(2)]],
    constant     uint  &nwords  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    uint sum = 0;
    for (uint k = 0; k < nwords; k++) {
        sum += popcount(a[tid * nwords + k]);
    }
    cnt[tid] = sum;
}

/// Pass 1: count neighbors above cutoff per row.
/// grid: (N, 1, 1), one thread per row.
kernel void fused_tanimoto_count(
    device const uint  *a       [[buffer(0)]],
    device const uint  *cnt     [[buffer(1)]],
    device       int   *num_nb  [[buffer(2)]],
    constant     uint  &N       [[buffer(3)]],
    constant     uint  &nwords  [[buffer(4)]],
    constant     float &cutoff  [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    uint cnt_i = cnt[tid];
    int count = 0;
    for (uint j = 0; j < N; j++) {
        if (j == tid) continue;
        uint inter = 0;
        for (uint k = 0; k < nwords; k++) {
            inter += popcount(a[tid * nwords + k] & a[j * nwords + k]);
        }
        uint union_ = cnt_i + cnt[j] - inter;
        float sim = (float)inter / ((float)union_ + 1e-12f);
        if (sim >= cutoff) count++;
    }
    num_nb[tid] = count;
}

/// Pass 2: fill neighbor indices into pre-allocated CSR.
/// grid: (N, 1, 1), one thread per row.
kernel void fused_tanimoto_fill(
    device const uint  *a       [[buffer(0)]],
    device const uint  *cnt     [[buffer(1)]],
    device const int   *offsets [[buffer(2)]],
    device       int   *indices [[buffer(3)]],
    constant     uint  &N       [[buffer(4)]],
    constant     uint  &nwords  [[buffer(5)]],
    constant     float &cutoff  [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    uint cnt_i = cnt[tid];
    int base = offsets[tid];
    int pos = 0;
    for (uint j = 0; j < N; j++) {
        if (j == tid) continue;
        uint inter = 0;
        for (uint k = 0; k < nwords; k++) {
            inter += popcount(a[tid * nwords + k] & a[j * nwords + k]);
        }
        uint union_ = cnt_i + cnt[j] - inter;
        float sim = (float)inter / ((float)union_ + 1e-12f);
        if (sim >= cutoff) {
            indices[base + pos] = (int)j;
            pos++;
        }
    }
}
