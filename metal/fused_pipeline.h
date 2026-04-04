#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fused Tanimoto + CSR neighbor list pipeline using pre-compiled Metal.
 *
 * Two-pass approach:
 *   1. popcount_rows → fused_tanimoto_count → build CSR offsets
 *   2. fused_tanimoto_fill → CSR indices
 *
 * All GPU dispatch is done internally via Metal API.
 *
 * @param fp_u32     Input fingerprints, row-major (N x nwords), uint32
 * @param N          Number of molecules
 * @param nwords     Number of uint32 words per fingerprint
 * @param cutoff     Tanimoto similarity threshold
 * @param offsets    Output: caller-allocated (N+1) int32 array for CSR offsets
 * @param indices    Output: pointer set to malloc'd int32 array of neighbor indices
 * @param n_edges    Output: total number of edges (size of *indices)
 * @param gpu_ms     Output: total GPU time in milliseconds
 * @return           0 on success, non-zero on error
 */
int fused_tanimoto_pipeline(
    const unsigned int* fp_u32,
    int N,
    int nwords,
    float cutoff,
    int* offsets,
    int** indices,
    int* n_edges,
    double* gpu_ms);

void free_indices(int* ptr);

#ifdef __cplusplus
}
#endif
