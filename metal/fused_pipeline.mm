/**
 * Native Metal pipeline: pre-compiled .metallib → direct dispatch.
 * Exposes a C API callable from Python via ctypes.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <dlfcn.h>
#include <filesystem>

#include "fused_pipeline.h"

using Clock = std::chrono::high_resolution_clock;

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pso_popcount = nil;
static id<MTLComputePipelineState> g_pso_count = nil;
static id<MTLComputePipelineState> g_pso_fill = nil;

static int ensure_init() {
    if (g_device) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return -1;

    // Find metallib next to this dylib
    Dl_info info;
    if (!dladdr((void*)&ensure_init, &info)) return -2;
    std::string dir = std::filesystem::path(info.dli_fname).parent_path().string();
    NSString* path = [NSString stringWithFormat:@"%s/fused_tanimoto.metallib",
                      dir.c_str()];

    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:path];
    id<MTLLibrary> lib = [g_device newLibraryWithURL:url error:&err];
    if (!lib) return -3;

    g_pso_popcount = [g_device newComputePipelineStateWithFunction:
                      [lib newFunctionWithName:@"popcount_rows"] error:&err];
    g_pso_count = [g_device newComputePipelineStateWithFunction:
                   [lib newFunctionWithName:@"fused_tanimoto_count"] error:&err];
    g_pso_fill = [g_device newComputePipelineStateWithFunction:
                  [lib newFunctionWithName:@"fused_tanimoto_fill"] error:&err];

    if (!g_pso_popcount || !g_pso_count || !g_pso_fill) return -4;

    g_queue = [g_device newCommandQueue];
    return 0;
}


extern "C" int fused_tanimoto_pipeline(
    const unsigned int* fp_u32,
    int N,
    int nwords,
    float cutoff,
    int* offsets,
    int** indices,
    int* n_edges,
    double* gpu_ms)
{
    int rc = ensure_init();
    if (rc != 0) return rc;

    auto t0 = Clock::now();
    NSUInteger tgSize = std::min(256, N);

    size_t fp_sz = (size_t)N * nwords * sizeof(uint32_t);
    id<MTLBuffer> buf_fp = [g_device newBufferWithBytes:fp_u32 length:fp_sz
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_cnt = [g_device newBufferWithLength:N * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_num = [g_device newBufferWithLength:N * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
    uint32_t uN = (uint32_t)N;
    uint32_t uNW = (uint32_t)nwords;
    id<MTLBuffer> buf_N  = [g_device newBufferWithBytes:&uN length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_NW = [g_device newBufferWithBytes:&uNW length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_C  = [g_device newBufferWithBytes:&cutoff length:4 options:MTLResourceStorageModeShared];

    // Pass 0: popcount
    {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso_popcount];
        [enc setBuffer:buf_fp  offset:0 atIndex:0];
        [enc setBuffer:buf_cnt offset:0 atIndex:1];
        [enc setBuffer:buf_N   offset:0 atIndex:2];
        [enc setBuffer:buf_NW  offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Pass 1: count
    {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso_count];
        [enc setBuffer:buf_fp  offset:0 atIndex:0];
        [enc setBuffer:buf_cnt offset:0 atIndex:1];
        [enc setBuffer:buf_num offset:0 atIndex:2];
        [enc setBuffer:buf_N   offset:0 atIndex:3];
        [enc setBuffer:buf_NW  offset:0 atIndex:4];
        [enc setBuffer:buf_C   offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Build CSR offsets (CPU)
    int32_t* num_ptr = (int32_t*)[buf_num contents];
    offsets[0] = 0;
    for (int i = 0; i < N; i++)
        offsets[i+1] = offsets[i] + num_ptr[i];
    int total = offsets[N];
    *n_edges = total;

    if (total == 0) {
        *indices = nullptr;
        auto t1 = Clock::now();
        *gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 0;
    }

    // Pass 2: fill
    id<MTLBuffer> buf_off = [g_device newBufferWithBytes:offsets
                                                  length:(N+1)*sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_idx = [g_device newBufferWithLength:total * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
    {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso_fill];
        [enc setBuffer:buf_fp  offset:0 atIndex:0];
        [enc setBuffer:buf_cnt offset:0 atIndex:1];
        [enc setBuffer:buf_off offset:0 atIndex:2];
        [enc setBuffer:buf_idx offset:0 atIndex:3];
        [enc setBuffer:buf_N   offset:0 atIndex:4];
        [enc setBuffer:buf_NW  offset:0 atIndex:5];
        [enc setBuffer:buf_C   offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(tgSize,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    auto t1 = Clock::now();
    *gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    *indices = (int*)malloc(total * sizeof(int32_t));
    memcpy(*indices, [buf_idx contents], total * sizeof(int32_t));

    return 0;
}

extern "C" void free_indices(int* ptr) {
    free(ptr);
}
