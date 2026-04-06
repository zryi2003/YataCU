//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef YATACU_CUDASPECS_CUH
#define YATACU_CUDASPECS_CUH

#include "CUDADefines.cuh"
#include <cuda_runtime.h>

namespace YATACU {

    // --- [Handles] ---
    // Opaque 64-bit handles with generational tracking.
    YATACU_DEFINE_HANDLE(CuBuffer)
    YATACU_DEFINE_HANDLE(CuStream)

    // --- [Enums] ---
    enum class CuMemoryUsage {
        // Device-local memory; physically resident on the GPU.
        // Inaccessible from Host directly; provides highest bandwidth.
        GPU_ONLY,

        // Page-locked (pinned) memory; host-visible and accessible by GPU via DMA.
        // Optimized for high-bandwidth transfers between Host and Device.
        CPU_PINNED,

        // Unified Memory; driver-managed data migration.
        // Accessible by both Host and Device with on-demand page faulting.
        GPU_MANAGED
    };

    // --- [Configurations] ---

    /**
     * @brief Parameters for creating a CuBuffer.
     */
    struct CuBufferCreateInfo {
        size_t size = 0;
        CuMemoryUsage usage = CuMemoryUsage::GPU_ONLY;
        const char* debugName = "Unnamed_Buffer";

        // VMA-like extension points can be added here
    };

    /**
     * @brief Parameters for kernel dispatching.
     * Mimics a subset of Vulkan's pipeline/dispatch state.
     */
    struct CuDispatchInfo {
        dim3 gridDim = {1, 1, 1};   // Explicit grid dimensions.
        dim3 blockDim = {256, 1, 1}; // Explicit block dimensions.
        size_t sharedMemBytes = 0;  // Dynamic shared memory size in bytes.
    };

} // namespace YATACU

#endif //YATACU_CUDASPECS_CUH
