//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef WAVEFRONTRENDERER_CUDADEFINES_CUH
#define WAVEFRONTRENDERER_CUDADEFINES_CUH

namespace YATACU {

#define YATACU_DEFINE_HANDLE(Name) \
    struct Name {                  \
        uint32_t id;               \
        bool operator==(const Name& other) const { return id == other.id; } \
        bool operator!=(const Name& other) const { return id != other.id; } \
    };

    YATACU_DEFINE_HANDLE(CuBuffer)
    YATACU_DEFINE_HANDLE(CuStream)

    constexpr uint32_t CU_NULL_HANDLE = 0xFFFFFFFF;

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

}

#endif //WAVEFRONTRENDERER_CUDADEFINES_CUH