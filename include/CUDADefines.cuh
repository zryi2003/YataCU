//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef YATACU_CUDADEFINES_CUH
#define YATACU_CUDADEFINES_CUH

#include <cstdint>

namespace YATACU {

    /**
     * @brief 64-bit handle layout:
     * [63:32] Generation - Incremented every time an index is reused to prevent ABA issues.
     * [31:0]  Index      - The actual offset into the resource registry.
     */
    constexpr uint64_t CU_NULL_HANDLE = 0xFFFFFFFFFFFFFFFFULL;

    // Macro helpers for handle manipulation to keep bitwise logic centralized.
    #define CU_GET_INDEX(handle)      ((uint32_t)((handle).id & 0xFFFFFFFFULL))
    #define CU_GET_GENERATION(handle) ((uint32_t)((handle).id >> 32))
    #define CU_PACK_HANDLE(gen, idx)  (((uint64_t)(gen) << 32) | (uint32_t)(idx))

    /**
     * @brief Defines a type-safe opaque handle for CUDA resources.
     * Mimics Vulkan's non-dispatchable handles but with built-in generational tracking.
     */
#define YATACU_DEFINE_HANDLE(Name) \
    struct Name {                  \
        uint64_t id;               \
        __host__ __device__ bool operator==(const Name& other) const { return id == other.id; } \
        __host__ __device__ bool operator!=(const Name& other) const { return id != other.id; } \
    };

} // namespace YATACU

#endif //YATACU_CUDADEFINES_CUH
