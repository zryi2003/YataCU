//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef YATACU_CUDADEVICE_CUH
#define YATACU_CUDADEVICE_CUH
#include <CUDADefines.cuh>
#include <CUDAError.cuh>
#include <CUDAMemory.cuh>
#include <source_location>
#include <string>
#include <vector>

namespace YATACU {

    struct CuBufferCreateInfo {
        size_t size = 0;
        CuMemoryUsage usage = CuMemoryUsage::GPU_ONLY;
        const char* debugName = "Unnamed_Buffer";

        // TODO: VMA-like memory allocator. Extra parameters, like alignment can be added
    };


    class CuDevice {
    public:
        CuDevice();
        ~CuDevice();

        CuDevice(const CuDevice&) = delete;
        CuDevice& operator=(const CuDevice&) = delete;

        CuBuffer CreateBuffer(
            const CuBufferCreateInfo& info,
            std::source_location loc = std::source_location::current()
        );

        void DestroyBuffer(CuBuffer handle);

        // Map a host visible buffer (pinned/managed) to HostPtr
        template <typename T>
        HostPtr<T> MapBuffer(CuBufferView<T> view) {
            if (view.handle.id == CU_NULL_HANDLE || CU_GET_INDEX(view.handle) >= m_registry.size()) {
                return HostPtr<T>(nullptr);
            }

            CuAssert(CU_GET_GENERATION(view.handle) == m_generations[CU_GET_INDEX(view.handle)],
                     "MapBuffer failed: Handle generation mismatch (Use-After-Free).");

            const auto& record = m_records[CU_GET_INDEX(view.handle)];
            CuAssert(record.usage == CuMemoryUsage::CPU_PINNED || record.usage == CuMemoryUsage::GPU_MANAGED,
                     "MapBuffer failed: Buffer was not created with CPU_PINNED or GPU_MANAGED usage.");

            T* host_ptr = GetRawPointerUnsafe<T>(view.handle, view.byteOffset);
            return HostPtr<T>(host_ptr);
        }

        template <typename T = void>
        T* GetRawPointerUnsafe(CuBuffer handle, size_t byteOffset = 0) const {
            if (handle.id == CU_NULL_HANDLE || CU_GET_INDEX(handle) >= m_registry.size()) return nullptr;

            // Generational Check
            if (CU_GET_GENERATION(handle) != m_generations[CU_GET_INDEX(handle)]) {
                return nullptr;
            }

            char* raw = static_cast<char*>(m_registry[CU_GET_INDEX(handle)]);
            if (!raw) return nullptr;

            return reinterpret_cast<T*>(raw + byteOffset);
        }

    private:

        std::vector<void*> m_registry;        // Pointer to raw memory (Host or Device)
        std::vector<uint32_t> m_generations;  // Current generation for each registry index (ABA prevention)
        std::vector<uint32_t> m_freeIndices;  // Recycled indices from destroyed resources to minimize registry growth

        /**
         * @brief Information about each allocation for debugging and leak detection.
         */
        struct AllocRecord {
            size_t size;
            std::string debugName;
            std::string sourceFile;
            uint32_t sourceLine;
            CuMemoryUsage usage = CuMemoryUsage::GPU_ONLY;
        };
        std::vector<AllocRecord> m_records;
    };
}
#endif //YATACU_CUDADEVICE_CUH