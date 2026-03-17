//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef WAVEFRONTRENDERER_CUDADEVICE_CUH
#define WAVEFRONTRENDERER_CUDADEVICE_CUH
#include <CUDADefines.cuh>
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
            if (view.handle.id == CU_NULL_HANDLE || view.handle.id >= m_registry.size()) {
                return HostPtr<T>(nullptr);
            }

            const auto& record = m_records[view.handle.id];
            CuAssert(record.usage == CuMemoryUsage::CPU_PINNED || record.usage == CuMemoryUsage::GPU_MANAGED,
                     "MapBuffer failed: Buffer was not created with CPU_PINNED or GPU_MANAGED usage.");

            T* host_ptr = GetRawPointerUnsafe<T>(view.handle, view.byteOffset);
            return HostPtr<T>(host_ptr);
        }

        template <typename T = void>
        T* GetRawPointerUnsafe(CuBuffer handle, size_t byteOffset = 0) const {
            if (handle.id >= m_registry.size()) return nullptr;

            char* raw = static_cast<char*>(m_registry[handle.id]);
            if (!raw) return nullptr;

            return reinterpret_cast<T*>(raw + byteOffset);
        }

    private:

        std::vector<void*> m_registry; // Handle registry

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
#endif //WAVEFRONTRENDERER_CUDADEVICE_CUH