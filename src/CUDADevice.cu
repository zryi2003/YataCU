//
// Created by Zhuoran Yi on 2026/2/22.
//

#include <CUDADevice.cuh>
#include <iostream>
#include <source_location>

#include "CUDADefines.cuh"
#include "CUDAError.cuh"

namespace YATACU {
    CuDevice::CuDevice() {
        // Handle ID 0 always map to CU_NULL_HANDLE
        m_registry.push_back(nullptr);
        m_records.push_back({});
    }

    CuDevice::~CuDevice() {
        bool hasLeak = false;
        for (size_t i = 1; i < m_registry.size(); ++i) {
            if (m_registry[i] != nullptr) {
                hasLeak = true;
                std::cerr << "[YataCU Memory] Buffer not destroyed: '" << m_records[i].debugName
                          << "' (" << m_records[i].size << " bytes)\n"
                          << "  -> Allocated at: " << m_records[i].sourceFile << ":" << m_records[i].sourceLine << "\n";
            }
        }
        if (hasLeak) {
            // std::abort(); If needed, we can trigger std::abort()
        }
    }

    CuBuffer CuDevice::CreateBuffer(const CuBufferCreateInfo& info, std::source_location loc) {
        CuAssert(info.size > 0, "Cannot allocate buffer with size 0", loc);

        void* raw_ptr = nullptr;
        cudaError_t err = cudaSuccess;

        // TODO: More precise memory allocation
        switch (info.usage) {
            case CuMemoryUsage::GPU_ONLY:
                err = cudaMalloc(&raw_ptr, info.size);
                break;
            case CuMemoryUsage::CPU_PINNED:
                err = cudaMallocHost(&raw_ptr, info.size); // Pinned Memory
                break;
            case CuMemoryUsage::GPU_MANAGED:
                err = cudaMallocManaged(&raw_ptr, info.size); // Unified Memory
                break;
            default:
                CuAssert(false, "Unsupported memory usage type", loc);
        }

        CuCheck(err, std::string("Failed to allocate " + std::to_string(info.size) + " bytes for " + info.debugName).c_str(), loc);

        // Register to the Handle table
        uint32_t new_id = static_cast<uint32_t>(m_registry.size());
        m_registry.push_back(raw_ptr);

        m_records.push_back({
            info.size,
            info.debugName,
            loc.file_name(),
            loc.line(),
            info.usage
        });

        std::cout << "[YataCU Device] Allocated Handle ID " << new_id
                  << " | " << info.debugName << " (" << info.size << " bytes)\n";

        return CuBuffer{new_id};
    }

    void CuDevice::DestroyBuffer(CuBuffer handle) {
        if (handle.id == CU_NULL_HANDLE || handle.id >= m_registry.size()) return;

        void* ptr = m_registry[handle.id];
        if (ptr) {
            CuMemoryUsage usage = m_records[handle.id].usage;
            switch (usage) {
                case CuMemoryUsage::CPU_PINNED:
                    cudaFreeHost(ptr);
                    break;
                default:
                    cudaFree(ptr);
                    break;
            }
            m_registry[handle.id] = nullptr; // Prevent Use-After-Free
            std::cout << "[YataCU Device] Destroyed Handle ID " << handle.id << "\n";
        }
    }
} // namespace YataCU