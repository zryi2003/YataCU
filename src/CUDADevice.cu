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
        // Index 0 is reserved.
        m_registry.push_back(nullptr);
        m_generations.push_back(0);
        m_records.push_back({});
    }

    CuDevice::~CuDevice() {
        bool hasLeak = false;
        for (size_t i = 1; i < m_registry.size(); ++i) {
            // We only care about entries that are not in the free pool and are not null
            if (m_registry[i] != nullptr) {
                hasLeak = true;
                std::cerr << "[YataCU Memory] Buffer not destroyed: '" << m_records[i].debugName
                          << "' (" << m_records[i].size << " bytes)\n"
                          << "  -> Allocated at: " << m_records[i].sourceFile << ":" << m_records[i].sourceLine << "\n";
            }
        }
    }

    CuBuffer CuDevice::CreateBuffer(const CuBufferCreateInfo& info, std::source_location loc) {
        CuAssert(info.size > 0, "Cannot allocate buffer with size 0", loc);

        // Fetch a recycled index if available, or expand the registry.
        uint32_t target_index = 0;
        if (!m_freeIndices.empty()) {
            target_index = m_freeIndices.back();
            m_freeIndices.pop_back();
        } else {
            target_index = static_cast<uint32_t>(m_registry.size());
            m_registry.push_back(nullptr);
            m_generations.push_back(1); // Start generation at 1
            m_records.push_back({});
        }

        // Allocate physical memory based on usage requirements.
        void* raw_ptr = nullptr;
        cudaError_t err = cudaSuccess;

        switch (info.usage) {
            case CuMemoryUsage::GPU_ONLY:
                err = cudaMalloc(&raw_ptr, info.size);
                break;
            case CuMemoryUsage::CPU_PINNED:
                err = cudaMallocHost(&raw_ptr, info.size);
                break;
            case CuMemoryUsage::GPU_MANAGED:
                err = cudaMallocManaged(&raw_ptr, info.size);
                break;
            default:
                CuAssert(false, "Unsupported memory usage type", loc);
        }

        CuCheck(err, std::string("Failed to allocate " + std::to_string(info.size) + " bytes for " + info.debugName).c_str(), loc);

        m_registry[target_index] = raw_ptr;
        m_records[target_index] = {
            info.size,
            info.debugName,
            loc.file_name(),
            loc.line(),
            info.usage
        };

        std::cout << "[YataCU Device] Allocated Buffer [Idx: " << target_index
                  << ", Gen: " << m_generations[target_index] << "] | "
                  << info.debugName << " (" << info.size << " bytes)\n";

        return CuBuffer{CU_PACK_HANDLE(m_generations[target_index], target_index)};
    }

    void CuDevice::DestroyBuffer(CuBuffer handle) {
        if (handle.id == CU_NULL_HANDLE || CU_GET_INDEX(handle) >= m_registry.size()) return;
        
        // Ensure the caller holds the most current version of this index.
        if (CU_GET_GENERATION(handle) != m_generations[CU_GET_INDEX(handle)]) {
            std::cerr << "[YataCU Warning] Attempted to destroy an invalid or expired handle [Idx: "
                      << CU_GET_INDEX(handle) << ", Gen: " << CU_GET_GENERATION(handle) << "]\n";
            return;
        }

        uint32_t index = CU_GET_INDEX(handle);
        void* ptr = m_registry[index];
        if (ptr) {
            CuMemoryUsage usage = m_records[index].usage;
            switch (usage) {
                case CuMemoryUsage::CPU_PINNED:
                    cudaFreeHost(ptr);
                    break;
                default:
                    cudaFree(ptr);
                    break;
            }

            m_registry[index] = nullptr;
            
            // Increment generation to invalidate all existing handles to this index.
            m_generations[index]++;
            if (m_generations[index] == 0) m_generations[index] = 1;

            m_freeIndices.push_back(index);

            std::cout << "[YataCU Device] Destroyed Buffer [Idx: " << index << "]\n";
        }
    }
} // namespace YataCU