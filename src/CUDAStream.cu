//
// Created by Zhuoran Yi on 2026/2/22.
//

#include "../include/CUDAStream.cuh"
#include "../include/CUDAError.cuh"
#include <iostream>

namespace YATACU {

    CuCommandStream::CuCommandStream(const CuDevice& device) : m_device(device), m_stream(nullptr) {
        // Create a non-blocking asynchronous stream (cudaStreamNonBlocking can avoid implicit synchronization conflicts with the default Stream 0).
        CuCheck(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking), "Failed to create CUDA stream");
        std::cout << "[YataCU Stream] Command stream created.\n";
    }

    CuCommandStream::~CuCommandStream() {
        if (m_stream) {
            // Destroy after all tasks in the stream finished
            cudaStreamSynchronize(m_stream);
            cudaStreamDestroy(m_stream);
            std::cout << "[YataCU Stream] Command stream destroyed.\n";
        }
    }

    void CuCommandStream::WaitIdle() {
        CuCheck(cudaStreamSynchronize(m_stream), "Stream WaitIdle failed");
    }

}