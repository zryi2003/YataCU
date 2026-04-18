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

    void CuCommandStream::BeginCapture() {
        CuCheck(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal), "Failed to begin stream capture");
    }

    void CuCommandStream::EndCapture(CuGraph& outGraph) {
        cudaGraph_t capturedGraph = nullptr;
        CuCheck(cudaStreamEndCapture(m_stream, &capturedGraph), "Failed to end stream capture");
        outGraph.InitializeFromCapture(capturedGraph);
    }

    void CuCommandStream::Launch(const CuGraph& graph) {
        if (graph.IsInstantiated()) {
            CuCheck(cudaGraphLaunch(graph.m_exec, m_stream), "Failed to launch CUDA Graph");
        }
    }

    void CuCommandStream::Record(const CuEvent& event) {
        CuCheck(cudaEventRecord(event.Get(), m_stream), "Failed to record event");
    }

    void CuCommandStream::WaitEvent(const CuEvent& event) {
        CuCheck(cudaStreamWaitEvent(m_stream, event.Get(), 0), "Stream WaitEvent failed");
    }

}