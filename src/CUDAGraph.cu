//
// Created by YataCU Developer.
//

#include "../include/CUDAGraph.cuh"

namespace YATACU {

    CuGraph::~CuGraph() {
        if (m_exec) {
            cudaGraphExecDestroy(m_exec);
            m_exec = nullptr;
        }
        if (m_graph) {
            cudaGraphDestroy(m_graph);
            m_graph = nullptr;
        }
    }

    CuGraph::CuGraph(CuGraph&& other) noexcept 
        : m_graph(std::exchange(other.m_graph, nullptr)),
          m_exec(std::exchange(other.m_exec, nullptr)) {}

    CuGraph& CuGraph::operator=(CuGraph&& other) noexcept {
        if (this != &other) {
            if (m_exec) cudaGraphExecDestroy(m_exec);
            if (m_graph) cudaGraphDestroy(m_graph);
            m_graph = std::exchange(other.m_graph, nullptr);
            m_exec = std::exchange(other.m_exec, nullptr);
        }
        return *this;
    }

    void CuGraph::InitializeFromCapture(cudaGraph_t graph) {
        if (m_exec) cudaGraphExecDestroy(m_exec);
        if (m_graph) cudaGraphDestroy(m_graph);

        m_graph = graph;
        if (m_graph) {
            // Using CUDA 12 compatible signature (flags = 0)
            CuCheck(cudaGraphInstantiate(&m_exec, m_graph, 0), "Failed to instantiate CUDA Graph");
        }
    }

} // namespace YATACU
