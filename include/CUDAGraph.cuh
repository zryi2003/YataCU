//
// Created by YataCU Developer.
//

#ifndef YATACU_CUDAGRAPH_CUH
#define YATACU_CUDAGRAPH_CUH

#include <cuda_runtime.h>
#include "CUDAError.cuh"
#include <utility>

namespace YATACU {

    class CuCommandStream;

    class CuGraph {
    public:
        CuGraph() = default;
        ~CuGraph();

        // Non-copyable
        CuGraph(const CuGraph&) = delete;
        CuGraph& operator=(const CuGraph&) = delete;

        // Movable
        CuGraph(CuGraph&& other) noexcept;
        CuGraph& operator=(CuGraph&& other) noexcept;

        bool IsInstantiated() const { return m_exec != nullptr; }

    private:
        friend class CuCommandStream;

        void InitializeFromCapture(cudaGraph_t graph);

        cudaGraph_t m_graph = nullptr;
        cudaGraphExec_t m_exec = nullptr;
    };

} // namespace YATACU

#endif // YATACU_CUDAGRAPH_CUH
