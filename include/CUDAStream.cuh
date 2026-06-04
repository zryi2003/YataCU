//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef YATACU_CUDASTREAM_CUH
#define YATACU_CUDASTREAM_CUH

#include <CUDASpecs.cuh>
#include <CUDAMemory.cuh>
#include <CUDAError.cuh>
#include <CUDAGraph.cuh>
#include <CUDAEvent.cuh>
#include <type_traits>

namespace YATACU {
    class CuDevice;

    class CuCommandStream {
    public:
        // Bind stream to a specific device
        CuCommandStream(const CuDevice& device);
        ~CuCommandStream();

        // Stream is not copyable
        CuCommandStream(const CuCommandStream&) = delete;
        CuCommandStream& operator=(const CuCommandStream&) = delete;

        // --- [Graph Capture] ---
        void BeginCapture();
        void EndCapture(CuGraph& outGraph);
        void Launch(const CuGraph& graph);

        // --- [Event Synchronization] ---
        void Record(const CuEvent& event);
        void WaitEvent(const CuEvent& event);

        /**
         * @brief Simple 1D dispatch for convenience. Grid/Block size is auto-calculated with default block size of 256.
         * Constraints: Arguments must be trivially copyable and NOT pointers or references.
         */
        template <typename KernelFunc, typename... Args>
        requires ((std::is_trivially_copyable_v<Args> && 
                   !std::is_pointer_v<Args> && 
                   !std::is_reference_v<Args>) && ...)
        void Dispatch(uint32_t threadCount, KernelFunc kernel, Args... args) {
            if (threadCount == 0) return;

            uint32_t blockSize = 256;
            uint32_t gridSize = (threadCount + blockSize - 1) / blockSize;

            DispatchInternal(gridSize, blockSize, 0, kernel, args...);
        }

        /**
         * @brief Explicit dispatch with full control over grid/block dimensions and shared memory.
         * Constraints: Arguments must be trivially copyable and NOT pointers or references.
         */
        template <typename KernelFunc, typename... Args>
        requires ((std::is_trivially_copyable_v<Args> && 
                   !std::is_pointer_v<Args> && 
                   !std::is_reference_v<Args>) && ...)
        void Dispatch(const CuDispatchInfo& info, KernelFunc kernel, Args... args) {
            DispatchInternal(info.gridDim, info.blockDim, info.sharedMemBytes, kernel, args...);
        }

        template <typename T>
        void CopyHostToDevice(CuBufferView<T> dst, const T* srcHost, size_t elementCount) {
            T* dst_raw = m_device.GetRawPointerUnsafe<T>(dst.handle, dst.byteOffset);
            CuCheck(cudaMemcpyAsync(dst_raw, srcHost, elementCount * sizeof(T), cudaMemcpyHostToDevice, m_stream),
                    "H2D Copy Failed");
        }

        template <typename T>
        void CopyDeviceToHost(T* dstHost, CuBufferView<T> src, size_t elementCount) {
            T* src_raw = m_device.GetRawPointerUnsafe<T>(src.handle, src.byteOffset);
            CuCheck(cudaMemcpyAsync(dstHost, src_raw, elementCount * sizeof(T), cudaMemcpyDeviceToHost, m_stream),
                    "D2H Copy Failed");
        }

        template <typename T>
        void CopyDeviceToDevice(CuBufferView<T> dst, CuBufferView<T> src, size_t elementCount) {
            T* dst_raw = m_device.GetRawPointerUnsafe<T>(dst.handle, dst.byteOffset);
            T* src_raw = m_device.GetRawPointerUnsafe<T>(src.handle, src.byteOffset);

            CuCheck(cudaMemcpyAsync(dst_raw, src_raw, elementCount * sizeof(T), cudaMemcpyDeviceToDevice, m_stream),
                    "D2D Copy Failed");
        }

        void WaitIdle(); // Wait for this stream idle

    private:
        const CuDevice& m_device;
        cudaStream_t m_stream;

        template <typename KernelFunc, typename... Args>
        void DispatchInternal(dim3 grid, dim3 block, size_t shmem, KernelFunc kernel, Args... args) {
            kernel<<<grid, block, shmem, m_stream>>>(ResolveArg(m_device, args)...);
            CuCheck(cudaGetLastError(), "Kernel Dispatch Failed");
        }

        template <typename T>
        __host__ inline T ResolveArg(const CuDevice&, T val) { 
            return val;
        }

        template <typename T>
        __host__ inline DevicePtr<T> ResolveArg(const CuDevice& dev, CuBufferView<T> view) {
            T* raw_ptr = dev.GetRawPointerUnsafe<T>(view.handle, view.byteOffset);
            return DevicePtr<T>(raw_ptr);
        }
    };
}

#endif //YATACU_CUDASTREAM_CUH
