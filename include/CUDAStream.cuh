//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef WAVEFRONTRENDERER_CUDASTREAM_CUH
#define WAVEFRONTRENDERER_CUDASTREAM_CUH

#include <CUDADevice.cuh>
#include <CUDAMemory.cuh>
#include <CUDAError.cuh>

namespace YATACU {
    class CuCommandStream {
    public:
        // Bind stream to a specific device
        CuCommandStream(const CuDevice& device);
        ~CuCommandStream();

        // Stream is not copyable
        CuCommandStream(const CuCommandStream&) = delete;
        CuCommandStream& operator=(const CuCommandStream&) = delete;

        template <typename KernelFunc, typename... Args>
        void Dispatch(uint32_t threadCount, KernelFunc kernel, Args... args) {
            if (threadCount == 0) return;

            // TODO: CuDispatchInfo
            int blockSize = 256;
            int gridSize = (threadCount + blockSize - 1) / blockSize;

            // Parameter Pack Expansion
            kernel<<<gridSize, blockSize, 0, m_stream>>>(ResolveArg(m_device, args)...);

            // Catch launch error
            CuCheck(cudaGetLastError(), "Kernel Dispatch Failed");
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


        void WaitIdle(); // Wait for this stream idle

    private:
        const CuDevice& m_device;
        cudaStream_t m_stream;

        template <typename T>
        __host__ inline T ResolveArg(const CuDevice&, T val) { // NOLINT
            if constexpr (std::is_pointer_v<T>) {
                static_assert(!std::is_pointer_v<T>,
                              "\n\n[YataCU FATAL ERROR] Raw C++ pointers cannot be passed directly to a kernel. "
                              "\n    - For device memory, wrap it in a CuBuffer/CuBufferView."
                              "\n    - For host data, use stream.CopyHostToDevice() to upload it first.\n");
            }
            return val;
        }

        template <typename T>
        __host__ inline DevicePtr<T> ResolveArg(const CuDevice& dev, CuBufferView<T> view) { // NOLINT
            T* raw_ptr = dev.GetRawPointerUnsafe<T>(view.handle, view.byteOffset);
            return DevicePtr<T>(raw_ptr);
        }
    };
}

#endif //WAVEFRONTRENDERER_CUDASTREAM_CUH