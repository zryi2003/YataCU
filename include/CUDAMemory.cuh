//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef YATACU_CUDAMEMORY_CUH
#define YATACU_CUDAMEMORY_CUH

#include <CUDADefines.cuh>

namespace YATACU {
    class CuDevice;
    class CuCommandStream;

    template <typename T> class DevicePtr;
    template <typename T> struct CuBufferView;

    template <typename T>
    class DevicePtr {
    private:
        T* m_ptr;

        __host__ __device__ explicit DevicePtr(T* ptr) : m_ptr(ptr) {}

        friend class CuCommandStream;

    public:

        __host__ __device__ DevicePtr() : m_ptr(nullptr) {}

        __device__ inline T& operator[](size_t index) { return m_ptr[index]; }
        __device__ inline const T& operator[](size_t index) const { return m_ptr[index]; }

        __device__ inline T& operator*() { return *m_ptr; }
        __device__ inline const T& operator*() const { return *m_ptr; }

        __device__ inline T* operator->() { return m_ptr; }
        __device__ inline const T* operator->() const { return m_ptr; }

        __device__ inline DevicePtr<T> operator+(size_t index) const { return DevicePtr<T>(m_ptr + index); }
    };


    template <typename T>
    class HostPtr {
    private:
        T* m_ptr;

        explicit HostPtr(T* ptr) : m_ptr(ptr) {}

        friend class CuDevice;

    public:
        HostPtr() : m_ptr(nullptr) {}

        __host__ inline T& operator[](size_t index) { return m_ptr[index]; }
        __host__ inline const T& operator[](size_t index) const { return m_ptr[index]; }

        __host__ inline T& operator*() { return *m_ptr; }
        __host__ inline const T& operator*() const { return *m_ptr; }

        __host__ inline T* get() { return m_ptr; }
        __host__ inline const T* get() const { return m_ptr; }

        explicit operator bool() const { return m_ptr != nullptr; }
    };

    template <typename T>
    struct CuBufferView {
        CuBuffer handle = { CU_NULL_HANDLE };
        size_t byteOffset = 0;

        CuBufferView() = default;
        CuBufferView(CuBuffer h, size_t offset = 0) : handle(h), byteOffset(offset) {}
    };
}

#endif //YATACU_CUDAMEMORY_CUH