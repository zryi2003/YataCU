# YataCU

**YataCU** is a lightweight, modern C++ CUDA wrapper designed for rapid prototyping and research. It abstracts the verbosity of the CUDA API while maintaining explicit control over GPU operations and memory management.

## Key Features

### RAII Resource Management
All core CUDA resources, such as device contexts and streams, are managed via RAII wrappers (`CuDevice`, `CuCommandStream`), ensuring proper initialization and cleanup. Device memory allocations (`CuBuffer`) are also handled with RAII.

### Safe and Explicit Memory Management
- **Structured Buffers**: `CuBuffer` and `CuBufferView` provide a type-safe and structured approach to managing device memory allocations.
- **Host-Device Interaction**: Supports various memory usages (`GPU_ONLY`, `CPU_PINNED`, `GPU_MANAGED`) and provides explicit functions for mapping host-visible buffers (`MapBuffer`) and asynchronous data transfers (`CopyHostToDevice`, `CopyDeviceToHost`).
- **Kernel Argument Safety**: A `ResolveArg` mechanism prevents accidental passing of raw host pointers directly to CUDA kernels, enforcing the use of `DevicePtr` for device memory access within kernels, enhancing type safety and preventing common CUDA programming errors.

### Asynchronous Operations
- **Stream-based Execution**: `CuCommandStream` encapsulates `cudaStream_t` to enable asynchronous kernel dispatches and memory transfer operations, allowing for concurrent execution.
- **Synchronization**: Provides `WaitIdle` for stream synchronization, ensuring operations complete before proceeding.

### Modern C++ Design
Leverages modern C++ features to provide a cleaner, more intuitive, and type-safe API for CUDA programming, reducing boilerplate and improving code readability.

## Components
- **CuDevice**: Manages the CUDA device context and acts as a factory for `CuBuffer` objects, handling device memory allocations and deallocations.
- **CuCommandStream**: Encapsulates a `cudaStream_t` for asynchronous kernel launches and memory copy operations.
- **CuBuffer / CuBufferView**: Abstractions for CUDA device memory allocations, providing handle-based access and type-safe views.

## Usage Example

```cpp
// Include individual headers like CUDADevice.cuh, CUDAStream.cuh

// Initialize Device
YATACU::CuDevice device;

// Create a buffer on the device
YATACU::CuBufferCreateInfo bufferInfo;
bufferInfo.size = 1024 * sizeof(float); // Example: 1KB float buffer
bufferInfo.usage = YATACU::CuMemoryUsage::GPU_ONLY;
bufferInfo.debugName = "MyFloatBuffer";
YATACU::CuBuffer myBuffer = device.CreateBuffer(bufferInfo);

// Create a command stream
YATACU::CuCommandStream stream(device);

// Define a simple CUDA kernel
__global__ void addKernel(YATACU::DevicePtr<float> data, float value, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] += value;
    }
}

// Assuming 'myBuffer' has been populated (e.g., via stream.CopyHostToDevice)
// For example, let's say we want to process 256 elements
int elementCount = 256;

// Dispatch the kernel
stream.Dispatch(elementCount, addKernel, YATACU::CuBufferView<float>(myBuffer), 5.0f, elementCount);

// Wait for the stream to complete all operations
stream.WaitIdle();

// At this point, 'myBuffer' on the device contains the results.
// You can then copy it back to host memory if needed using stream.CopyDeviceToHost.
```