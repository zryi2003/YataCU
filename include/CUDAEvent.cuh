//
// Created by YataCU Developer.
//

#ifndef YATACU_CUDAEVENT_CUH
#define YATACU_CUDAEVENT_CUH

#include <cuda_runtime.h>
#include "CUDAError.cuh"

namespace YATACU {

    class CuCommandStream;

    class CuEvent {
    public:
        // Default constructor creates a timing-enabled event
        explicit CuEvent(bool enableTiming = true);
        ~CuEvent();

        // Non-copyable
        CuEvent(const CuEvent&) = delete;
        CuEvent& operator=(const CuEvent&) = delete;

        // Movable
        CuEvent(CuEvent&& other) noexcept;
        CuEvent& operator=(CuEvent&& other) noexcept;

        cudaEvent_t Get() const { return m_event; }

        // Block CPU until this event is reached in the recorded stream
        void Wait() const;

        // Calculate elapsed time between two events in milliseconds
        static float ElapsedTime(const CuEvent& start, const CuEvent& end);

    private:
        cudaEvent_t m_event = nullptr;
    };

} // namespace YATACU

#endif // YATACU_CUDAEVENT_CUH
