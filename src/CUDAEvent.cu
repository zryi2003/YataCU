//
// Created by YataCU Developer.
//

#include "../include/CUDAEvent.cuh"
#include <utility>

namespace YATACU {

    CuEvent::CuEvent(bool enableTiming) {
        unsigned int flags = enableTiming ? cudaEventDefault : cudaEventDisableTiming;
        CuCheck(cudaEventCreateWithFlags(&m_event, flags), "Failed to create CUDA Event");
    }

    CuEvent::~CuEvent() {
        if (m_event) {
            cudaEventDestroy(m_event);
            m_event = nullptr;
        }
    }

    CuEvent::CuEvent(CuEvent&& other) noexcept : m_event(std::exchange(other.m_event, nullptr)) {}

    CuEvent& CuEvent::operator=(CuEvent&& other) noexcept {
        if (this != &other) {
            if (m_event) cudaEventDestroy(m_event);
            m_event = std::exchange(other.m_event, nullptr);
        }
        return *this;
    }

    void CuEvent::Wait() const {
        CuCheck(cudaEventSynchronize(m_event), "Event Wait failed");
    }

    float CuEvent::ElapsedTime(const CuEvent& start, const CuEvent& end) {
        float ms = 0.0f;
        CuCheck(cudaEventElapsedTime(&ms, start.m_event, end.m_event), "Failed to calculate elapsed time");
        return ms;
    }

} // namespace YATACU
