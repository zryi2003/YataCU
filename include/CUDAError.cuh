//
// Created by Zhuoran Yi on 2026/2/22.
//

#ifndef WAVEFRONTRENDERER_CUDAERROR_CUH
#define WAVEFRONTRENDERER_CUDAERROR_CUH
#include <cuda_runtime.h>
#include <iostream>
#include <source_location>

namespace YATACU {
    inline void CuCheck(
    cudaError_t err,
    const char* customMsg = "",
    std::source_location loc = std::source_location::current()
) {
        if (err != cudaSuccess) {
            std::cerr << "\n==============================================================\n"
                      << "[YataCU FATAL ERROR] " << cudaGetErrorString(err) << "\n"
                      << "--------------------------------------------------------------\n"
                      << "Context  : " << customMsg << "\n"
                      << "Location : " << loc.file_name() << ":" << loc.line() << "\n"
                      << "Function : " << loc.function_name() << "\n"
                      << "==============================================================\n";

            std::abort();
        }
    }

    inline void CuAssert(
        bool condition,
        const char* msg,
        std::source_location loc = std::source_location::current()
    ) {
        if (!condition) {
            std::cerr << "\n[YataCU ASSERTION FAILED]\n"
                      << "Msg: " << msg << "\n"
                      << "Loc: " << loc.file_name() << ":" << loc.line() << "\n";
            std::abort();
        }
    }
}

#endif //WAVEFRONTRENDERER_CUDAERROR_CUH