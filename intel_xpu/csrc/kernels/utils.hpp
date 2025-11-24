#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include "config.hpp"

namespace deep_ep_xpu {

// BF16 conversion utilities
SYCL_EXTERNAL inline sycl::ext::oneapi::bfloat16 float_to_bf16(float x) {
    return sycl::ext::oneapi::bfloat16(x);
}

SYCL_EXTERNAL inline float bf16_to_float(sycl::ext::oneapi::bfloat16 x) {
    return float(x);
}

// Overload for uint16_t (raw BF16 bits)
SYCL_EXTERNAL inline float bf16_to_float(uint16_t x) {
    sycl::ext::oneapi::bfloat16 bf16;
    *reinterpret_cast<uint16_t*>(&bf16) = x;
    return float(bf16);
}

SYCL_EXTERNAL inline uint16_t float_to_bf16_bits(float x) {
    sycl::ext::oneapi::bfloat16 bf16(x);
    return *reinterpret_cast<uint16_t*>(&bf16);
}

// Atomic operations with memory ordering
// Port of: ld_acquire_global, st_release_global, atomic_add_release_global from CUDA

template<typename T>
SYCL_EXTERNAL inline T atomic_load_acquire(T* ptr) {
    sycl::atomic_ref<T, sycl::memory_order::acquire,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> atomic_ptr(*ptr);
    return atomic_ptr.load();
}

template<typename T>
SYCL_EXTERNAL inline void atomic_store_release(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::release,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> atomic_ptr(*ptr);
    atomic_ptr.store(value);
}

template<typename T>
SYCL_EXTERNAL inline T atomic_add_release(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> atomic_ptr(*ptr);
    return atomic_ptr.fetch_add(value);
}

template<typename T>
SYCL_EXTERNAL inline T atomic_exchange(T* ptr, T value) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> atomic_ptr(*ptr);
    return atomic_ptr.exchange(value);
}

// FP8 E4M3 conversion (simplified)
struct fp8_e4m3 {
    uint8_t data;
    
    fp8_e4m3() : data(0) {}
    explicit fp8_e4m3(uint8_t d) : data(d) {}
    
    // Convert from float
    static fp8_e4m3 from_float(float x, float scale = 1.0f) {
        x *= scale;
        // Clamp to FP8 E4M3 range: [-448, 448]
        x = sycl::clamp(x, -448.0f, 448.0f);
        
        // Simple quantization (production code would use proper FP8 conversion)
        int8_t quantized = static_cast<int8_t>(sycl::round(x));
        return fp8_e4m3(static_cast<uint8_t>(quantized));
    }
    
    // Convert to float
    float to_float(float scale = 1.0f) const {
        // Simple dequantization
        return static_cast<float>(static_cast<int8_t>(data)) / scale;
    }
};

// Per-token FP8 quantization
template<typename T>
inline void quantize_per_token_fp8(const T* input, fp8_e4m3* output, float* scales,
                                   int num_tokens, int hidden, sycl::nd_item<1> item) {
    int token_idx = item.get_global_id(0);
    if (token_idx >= num_tokens) return;
    
    const T* token_input = input + token_idx * hidden;
    fp8_e4m3* token_output = output + token_idx * hidden;
    
    // Compute scale (max absolute value)
    float max_val = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float val = sycl::fabs(bf16_to_float(token_input[i]));
        max_val = sycl::max(max_val, val);
    }
    
    float scale = max_val > 0.0f ? (448.0f / max_val) : 1.0f;
    scales[token_idx] = scale;
    
    // Quantize
    for (int i = 0; i < hidden; i++) {
        float val = bf16_to_float(token_input[i]);
        token_output[i] = fp8_e4m3::from_float(val, scale);
    }
}

// Per-block FP8 quantization (for scales)
template<typename T>
inline void quantize_per_block_fp8(const T* input, fp8_e4m3* output, float* scales,
                                   int num_elements, int block_size, sycl::nd_item<1> item) {
    int block_idx = item.get_global_id(0);
    int num_blocks = (num_elements + block_size - 1) / block_size;
    if (block_idx >= num_blocks) return;
    
    int start = block_idx * block_size;
    int end = sycl::min(start + block_size, num_elements);
    
    // Compute scale
    float max_val = 0.0f;
    for (int i = start; i < end; i++) {
        float val = sycl::fabs(bf16_to_float(input[i]));
        max_val = sycl::max(max_val, val);
    }
    
    float scale = max_val > 0.0f ? (448.0f / max_val) : 1.0f;
    scales[block_idx] = scale;
    
    // Quantize
    for (int i = start; i < end; i++) {
        float val = bf16_to_float(input[i]);
        output[i] = fp8_e4m3::from_float(val, scale);
    }
}

// Dequantize FP8 to BF16
inline void dequantize_fp8(const fp8_e4m3* input, sycl::ext::oneapi::bfloat16* output,
                          const float* scales, int num_elements, int scale_block_size,
                          sycl::nd_item<1> item) {
    int idx = item.get_global_id(0);
    if (idx >= num_elements) return;
    
    int scale_idx = idx / scale_block_size;
    float scale = scales[scale_idx];
    float val = input[idx].to_float(scale);
    output[idx] = float_to_bf16(val);
}

// Weighted reduction (for combine operation)
template<typename T>
inline T weighted_reduce(const T* values, const float* weights, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += bf16_to_float(values[i]) * weights[i];
    }
    return float_to_bf16(sum);
}

// Sub-group utilities
template<typename T>
inline T sub_group_reduce_add(T val, sycl::sub_group sg) {
    return sycl::reduce_over_group(sg, val, sycl::plus<T>());
}

template<typename T>
inline T sub_group_reduce_max(T val, sycl::sub_group sg) {
    return sycl::reduce_over_group(sg, val, sycl::maximum<T>());
}

}  // namespace deep_ep_xpu

