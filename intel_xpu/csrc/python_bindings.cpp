/*
 * Python bindings for Intel XPU DeepEP
 *
 * This file provides Python bindings matching the CUDA version's API style.
 * The Buffer class now uses torch::Tensor directly instead of void* pointers.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>
#include "deep_ep_xpu.hpp"

namespace py = pybind11;

#ifndef MODULE_NAME
#define MODULE_NAME _C
#endif

PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = "DeepEP XPU: an efficient expert-parallel communication library for Intel XPU";

    // EventHandle class - matching CUDA version
    py::class_<deep_ep_xpu::EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &deep_ep_xpu::EventHandle::current_stream_wait);

    // Buffer class - directly exposing C++ Buffer with tensor-based API
    py::class_<deep_ep_xpu::Buffer>(m, "Buffer")
        .def(py::init<int, int, size_t, size_t, bool, int, bool>(),
             py::arg("rank"),
             py::arg("num_ranks"),
             py::arg("buffer_size"),
             py::arg("rdma_buffer_size"),
             py::arg("low_latency_mode") = false,
             py::arg("num_qps_per_rank") = 1,
             py::arg("explicitly_destroy") = false)
        // Static method - get low latency RDMA size hint
        .def_static("get_low_latency_rdma_size_hint", &deep_ep_xpu::Buffer::get_low_latency_rdma_size_hint,
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("hidden"),
             py::arg("num_ranks"),
             py::arg("num_experts"))
        .def("low_latency_dispatch", &deep_ep_xpu::Buffer::low_latency_dispatch,
             py::arg("x"),
             py::arg("topk_idx"),
             py::arg("cumulative_local_expert_recv_stats") = py::none(),
             py::arg("dispatch_wait_recv_cost_stats") = py::none(),
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("num_experts"),
             py::arg("use_fp8") = true,
             py::arg("round_scale") = false,
             py::arg("use_ue8m0") = false,
             py::arg("async_finish") = false,
             py::arg("return_recv_hook") = false)
        .def("low_latency_combine", &deep_ep_xpu::Buffer::low_latency_combine,
             py::arg("x"),
             py::arg("topk_idx"),
             py::arg("topk_weights"),
             py::arg("src_info"),
             py::arg("layout_range"),
             py::arg("combine_wait_recv_cost_stats") = py::none(),
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("num_experts"),
             py::arg("use_logfmt") = false,
             py::arg("zero_copy") = false,
             py::arg("async_finish") = false,
             py::arg("return_recv_hook") = false,
             py::arg("out") = py::none())
        .def("clean_low_latency_buffer", &deep_ep_xpu::Buffer::clean_low_latency_buffer,
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("hidden"),
             py::arg("num_experts"))
        .def("low_latency_query_mask_buffer", &deep_ep_xpu::Buffer::low_latency_query_mask_buffer,
             py::arg("mask_status"))
        .def("low_latency_update_mask_buffer", &deep_ep_xpu::Buffer::low_latency_update_mask_buffer,
             py::arg("rank_to_mask"),
             py::arg("mask"))
        .def("low_latency_clean_mask_buffer", &deep_ep_xpu::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep_xpu::Buffer::get_next_low_latency_combine_buffer,
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("hidden"),
             py::arg("num_experts"))
        .def("synchronize", &deep_ep_xpu::Buffer::synchronize)
        .def("destroy", &deep_ep_xpu::Buffer::destroy);
}

