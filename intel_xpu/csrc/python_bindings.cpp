#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>
#include "deep_ep_xpu.hpp"

namespace py = pybind11;

namespace deep_ep_xpu {

// Wrapper class for Python bindings
class BufferWrapper {
public:
    BufferWrapper(py::object process_group,
                 size_t buffer_size,
                 size_t rdma_buffer_size,
                 bool low_latency_mode = false,
                 int num_qps_per_rank = 1,
                 bool explicitly_destroy = false) {
        // Convert Python process group to void*
        void* pg_ptr = process_group.ptr();
        buffer_ = std::make_unique<Buffer>(pg_ptr, buffer_size, rdma_buffer_size,
                                          low_latency_mode, num_qps_per_rank, explicitly_destroy);
    }
    
    py::tuple low_latency_dispatch(
        torch::Tensor x,
        torch::Tensor topk_idx,
        std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
        std::optional<torch::Tensor> dispatch_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_fp8 = true,
        bool round_scale = false,
        bool use_ue8m0 = false,
        bool async_finish = false,
        bool return_recv_hook = false) {
        
        std::optional<void*> stats1 = cumulative_local_expert_recv_stats.has_value() ?
            std::optional<void*>(cumulative_local_expert_recv_stats->data_ptr()) : std::nullopt;
        std::optional<void*> stats2 = dispatch_wait_recv_cost_stats.has_value() ?
            std::optional<void*>(dispatch_wait_recv_cost_stats->data_ptr()) : std::nullopt;
        
        auto [recv_x, recv_x_scales, recv_count, src_info, layout_range, event, hook] =
            buffer_->low_latency_dispatch(
                x.data_ptr(), topk_idx.data_ptr(), stats1, stats2,
                num_max_dispatch_tokens_per_rank, num_experts,
                use_fp8, round_scale, use_ue8m0, async_finish, return_recv_hook);
        
        // Convert pointers back to tensors
        int hidden = x.size(1);
        int num_local_experts = num_experts / 1;  // Would get from distributed
        
        auto recv_x_tensor = torch::from_blob(recv_x, 
            {num_local_experts * num_max_dispatch_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kXPU));
        
        auto recv_count_tensor = torch::from_blob(recv_count,
            {num_local_experts},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kXPU));
        
        auto src_info_tensor = torch::from_blob(src_info,
            {num_experts},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kXPU));
        
        auto layout_range_tensor = torch::from_blob(layout_range,
            {num_experts},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kXPU));
        
        py::object event_obj = py::none();
        py::object hook_obj = py::none();
        
        if (async_finish && event.has_value()) {
            // Wrap event (simplified)
            event_obj = py::cast(event.value());
        }
        
        if (return_recv_hook && hook.has_value()) {
            hook_obj = py::cpp_function(hook.value());
        }
        
        if (recv_x_scales.has_value()) {
            int num_scales = hidden / 128;
            auto scales_tensor = torch::from_blob(recv_x_scales.value(),
                {num_local_experts * num_max_dispatch_tokens_per_rank, num_scales},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kXPU));
            return py::make_tuple(recv_x_tensor, scales_tensor, recv_count_tensor,
                                 src_info_tensor, layout_range_tensor, event_obj, hook_obj);
        }
        
        return py::make_tuple(recv_x_tensor, py::none(), recv_count_tensor,
                             src_info_tensor, layout_range_tensor, event_obj, hook_obj);
    }
    
    py::tuple low_latency_combine(
        torch::Tensor x,
        torch::Tensor topk_idx,
        torch::Tensor topk_weights,
        torch::Tensor src_info,
        torch::Tensor layout_range,
        std::optional<torch::Tensor> combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_logfmt = false,
        bool zero_copy = false,
        bool async_finish = false,
        bool return_recv_hook = false,
        std::optional<torch::Tensor> out = std::nullopt) {
        
        std::optional<void*> stats = combine_wait_recv_cost_stats.has_value() ?
            std::optional<void*>(combine_wait_recv_cost_stats->data_ptr()) : std::nullopt;
        std::optional<void*> out_ptr = out.has_value() ?
            std::optional<void*>(out->data_ptr()) : std::nullopt;
        
        auto [combined_x, event, hook] = buffer_->low_latency_combine(
            x.data_ptr(), topk_idx.data_ptr(), topk_weights.data_ptr(),
            src_info.data_ptr(), layout_range.data_ptr(), stats,
            num_max_dispatch_tokens_per_rank, num_experts,
            use_logfmt, zero_copy, async_finish, return_recv_hook, out_ptr);
        
        // Convert pointer back to tensor
        int num_tokens = topk_idx.size(0);
        int hidden = x.size(1);
        
        auto combined_tensor = torch::from_blob(combined_x,
            {num_tokens, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kXPU));
        
        py::object event_obj = py::none();
        py::object hook_obj = py::none();
        
        if (async_finish && event.has_value()) {
            event_obj = py::cast(event.value());
        }
        
        if (return_recv_hook && hook.has_value()) {
            hook_obj = py::cpp_function(hook.value());
        }
        
        return py::make_tuple(combined_tensor, event_obj, hook_obj);
    }
    
    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        buffer_->clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts);
    }
    
    void synchronize() {
        buffer_->synchronize();
    }
    
    void destroy() {
        buffer_->destroy();
    }

private:
    std::unique_ptr<Buffer> buffer_;
};

}  // namespace deep_ep_xpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Intel XPU Low Latency MoE Communication Library";
    
    py::class_<deep_ep_xpu::BufferWrapper>(m, "Buffer")
        .def(py::init<py::object, size_t, size_t, bool, int, bool>(),
             py::arg("process_group"),
             py::arg("buffer_size"),
             py::arg("rdma_buffer_size"),
             py::arg("low_latency_mode") = false,
             py::arg("num_qps_per_rank") = 1,
             py::arg("explicitly_destroy") = false)
        .def("low_latency_dispatch", &deep_ep_xpu::BufferWrapper::low_latency_dispatch,
             py::arg("x"), py::arg("topk_idx"),
             py::arg("cumulative_local_expert_recv_stats") = py::none(),
             py::arg("dispatch_wait_recv_cost_stats") = py::none(),
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("num_experts"),
             py::arg("use_fp8") = true,
             py::arg("round_scale") = false,
             py::arg("use_ue8m0") = false,
             py::arg("async_finish") = false,
             py::arg("return_recv_hook") = false)
        .def("low_latency_combine", &deep_ep_xpu::BufferWrapper::low_latency_combine,
             py::arg("x"), py::arg("topk_idx"), py::arg("topk_weights"),
             py::arg("src_info"), py::arg("layout_range"),
             py::arg("combine_wait_recv_cost_stats") = py::none(),
             py::arg("num_max_dispatch_tokens_per_rank"),
             py::arg("num_experts"),
             py::arg("use_logfmt") = false,
             py::arg("zero_copy") = false,
             py::arg("async_finish") = false,
             py::arg("return_recv_hook") = false,
             py::arg("out") = py::none())
        .def("clean_low_latency_buffer", &deep_ep_xpu::BufferWrapper::clean_low_latency_buffer)
        .def("synchronize", &deep_ep_xpu::BufferWrapper::synchronize)
        .def("destroy", &deep_ep_xpu::BufferWrapper::destroy)
        .def_static("get_low_latency_rdma_size_hint", &deep_ep_xpu::Buffer::get_low_latency_rdma_size_hint);
}

