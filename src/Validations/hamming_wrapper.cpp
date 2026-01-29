// hamming_wrapper.cpp
#include <torch/extension.h>

torch::Tensor hamming_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor hamming_pairwise_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor hamming_indexed_cuda(torch::Tensor A_full, torch::Tensor indices, torch::Tensor B);
torch::Tensor hamming_indexed_pairwise_cuda(torch::Tensor A_full, torch::Tensor indices, torch::Tensor B);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamming_cuda", &hamming_cuda, "Bitpacked Hamming Distance (CUDA)");
    m.def("hamming_pairwise_cuda", &hamming_pairwise_cuda, "Pairwise Bitpacked Hamming Distance (CUDA)");
    m.def("hamming_indexed_cuda", &hamming_indexed_cuda, "Bitpacked Hamming Distance with Row Indices (CUDA)");
    m.def("hamming_indexed_pairwise_cuda", &hamming_indexed_pairwise_cuda, "Pairwise Hamming Distance with Indexed A (CUDA)");
}
