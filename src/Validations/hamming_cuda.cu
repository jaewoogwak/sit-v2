#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ int popcount(scalar_t x) {
    return __popcll(static_cast<unsigned long long>(x));
}

// Kernel: A [N, W], B [1, W]
template <typename scalar_t>
__global__ void hamming_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    int* __restrict__ out,
    int N,
    int W
) {
    int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_global_id >= N) return;

    int dist = 0;
    for (int w = lane_id; w < W; w += 32) {
        scalar_t a = A[warp_global_id * W + w];
        scalar_t b = B[w];
        dist += popcount(a ^ b);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        dist += __shfl_down_sync(0xffffffff, dist, offset);
    }

    if (lane_id == 0) {
        out[warp_global_id] = dist;
    }
}


// Kernel: A [N, W], B [M, W]
template <typename scalar_t>
__global__ void hamming_pairwise_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    int* __restrict__ out,
    int N, int M, int W
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    int total = N * M;
    if (warp_id >= total) return;

    int row = warp_id / M;
    int col = warp_id % M;

    int dist = 0;
    for (int w = lane_id; w < W; w += 32) {
        scalar_t a = A[row * W + w];
        scalar_t b = B[col * W + w];
        dist += popcount(a ^ b);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        dist += __shfl_down_sync(0xffffffff, dist, offset);
    }

    if (lane_id == 0) {
        out[row * M + col] = dist;
    }
}

template <typename scalar_t>
__global__ void hamming_indexed_kernel(
    const scalar_t* __restrict__ A_full,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ B,
    int* __restrict__ out,
    int N, int W
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= N) return;

    int row_idx = indices[warp_id];  // Indexed row
    int offset = row_idx * W;
    int dist = 0;
    for (int w = lane_id; w < W; w += 32) {
        scalar_t a = A_full[offset + w];
        scalar_t b = B[w];
        dist += popcount(a ^ b);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        dist += __shfl_down_sync(0xffffffff, dist, offset);
    }

    if (lane_id == 0) {
        out[warp_id] = dist;
    }
}


template <typename scalar_t>
__global__ void hamming_indexed_pairwise_kernel(
    const scalar_t* __restrict__ A_full,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ B,
    int* __restrict__ out,
    int N, int M, int W
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    int total = N * M;
    if (warp_id >= total) return;

    int row = warp_id / M;
    int col = warp_id % M;

    int a_idx = indices[row];  // lookup from global data

    int dist = 0;
    for (int w = lane_id; w < W; w += 32) {
        scalar_t a = A_full[a_idx * W + w];
        scalar_t b = B[col * W + w];
        dist += popcount(a ^ b);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        dist += __shfl_down_sync(0xffffffff, dist, offset);
    }

    if (lane_id == 0) {
        out[row * M + col] = dist;
    }
}


torch::Tensor hamming_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(B.size(0) == 1, "B must be a single query [1, W]");
    TORCH_CHECK(A.size(1) == B.size(1), "Bit width must match");

    const auto N = A.size(0);
    const auto W = A.size(1);

    auto out = torch::empty({N}, torch::dtype(torch::kInt32).device(A.device()));

    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "hamming_cuda", ([&] {
        hamming_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<int>(),
            N, W
        );
    }));

    return out;
}

torch::Tensor hamming_pairwise_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Bit width must match");

    const auto N = A.size(0);
    const auto M = B.size(0);
    const auto W = A.size(1);

    auto out = torch::empty({N, M}, torch::dtype(torch::kInt32).device(A.device()));

    const int total_pairs = N * M;
    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (total_pairs + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_INTEGRAL_TYPES(A.scalar_type(), "hamming_pairwise_cuda", ([&] {
        hamming_pairwise_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<int>(),
            N, M, W
        );
    }));

    return out;
}

torch::Tensor hamming_indexed_cuda(torch::Tensor A_full, torch::Tensor indices, torch::Tensor B) {
    TORCH_CHECK(A_full.dim() == 2 && B.dim() == 2, "A_full and B must be 2D");
    TORCH_CHECK(B.size(0) == 1, "B must be [1, W]");
    TORCH_CHECK(A_full.size(1) == B.size(1), "Bit width mismatch");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");

    const auto N = indices.size(0);
    const auto W = A_full.size(1);

    auto out = torch::empty({N}, torch::dtype(torch::kInt32).device(A_full.device()));

    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_INTEGRAL_TYPES(A_full.scalar_type(), "hamming_indexed_cuda", ([&] {
        hamming_indexed_kernel<scalar_t><<<blocks, threads>>>(
            A_full.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<int>(),
            N, W
        );
    }));

    return out;
}

torch::Tensor hamming_indexed_pairwise_cuda(
    torch::Tensor A_full, torch::Tensor indices, torch::Tensor B) {

    TORCH_CHECK(A_full.dim() == 2 && B.dim() == 2, "A_full and B must be 2D");
    TORCH_CHECK(A_full.size(1) == B.size(1), "Bit width mismatch");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");

    const auto N = indices.size(0);
    const auto M = B.size(0);
    const auto W = A_full.size(1);

    auto out = torch::empty({N, M}, torch::dtype(torch::kInt32).device(A_full.device()));

    const int total = N * M;
    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_INTEGRAL_TYPES(A_full.scalar_type(), "hamming_indexed_pairwise_cuda", ([&] {
        hamming_indexed_pairwise_kernel<scalar_t><<<blocks, threads>>>(
            A_full.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<int>(),
            N, M, W
        );
    }));

    return out;
}


//torch::Tensor hdnsw_search(
//        int N, int dim_bits, int k,
//        torch::Tensor query,             // [1, W] on GPU
//        torch::Tensor entry_points,  // on GPU
//        torch::Tensor data,
//        ) {
//    auto max_hamming = dim_bits;
//
//    // Init tracking
//    auto device = query.device();
//    auto topk_ids = torch::full({k}, -1, torch::dtype(torch::kInt64).device(device));
//    auto topk_dists = torch::full({k}, max_hamming, torch::dtype(torch::kInt32).device(device));
//    auto dists = torch::full({N}, max_hamming, torch::dtype(torch::kInt32).device(device));
//    auto seen_bundles = torch::zeros({N}, torch::dtype(torch::kBool).device(device));
//
//    // === Use multiple entry points ===
//    seen_bundles.index_put_({entry_points}, true);
//
//    // Pre-score entry points
//    auto entry_dists = hamming_indexed_pairwise_cuda(data, entry_dists, query);
//    dists.index_put_({entry_points}, entry_dists.squeeze(1));
//    dists[beam] = entry_dists;
//
//    int loop_idx = 0;
//    while (true) {
//        loop_idx += 1;
//
//    }
//}
