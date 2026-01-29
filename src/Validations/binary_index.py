# binary_index.py
import torch
import hdc_cuda

class bp_hamming:
    def __init__(self, dim_bits):
        assert dim_bits % 64 == 0, "Only dimensions divisible by 64 are supported."
        self.dim_bits = dim_bits
        self.nwords = dim_bits // 64
        self.device = torch.device('cuda')

        self.data = []
        self.ntotal = 0

    def add(self, vecs):
        if vecs.dim() == 1:
            vecs = vecs.unsqueeze(0)
        assert vecs.shape[1] == self.nwords and vecs.dtype == torch.int64
        vecs = vecs.to(self.device)
        self.data.append(vecs)
        self.ntotal += vecs.size(0)

    def reset(self):
        self.data = []
        self.ntotal = 0

    def search(self, query, k):
        assert self.ntotal > 0, "Index is empty."
        if isinstance(query, list):
            query = torch.stack(query)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = query.to(self.device)
        all_data = torch.cat(self.data, dim=0)  # [N, nwords]

        dists = hdc_cuda.hamming_cuda(all_data, query)  # [N]
        top_k = torch.topk(dists, k, largest=False)

        return top_k.values.unsqueeze(0), top_k.indices.unsqueeze(0)

    def __len__(self):
        return self.ntotal
