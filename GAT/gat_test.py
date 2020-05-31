import torch

#a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#  (N,N*F')  (N*N,F)



h = torch.tensor([[1,2],[3,4]])
a = torch.tensor([[1],[2],[3],[4]])
N = 2
out_features = 2
a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * out_features)
output = torch.matmul(a_input, a)
print(output.squeeze(2))


attn_for_self = torch.mm(h,a[0:out_features,:])
attn_for_neighs = torch.mm(h,a[out_features:,:])
#print("=================================>",attn_for_self.shape)
dense = attn_for_self + attn_for_neighs.T #[N,N]
print(dense)