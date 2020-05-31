import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,input,adj):
        support = torch.matmul(input,self.weights)
        output = torch._sparse_mm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


