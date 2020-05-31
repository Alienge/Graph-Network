import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self,in_feature,out_feature,dropout,alpha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.dropout = dropout
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alpha = alpha
        self.concat = concat
        #self.nheads = n_heads

        self.W = nn.Parameter(torch.zeros(size=(self.in_feature,self.out_feature)))
        nn.init.xavier_normal(self.W.data,gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_feature,1)))
        nn.init.xavier_normal(self.a.data,gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,input,adj):
        h = torch.mm(input,self.W)# [N,F']
        N = h.size()[0]
        '''
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        mask = -9e15 * (1.0 - adj)
        attention = e + mask


        #zero_vec = -9e15 * torch.ones_like(e)
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        '''

        #a1 = self.a[0:self.out_feature,:] # [F',1]
        #a2 = self.a[self.out_feature:2*self.out_feature+1,:] # [F',1]
        #print("================a1 shape======>",self.a.shape)
        #print("================a2 shape======>", a2.shape)
        # divide the vector of "a" to two part :
        # attention for self node and attention for neighbor's node
        attn_for_self = torch.mm(h,self.a[0:self.out_feature,:])
        attn_for_neighs = torch.mm(h,self.a[self.out_feature:,:])
        #print("=================================>",attn_for_self.shape)
        dense = attn_for_self + attn_for_neighs.T #[N,N]
        #print("=================================>",dense.shape)

        dense = self.leakyrelu(dense)

        # infinite the unrelated edges
        #mask = -9e15*(1.0-adj)
        #dense = dense + mask
        zero_vec = -9e15 * torch.ones_like(dense)
        attention = torch.where(adj > 0, dense, zero_vec)
        # get the value of the attention
        attention = F.softmax(attention,dim=1)
        attention = F.dropout(attention,self.dropout,training=self.training)
        
        # get output features value
        h_prime = torch.matmul(attention,h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





