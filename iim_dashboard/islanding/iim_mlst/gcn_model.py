import torch
import numpy as np
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering,FeatureAgglomeration
from utilities import weighted_KMeans


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg,K=3):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *self.K, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def concat(self,x,x_):
        x_=torch.unsqueeze(x_,0)
        return torch.cat([x,x_],dim=0)

    def forward_back(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        if self.K>1:
            agg_feats = self.agg(features,A)
            cat_feats = torch.cat([features, agg_feats], dim=2)
       
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = out + self.bias

        return out

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        x0=features.permute(1,2,0)
        x0=torch.reshape(x0,[n,b*d])
        x=torch.unsqueeze(x0,0)
        if self.K>1:
            x1=torch.sparse.mm(A,x0)
            x=self.concat(x,x1)
        for k in range(2,self.K):
            x2=2*torch.sparse.mm(A,x1)-x0
            x=self.concat(x,x2)
        x=x.view(self.K,n,d,b)
        x=x.permute(3,1,2,0)
        x = torch.reshape(x, (b*n, d*self.K))
        x=torch.matmul(x,self.weight)
        out=x+self.bias
        out=torch.reshape(out,[b,n,self.out_dim])

        return out


class GraphEncoder(nn.Module):
    def __init__(self, layers, clusters,adj_matrix,input_feature_dim,device,cluster_num=0):
        super(GraphEncoder, self).__init__()
        self.device=device
        self.cluster_num=cluster_num
        self.adj_matrix=adj_matrix
        self.num_node=len(adj_matrix)
        self.input_feature_dim=input_feature_dim
        self.block_conv1 = GraphConv(self.input_feature_dim, 48, MeanAggregator)
        self.sig1 = nn.ReLU()
        self.block_conv2 = GraphConv(48, 96, MeanAggregator)
        self.sig2 = nn.ReLU()
        self.block_conv3 = GraphConv(96, 128, MeanAggregator)
        self.sig3 = nn.ReLU()
        self.block_conv4 = GraphConv(128, 256, MeanAggregator)
        self.sig4 = nn.ReLU()
        self.clusters = clusters
        self.output_layers = nn.Sequential(OrderedDict({
            'lin2': nn.Linear(256, 30*cluster_num),
            'lin3': nn.Linear(30*cluster_num, cluster_num),
            'sig4': nn.Softmax(dim=-1),
        }))
        self.outputs = {}
        self.soft_cross=nn.CrossEntropyLoss()
        self.output_layers[0].register_forward_hook(self.get_activation('lin2'))

    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def forward(self, x):
        x=x.unsqueeze(0)
        self.x=self.block_conv1(x,self.adj_matrix)
        self.x=self.sig1(self.x)
        self.x = self.block_conv2(self.x,self.adj_matrix)
        self.x = self.sig2(self.x)
        self.x = self.block_conv3(self.x,self.adj_matrix)
        self.x = self.sig3(self.x)
        output = self.block_conv4(self.x,self.adj_matrix)
        output=self.sig4(output)
        output=self.output_layers(output)
        output = output.squeeze()
        return output
    def forward_old(self, x):
        output = self.layers(x)
        return output
    def gap_loss(self,Y_output,adjacent_matrix,BUS_INCE,min_cut_pram=1,balance_para=0.1,inj_pram=0.01,gen_param=1,gen_index=None):
        '''Y_output:n*g tensor


        '''

        n,g=Y_output.shape[0],Y_output.shape[1]
        adjacent_matrix=torch.from_numpy(adjacent_matrix).float().to(self.device)
        Degrees=torch.sum(adjacent_matrix,axis=0).unsqueeze(1).float()
        BUS_INCE=torch.from_numpy(BUS_INCE).unsqueeze(1).float()

        Y_trans=Y_output.t()
        Gam=torch.mm(Y_trans,Degrees)
        Gam=1.0/Gam
        Gam=Gam.squeeze()
        first_term=Y_output * Gam
        seconde_term=(1-Y_output).t()
        Loss_cut=torch.sum( torch.mm(first_term,seconde_term) *adjacent_matrix             )
        one_lines=torch.ones(1,n).to(self.device)
        before_min=torch.mm(one_lines,Y_output)
        compare_zeros=torch.zeros_like(before_min)
        sub_num=np.int64(n / g/3)
        loss_balance=torch.sum(torch.max(compare_zeros,(n / g-sub_num) - before_min))

        every_loc=torch.abs(torch.mm(Y_trans,BUS_INCE))
        loss_bus_inc=torch.sum(every_loc)

        gen_index=gen_index[:g]
        bus_gen=Y_output[gen_index,:].to(self.device)

        labels=np.array(range(0,g))
        labels=torch.from_numpy(labels).long().to(self.device)
        loss_gen=self.soft_cross(bus_gen,labels)

        # total loss
        loss = min_cut_pram * Loss_cut + balance_para * loss_balance  + gen_param * loss_gen
        return loss,Loss_cut,loss_balance,loss_bus_inc,loss_gen

    def min_cut_loss(self,adj,s,BUS_INCE,balance_para=0.1,inj_pram=0.01,gen_param=1,gen_index=None):
        adj=torch.from_numpy(adj)#.cuda()
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s=s.float()
        adj=adj.float()

        k=s.size(-1)
        s = torch.softmax(s, dim=-1)

        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # MinCUT regularization.
        mincut_num = _rank3_trace(out_adj)
        d_flat = torch.einsum('ijk->ij', adj)
        d = _rank3_diag(d_flat)
        mincut_den = _rank3_trace(
            torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
        mincut_loss = -(mincut_num / mincut_den)
        mincut_loss = torch.mean(mincut_loss)

        # Orthogonality regularization.
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # balance_cut
        s=s.squeeze()
        n,g=s.size(0),s.size(1)
        one_lines = torch.ones(1, n)#.cuda()
        before_min = torch.mm(one_lines, s)

        compare_zeros = torch.zeros_like(before_min)
        sub_num = np.int64(n / g / 3)
        loss_balance = torch.sum(torch.max(compare_zeros, (n / g - sub_num) - before_min))

        #bus injec
        BUS_INCE = torch.from_numpy(BUS_INCE).unsqueeze(1).float()#.cuda()
        Y_trans=s.t()
        Y_output=s
        every_loc = torch.abs(torch.mm(Y_trans, BUS_INCE))
       
        loss_bus_inc = torch.sum(every_loc)

        # generator to be seperate
        gen_index = gen_index[:g]
        bus_gen = Y_output[gen_index, :]

        labels = np.array(range(0, g))
        labels = torch.from_numpy(labels).long()
        loss_gen = self.soft_cross(bus_gen, labels)

        total_loss=mincut_loss+ortho_loss+ balance_para*loss_balance+ inj_pram*loss_bus_inc +gen_param*loss_gen
        return total_loss

    def layer_activations(self, layername):
        return torch.mean(torch.sigmoid(self.outputs[layername]), dim=0)

    def sparse_result(self, rho, data):
        data=data.squeeze()
        rho_hat = torch.mean(torch.sigmoid(data), dim=0)
        return rho * np.log(rho) - rho * torch.log(rho_hat) + (1 - rho) * np.log(1 - rho) \
               - (1 - rho) * torch.log(1 - rho_hat)

    def kl_div(self, rho):
        first = torch.mean(self.sparse_result(rho, self.record1))
        second = torch.mean(self.sparse_result(rho, self.record2))
        return first + second

    def get_index_by_name(self, name):
        return list(dict(self.layers.named_children()).keys()).index(name)

    def loss(self, x_hat, x, beta, rho,gen_index,rank_hyper=5.0):
        loss = F.mse_loss(x_hat, x) + beta * self.kl_div(rho)
        return loss,0

    def get_cluster(self):
        kmeans = KMeans(n_clusters=self.clusters).fit(self.record1.squeeze().detach().cpu().numpy())
        self.centroids = kmeans.cluster_centers_
        return kmeans.labels_

    def get_cluster_weighted_kmeans(self, generator_index):
        centroids, label = weighted_KMeans(self.outputs['lin2'].detach().cpu().numpy(), generator_index=generator_index)
        return label

    def get_cluster_spectral(self):
        labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
        self.centroids = labels.cluster_centers_
        return labels.labels_

    def get_cluster_FeatureAgglomeration(self):
        a = self.outputs['lin2'].detach().cpu().numpy()
        labels = FeatureAgglomeration(n_clusters=self.clusters).fit(np.transpose(a))
        return labels.labels_

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, x, L ):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)
        
        x = x.permute(1, 2, 0)

        x = torch.reshape(x, (M, Fin * N))
        x = torch.sparse.mm(L, x)
        x = x.view(Mp, Fin, N)
        x = x.permute(2, 0, 1)
        return x


class CutLoss(torch.autograd.Function):
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614
    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix
    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    @staticmethod
    def forward(ctx, Y, A):
        ctx.save_for_backward(Y,A)
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        YbyGamma = torch.div(Y, Gamma.t())
        Y_t = (1 - Y).t()

        loss = torch.tensor([0.], requires_grad=True).to('cuda')
        idx = A._indices()
        data = A._values()
        for i in range(idx.shape[1]):
            loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i]
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        Y, A, = ctx.saved_tensors
        idx = A._indices()
        data = A._values()
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        gradient = torch.zeros_like(Y)
        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                alpha_ind = (idx[0, :] == i).nonzero()
                alpha = idx[1, alpha_ind]
                A_i_alpha = data[alpha_ind]
                temp = A_i_alpha / torch.pow(Gamma[j], 2) * (Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * (
                            Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j])))
                gradient[i, j] = torch.sum(temp)

                l_idx = list(idx.t())
                l2 = []
                l2_val = []
                for ptr, mem in enumerate(l_idx):
                    if ((mem[0] != i).item() and (mem[1] != i).item()):
                        l2.append(mem)
                        l2_val.append(data[ptr])
                extra_gradient = 0
                if (l2 != []):
                    for val, mem in zip(l2_val, l2):
                        extra_gradient += (-D[i] * torch.sum(
                            Y[mem[0], j] * (1 - Y[mem[1], j]) / torch.pow(Gamma[j], 2))) * val

                gradient[i, j] += extra_gradient
        return gradient, None

def _rank3_trace(x):
    return torch.einsum('ijj->i', x)
    
def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out