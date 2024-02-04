import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_ntypes, num_etypes, n_heads, dropout=0.2, use_norm=True):
        super(AttentionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_ntypes
        self.num_etypes = num_etypes
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_ws = nn.ModuleList()
        self.q_ws = nn.ModuleList()
        self.v_ws = nn.ModuleList()
        self.a_ws = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(num_ntypes):
            self.k_ws.append(nn.Linear(in_dim, out_dim))
            self.q_ws.append(nn.Linear(in_dim, out_dim))
            self.v_ws.append(nn.Linear(in_dim, out_dim))
            self.a_ws.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
                
        self.relation_pri = nn.Parameter(torch.ones(num_etypes, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_etypes, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_etypes, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['etype_id'][0]
        relation_att = self.relation_att[etype]

        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    @staticmethod
    def message_func(edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, graph):
        for srctype, etype, dsttype in graph.canonical_etypes:
            k_linear = self.k_ws[graph.ntype2id[srctype]]
            v_linear = self.v_ws[graph.ntype2id[srctype]]
            q_linear = self.q_ws[graph.ntype2id[dsttype]]

            graph.nodes[srctype].data['k'] = k_linear(graph.nodes[srctype].data['h']).view(-1, self.n_heads, self.d_k)
            graph.nodes[srctype].data['v'] = v_linear(graph.nodes[srctype].data['h']).view(-1, self.n_heads, self.d_k)
            graph.nodes[dsttype].data['q'] = q_linear(graph.nodes[dsttype].data['h']).view(-1, self.n_heads, self.d_k)

            graph.apply_edges(func=self.edge_attention, etype=etype)

        graph.multi_update_all({etype: (self.message_func, self.reduce_func) for etype in graph.etypes},
                               cross_reducer='mean')

        for ntype in graph.ntypes:
            ntype_id = graph.ntype2id[ntype]
            alpha = torch.sigmoid(self.skip[ntype_id])
            trans_out = self.a_ws[ntype_id](graph.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + graph.nodes[ntype].data['h'] * (1 - alpha)

            if self.use_norm:
                graph.nodes[ntype].data['h'] = self.drop(self.norms[ntype_id](trans_out))
            else:
                graph.nodes[ntype].data['h'] = self.drop(trans_out)


class GraphEmbeddingModel(nn.Module):
    def __init__(self, num_ntypes, num_etypes, n_inp_per_ntype, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(GraphEmbeddingModel, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp_per_ntype = n_inp_per_ntype
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()

        for t in n_inp_per_ntype:
            self.adapt_ws.append(nn.Linear(n_inp_per_ntype[t], n_hid))

        for _ in range(n_layers):
            self.gcs.append(AttentionLayer(n_hid, n_hid, num_ntypes, num_etypes, n_heads, use_norm=use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, graph):
        for ntype in graph.ntypes:
            ntype_id = graph.ntype2id[ntype]
            graph.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[ntype_id](graph.nodes[ntype].data['x']))
        for i in range(self.n_layers):
            self.gcs[i](graph)

    def node_embed(self, graph, ntype):
        return self.out(graph.nodes[ntype].data['h'])

    def graph_embed(self, graph):
        node_embeddings = []
        for ntype in graph.ntypes:
            node_embedding = self.node_embed(graph, ntype)
            node_embeddings.append(node_embedding)
        node_embeddings = torch.cat(node_embeddings, dim=0)
        graph_embedding = node_embeddings.mean(dim=0).unsqueeze(dim=0)
        return graph_embedding

    def graphs_embed(self, graphs):
        graph_embeddings = []
        for graph in graphs:
            graph_embedding = self.graph_embed(graph)
            graph_embeddings.append(graph_embedding)
        return torch.cat(graph_embeddings, dim=0)


class NodePredictor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim),
                                   nn.Softmax())

    def forward(self, h):
        return self.model(h)


class EdgePredictor(nn.Module):
    def __init__(self, src_feat_dim, dst_feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden = nn.Linear(src_feat_dim + dst_feat_dim, hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Softmax())

    def apply_edges(self, edges):
        h = torch.cat([edges.src['src_h'], edges.dst['dst_h']], -1)
        return {'score': self.out(F.relu(self.hidden(h))).squeeze(1)}

    def forward(self, g, src_h, dst_h):
        with g.local_scope():
            g.srcdata['src_h'] = src_h[g.srcnodes().long(), :]
            g.dstdata['dst_h'] = dst_h[g.dstnodes().long(), :]
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class DistEstNet(nn.Module):
    def __init__(self, latent_dim, h_dim, gamma_dim, dropout):
        super().__init__()
        self.calc_gamma = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, gamma_dim),
            nn.Sigmoid()
        )

    @staticmethod
    def _calc_gmm_params(latent_samples, gamma):
        gamma_sum = gamma.sum(dim=0)
        phi = gamma_sum / gamma.size(0)
        mu = (gamma.unsqueeze(-1) * latent_samples.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(-1)
        z_mu = latent_samples.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        sigma = (gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer).sum(dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)
        return phi, mu, sigma

    @staticmethod
    def _to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def _calc_energy(self, latent_samples, phi, mu, sigma):
        k, d, _ = sigma.size()
        z_mu = (latent_samples.unsqueeze(1) - mu.unsqueeze(0))
        sigma_inverse = []
        sigma_det = []
        sigma_diag = 0
        eps = 1e-12
        for i in range(k):
            sigma_k = sigma[i] + self._to_var(torch.eye(d) * eps)
            sigma_inverse.append(torch.inverse(sigma_k).unsqueeze(0))
            sigma_det.append((torch.linalg.cholesky(sigma_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            sigma_diag = sigma_diag + torch.sum(1 / sigma_k.diag())

        cov_inverse = torch.cat(sigma_inverse, dim=0)
        sigma_det = torch.cat(sigma_det)
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(sigma_det)).unsqueeze(0), dim=1) + eps)

        return sample_energy, sigma_diag

    def calc_energy(self, latent_samples):
        gamma = self.calc_gamma(latent_samples)
        phi, mu, sigma = self._calc_gmm_params(latent_samples, gamma)
        sample_energy, sigma_diag = self._calc_energy(latent_samples, phi, mu, sigma)
        return sample_energy, sigma_diag

    def forward(self, latent_samples, lambda_energy=0.2, lambda_sigma=0.02):
        sample_energy, sigma_diag = self.calc_energy(latent_samples)
        loss = lambda_energy * sample_energy + lambda_sigma * sigma_diag
        return loss
