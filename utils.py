# encoding:utf-8
import dgl
from dgl import DGLError
import numpy as np
import torch

from settings import MetaRelations

META_RELATIONS = [
    MetaRelations.OPE_RE,
    MetaRelations.OPE_RE_,
    MetaRelations.QUE_RE,
    MetaRelations.QUE_RE_,
    MetaRelations.UPD_RE,
    MetaRelations.UPD_RE_,
    MetaRelations.ADD_RE,
    MetaRelations.ADD_RE_,
    MetaRelations.DEL_RE,
    MetaRelations.DEL_RE_,
    MetaRelations.DOW_RE,
    MetaRelations.DOW_RE_,
]


def gen_graph_pairs(graph):
    graphs_pairs = {}
    for meta_rel in META_RELATIONS:
        src_ntype, etype, dst_ntype = meta_rel
        try:
            src_node, dst_node = graph.edges(etype=etype)
        except DGLError:
            continue
        adj_mat = np.zeros((graph.num_src_nodes(ntype=src_ntype), graph.num_dst_nodes(ntype=dst_ntype)),
                           dtype=np.float32)
        adj_mat[src_node.numpy(), dst_node.numpy()] = 1.0
        src_neg, dst_neg = np.where(adj_mat != 1.0)
        if src_neg.shape[0] == 0:
            neg_graph = None
        else:
            index = np.random.permutation(range(src_neg.shape[0]))[:src_node.size(0)]
            sampled_neg_src = torch.from_numpy(src_neg[index])
            sampled_neg_dst = torch.from_numpy(dst_neg[index])
            neg_graph = dgl.heterograph({meta_rel: (sampled_neg_src, sampled_neg_dst)})

        pos_graph = graph.edge_type_subgraph(etypes=[etype])
        graphs_pairs[meta_rel] = [pos_graph, neg_graph]
    return graphs_pairs
