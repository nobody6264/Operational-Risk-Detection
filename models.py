# encoding: utf-8

import torch
from modules import GraphEmbeddingModel, NodePredictor, EdgePredictor
from modules import DistEstNet
from settings import MetaRelations, HyperParam


class GraphSeqEmbeddingModel(torch.nn.Module):
    def __init__(self,
                 num_ntypes=HyperParam.NUM_NTYPES,
                 num_etypes=HyperParam.NUM_ETYPES,
                 n_inp_per_ntype=HyperParam.N_INP_PER_NTYPE,
                 n_hid=HyperParam.N_HID,
                 n_layers=HyperParam.N_LAYERS,
                 n_heads=HyperParam.N_HEADS,
                 embedding_dim=HyperParam.EMBEDDING_DIM,
                 alpha=HyperParam.ALPHA,
                 node_labels=HyperParam.NODE_LABELS,
                 edge_labels=HyperParam.EDGE_LABELS
                 ):
        super(GraphSeqEmbeddingModel, self).__init__()

        self.alpha = torch.nn.Parameter(torch.FloatTensor([alpha]))

        # initialize the model
        # aggregate the main parameters and define the embedding model
        self.embedding_model = GraphEmbeddingModel(num_ntypes, num_etypes, n_inp_per_ntype, n_hid, embedding_dim,
                                                   n_layers, n_heads)

        # define the prediction model for node label
        self.node_label_predictor = NodePredictor(embedding_dim, embedding_dim, node_labels)

        # define the prediction model for graph structure
        self.structure_predictor = EdgePredictor(embedding_dim, embedding_dim, embedding_dim, edge_labels)

        self.density_est_net = DistEstNet(embedding_dim, embedding_dim, 1, 0.05)

        # define the classification loss and the optimizer.
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, graph_seq, contrastive_pairs, calc_energy):
        num_graphs = len(graph_seq)
        mean_structural_loss = 0
        mean_node_loss = 0
        for graph, contrastive_pair in zip(graph_seq, contrastive_pairs):
            # define the label for every node types
            node_label = torch.LongTensor([0] * graph.num_nodes(ntype='staff') +
                                          [1] * graph.num_nodes(ntype='address') +
                                          [2] * graph.num_nodes(ntype='system'))

            # predict the node embedding
            self.embedding_model(graph)

            # reconstruct the semantic information of node
            staff_embedding = self.embedding_model.node_embed(graph, 'staff')
            address_embedding = self.embedding_model.node_embed(graph, 'address')
            system_embedding = self.embedding_model.node_embed(graph, 'system')
            node_embeddings = torch.cat([staff_embedding, address_embedding, system_embedding], dim=0)
            node_pred = self.node_label_predictor(node_embeddings)
            node_semantic_loss = self.loss_fn(node_pred, node_label)
            mean_node_loss += node_semantic_loss / num_graphs

            # reconstruct the structural information of edge of different types
            embedding_pairs = {
                MetaRelations.OPE_RE: (staff_embedding, address_embedding),
                MetaRelations.DEL_RE: (address_embedding, system_embedding),
                MetaRelations.ADD_RE: (address_embedding, system_embedding),
                MetaRelations.UPD_RE: (address_embedding, system_embedding),
                MetaRelations.QUE_RE: (address_embedding, system_embedding),
                MetaRelations.DOW_RE: (address_embedding, system_embedding)}

            structural_loss = 0.
            for meta_rel in embedding_pairs:
                pos_graph, neg_graph = contrastive_pair.get(meta_rel, (None, None))
                if pos_graph is None:
                    continue

                pos_scores = self.structure_predictor(pos_graph, *embedding_pairs[meta_rel])
                if neg_graph is not None:
                    neg_scores = self.structure_predictor(neg_graph, *embedding_pairs[meta_rel])
                    scores = torch.cat([pos_scores, neg_scores])
                    labels = torch.cat([torch.ones(pos_scores.size(0), dtype=torch.int64),
                                        torch.zeros(neg_scores.size(0), dtype=torch.int64)])
                else:
                    scores = pos_scores
                    labels = torch.ones(pos_scores.size(0), dtype=torch.int64)

                loss_per_relation = self.loss_fn(scores, labels)
                if not torch.isnan(loss_per_relation):
                    structural_loss += loss_per_relation

            mean_structural_loss += structural_loss / num_graphs

        # compute the probability density of graph sequence.
        if calc_energy:
            graph_embeddings = self.embedding_model.graphs_embed(graph_seq)
            energy_loss = self.density_est_net(graph_embeddings).mean()
        else:
            energy_loss = torch.FloatTensor([0])

        # total loss
        loss_weight = torch.sigmoid(self.alpha)
        total_loss = (loss_weight * mean_node_loss + (1 - loss_weight) * mean_structural_loss + 1e-6 * energy_loss)
        return mean_node_loss, mean_structural_loss, energy_loss, total_loss, loss_weight

    def train(self, graph_seq,
              contrastive_pairs,
              calc_energy=HyperParam.CALC_ENERGY,
              epochs=HyperParam.EPOCHS,
              lr=HyperParam.LR,
              weight_decay=HyperParam.WEIGHT_DECAY):

        optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay, lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(epochs):
            (mean_node_loss, mean_structural_loss,
             energy_loss, total_loss, loss_weight) = self(graph_seq, contrastive_pairs, calc_energy)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print('epoch: {}, lr: {:.6f}, '
                  'loss_weight {:.4f}, '
                  'node_loss: {:.4f}, '
                  'structure_loss: {:.4f}, '
                  'energy_loss: {:.4f}, '
                  'total_loss: {:.4f}'.format(epoch,
                                              optimizer.param_groups[0]['lr'],
                                              loss_weight.item(),
                                              mean_node_loss.item(),
                                              mean_structural_loss.item(),
                                              energy_loss.item(),
                                              total_loss.item(),
                                              ))

    def node_embed(self, graph, ntype):
        embeddings = self.embedding_model.node_embed(graph, ntype)
        return embeddings

    def graph_embed(self, graph):
        return self.embedding_model.graph_embed(graph)

    def graphs_embed(self, graphs):
        return self.embedding_model.graphs_embed(graphs)

    def calc_energy(self, graphs):
        graph_embeddings = self.embedding_model.graphs_embed(graphs)
        energy, _ = self.density_est_net.calc_energy(graph_embeddings)
        return energy

    def calc_gamma(self, graphs):
        graph_embeddings = self.embedding_model.graphs_embed(graphs)
        gamma = self.density_est_net.calc_gamma(graph_embeddings)
        return gamma
