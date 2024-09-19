import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, GATConv

class GHMT(nn.Module):
    """
    Graph Heterogeneous Memory Transformer (GHMT) module designed for graph-based learning
    tasks that involve user-item interactions or similar types of entities.

    This module integrates memory layers and transformer layers to handle features from
    heterogeneous sources, providing a robust feature transformation and interaction mechanism.

    Parameters:
    - args (Namespace): Configuration containing model hyperparameters.
    - n_user (int): Number of user nodes in the graph.
    - n_item (int): Number of item nodes in the graph.

    Attributes:
    - user_item_embedding (Parameter): Embeddings for user and item nodes initialized randomly.
    - feature_normalization (LayerNorm): Normalization layer for combined embeddings.
    - memory_modules (ModuleList): List of memory layers for feature transformation.
    - transformer_modules (ModuleList): List of transformer layers for handling heterogeneous graph data.
    """

    def __init__(self, args, n_user, n_item):
        super(GHMT, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_hidden_units = args.n_hid
        self.number_of_layers = args.n_layers
        self.memory_size = args.mem_size
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

        self.user_item_embedding = nn.Parameter(torch.empty(n_user + n_item, self.n_hidden_units))
        self.feature_normalization = nn.LayerNorm((self.number_of_layers + 1) * self.n_hidden_units)

        self.memory_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()

        for _ in range(self.number_of_layers):
            self.memory_modules.append(
                MemoryLayer(self.n_hidden_units, self.n_hidden_units, self.memory_size, 5, layer_norm=True,
                            dropout=args.dropout,
                            activation=nn.LeakyReLU(0.2, inplace=True)))
            self.transformer_modules.append(
                HeteroGraphTransformerLayer(self.n_hidden_units, self.n_hidden_units, num_heads=4))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes or resets the parameters of the module, particularly the embeddings.
        """
        nn.init.normal_(self.user_item_embedding)

    def forward(self, graph, feature_dictionary):
        """
               Forward pass of the GHMT module.

               Parameters:
               - graph (DGLGraph): The graph on which to perform the operations.
               - feature_dictionary (dict): Dictionary containing node features with node types as keys.

               Returns:
               - Tensor: The transformed node features after processing through memory and transformer modules.
               """
        features = self.user_item_embedding

        all_embeddings = [features]
        for memory_module, transformer_module in zip(self.memory_modules, self.transformer_modules):
            transformed_features = transformer_module(graph, {'user': features[:self.n_user],
                                                              'item': features[self.n_user:]})
            features = torch.cat(list(transformed_features.values()), dim=0)
            features = memory_module(graph, all_embeddings[-1]) + features
            all_embeddings.append(features)

        features = torch.cat(all_embeddings, dim=1)
        features = self.feature_normalization(features)

        return features



class HeteroGraphTransformerLayer2(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(HeteroGraphTransformerLayer2, self).__init__()
        self.attn = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphAttentionConv(in_feats, out_feats, num_heads=num_heads, feat_drop=0.6, attn_drop=0.6,
                                           residual=True)
            for rel in ['user_to_item', 'item_to_user']  # Example relation types
        }, aggregate='sum')

    def forward(self, graph, feat_dict):
        # feat_dict is a dictionary with node types as keys and features as values
        return self.attn(graph, feat_dict)

class HeteroGraphTransformerLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, num_types, residual=False, feat_drop=0.1, attn_drop=0.1):
        super(HeteroGraphTransformerLayer, self).__init__()
        """
        Initialize the HeteroGraphTransformer layer.

        Parameters:
        in_feats (int): Size of each input sample.
        out_feats (int): Size of each output sample.
        num_heads (int): Number of attention heads.
        num_types (int): Number of types of nodes/edges.
        residual (bool): If True, include a residual connection.
        feat_drop (float): Dropout rate for features.
        attn_drop (float): Dropout rate for attention weights.
        """
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.num_types = num_types
        self.residual = residual
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop

        # Different relation types might need different attention mechanisms
        self.attention_layers = nn.ModuleDict({
            f'rel_{i}': GATConv(in_feats, out_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                                residual=residual, activation=torch.nn.functional.elu)
            for i in range(num_types)
        })

        self.type_projection = nn.ModuleDict({
            f'type_{i}': nn.Linear(num_heads * out_feats, out_feats)
            for i in range(num_types)
        })

    def forward(self, graph, feat_dict):
        """
        Forward pass for applying heterogeneous graph transformations.

        Parameters:
        graph (DGLGraph): The graph on which operations are applied.
        feat_dict (dict): Dictionary of features for different node types.

        Returns:
        dict: Updated features for each node type after applying transformations.
        """
        # Processing different node types separately
        for ntype in graph.ntypes:
            if ntype in feat_dict:
                feat_dict[ntype] = feat_dict[ntype].float()

        # Applying different GATConvs for each relation type
        result_dict = {}
        for etype in graph.canonical_etypes:
            src_ntype, relation, dst_ntype = etype
            rel_key = f'rel_{relation}'
            if rel_key in self.attention_layers:
                result = self.attention_layers[rel_key](graph[etype], (feat_dict[src_ntype], feat_dict[dst_ntype]))
                result = result.view(result.shape[0], -1)
                result_dict[dst_ntype] = self.type_projection[f'type_{relation}'](
                    result) if dst_ntype not in result_dict else result_dict[dst_ntype] + self.type_projection[
                    f'type_{relation}'](result)

        # Optionally applying a non-linearity (e.g., ELU)
        for ntype in result_dict:
            result_dict[ntype] = torch.nn.functional.elu(result_dict[ntype])

        return result_dict


class MemoryEncoding(nn.Module):
    def __init__(self, in_feats, out_feats, mem_size):
        """
        Initialize the MemoryEncoding layer.

        Parameters:
        in_feats (int): Size of each input sample.
        out_feats (int): Size of each output sample.
        mem_size (int): Size of the intermediary memory representation.
        """
        super(MemoryEncoding, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size
        self.linear_coef = nn.Linear(in_feats, mem_size, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.linear_w = nn.Linear(mem_size, out_feats * in_feats, bias=False)

    def get_weight(self, x):
        coef = self.linear_coef(x)
        if self.act is not None:
            coef = self.act(coef)
        w = self.linear_w(coef)
        w = w.view(-1, self.out_feats, self.in_feats)
        return w

    def forward(self, h_dst, h_src):
        """
        Forward pass of MemoryEncoding layer.

        Parameters:
        h_dst (Tensor): Destination node features.
        h_src (Tensor): Source node features.

        Returns:
        Tensor: Resulting node features after applying memory encoding.
        """
        w = self.get_weight(h_dst)
        res = torch.einsum('boi, bi -> bo', w, h_src)
        return res

class BPRLoss(nn.Module):
    def __init__(self, lamb_reg):
        super(BPRLoss, self).__init__()
        self.lamb_reg = lamb_reg

    def forward(self, pos_preds, neg_preds, *reg_vars):
        batch_size = pos_preds.size(0)

        bpr_loss = -0.5 * (pos_preds - neg_preds).sigmoid().log().sum() / batch_size
        reg_loss = torch.tensor([0.], device=bpr_loss.device)
        for var in reg_vars:
            reg_loss += self.lamb_reg * 0.5 * var.pow(2).sum()
        reg_loss /= batch_size

        loss = bpr_loss + reg_loss

        return loss, [bpr_loss.item(), reg_loss.item()]


class MemoryLayer(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                mem_size,
                num_rels,
                bias=True,
                activation=None,
                self_loop=True,
                dropout=0.0,
                layer_norm=False):
        super(MemoryLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size

        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        self.node_ME = MemoryEncoding(in_feats, out_feats, mem_size)
        self.rel_ME = nn.ModuleList([
            MemoryEncoding(in_feats, out_feats, mem_size)
                for i in range(self.num_rels)
        ])

        if self.bias:
            self.h_bias = nn.Parameter(torch.empty(out_feats))
            nn.init.zeros_(self.h_bias)

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feats)

        self.dropout = nn.Dropout(dropout)

    def message_func1(self, edges):
        msg = torch.empty((edges.src['h'].shape[0], self.out_feats),
                           device=edges.src['h'].device)
        for etype in range(self.num_rels):
            loc = edges.data['type'] == etype
            if loc.sum() == 0:
                continue
            src = edges.src['h'][loc]
            dst = edges.dst['h'][loc]
            sub_msg = self.rel_ME[etype](dst, src)
            msg[loc] = sub_msg
        return {'m': msg}

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['h'] = feat

            g.update_all(self.message_func1, fn.mean(msg='m', out='h'))
            # g.update_all(self.message_func2, fn.mean(msg='m', out='h'))

            node_rep = g.ndata['h']
            if self.layer_norm:
                node_rep = self.layer_norm_weight(node_rep)
            if self.bias:
                node_rep = node_rep + self.h_bias
            if self.self_loop:
                h = self.node_ME(feat, feat)
                node_rep = node_rep + h
            if self.activation:
                node_rep = self.activation(node_rep)
            node_rep = self.dropout(node_rep)
            return node_rep
