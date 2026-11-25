import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
import torch.nn.functional as F
import pandas as pd
from torch_geometric.utils import to_networkx, from_networkx,degree
from scipy.stats import rankdata
from sklearn.cluster import KMeans
import math

class CitationDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 domain,
                 lblToRemove1,
                 lblToRemove2,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.domain = domain
        self.lblToRemove1 = lblToRemove1
        self.lblToRemove2 = lblToRemove2
        # self.root = root  # data/acmv9
        super(CitationDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y_df = pd.DataFrame(y)

        data = Data(edge_index=edge_index, x=x, y=y)

        label_to_remove = [self.lblToRemove1, self.lblToRemove2]

        if self.domain == 'source':
            print("source_name:",self.name)
            print("before_process_source_nodes_of_class:")
            for i in range(5):
                print("label:{}, number:{}".format(i, np.count_nonzero(data.y == i)))

            data_nx = to_networkx(data=data, node_attrs=["x"], to_undirected=False)
            data_labels = pd.read_csv(label_path, sep=' ', header=None)
            node_to_remove = []
            for l in label_to_remove:
                node_to_remove = node_to_remove + (data_labels[(data_labels[0] == l)].index.tolist())
            data_nx.remove_nodes_from(node_to_remove)
            y_df_new = y_df.drop(node_to_remove)
            y_array_new = y_df_new[0].to_numpy()
            y_array_new = rankdata(y_array_new, method='dense') - 1
            y = torch.from_numpy(y_array_new).to(torch.int64)
            data = from_networkx(data_nx)
            data.y = y

            print("after_process_source_nodes_of_class:")
            for i in range(3):
                print("label:{}, number:{}".format(i, np.count_nonzero(data.y == i)))

        if self.domain == 'target':

            print("target_name:",self.name)
            print("before_process_target_nodes_of_class:")
            for i in range(5):
                print("label:{}, number:{}".format(i, np.count_nonzero(data.y == i)))

            y_original = y.copy()
            y[np.any([y_original==self.lblToRemove1, y_original==self.lblToRemove2], axis=0)] = 99
            y = rankdata(y, method='dense') - 1
            y = torch.from_numpy(y).to(torch.int64)
            data.y = y
            print("after_process_target_nodes_of_class:")
            for i in range(4):
                print("label:{}, number:{}".format(i, np.count_nonzero(data.y == i)))

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


class UnknownDetector:

    def __init__(self, source_labels, num_known_classes, beta, gamma):
        counts = torch.bincount(source_labels, minlength=num_known_classes).float()
        self.prior = counts / counts.sum()
        self.beta = beta
        self.source_rank = max(1, math.ceil(gamma * source_labels.numel()))
        self.threshold = None

    def _divergence(self, logits):
        log_probabilities = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probabilities, self.prior, reduction='none').sum(dim=1)

    @torch.no_grad()
    def __call__(self, source_logits, target_logits, target_edge_index):
        source_divergence = self._divergence(source_logits)
        cutoff = source_divergence.kthvalue(self.source_rank).values
        if self.threshold is None:
            self.threshold = cutoff
        else:
            self.threshold = self.beta * self.threshold + (1 - self.beta) * cutoff

        candidate = self._divergence(target_logits) < self.threshold
        edge_from, edge_to = target_edge_index
        supported = torch.zeros_like(candidate)
        supporting_edges = candidate[edge_from] & (edge_from != edge_to)
        supported.index_fill_(0, edge_to[supporting_edges], True)
        unknown = candidate & supported
        return torch.where(~unknown), torch.where(unknown)

def get_neighbors(logits_source, logits_target, topk, device):
    prob_t, prob_s = F.softmax(logits_target.detach().cpu(), dim=1), F.softmax(logits_source.detach().cpu(), dim=1)
    rank_t = prob_t
    rank_idx_t = torch.argsort(rank_t, dim = 1, descending = True)
    rank_idx_t_aug = rank_idx_t.repeat(prob_s.size(0), 1)
    rank_s = prob_s
    rank_idx_s = torch.argsort(rank_s, dim = 1, descending = True)
    rank_idx_s_aug = rank_idx_s.repeat(1, prob_t.size(0)).view(-1, rank_s.size(1))
    rank_idx1, rank_idx2 = rank_idx_t_aug[:,:topk], rank_idx_s_aug[:,:topk]
    rank_idx1, _ = torch.sort(rank_idx1, dim = 1)
    rank_idx2, _ = torch.sort(rank_idx2, dim = 1)
    rank_diff = rank_idx1 - rank_idx2
    rank_diff = torch.mean(torch.abs(rank_diff).float(), dim = 1)
    neighbor = torch.ones_like(rank_diff).float().to(device)
    neighbor[rank_diff > 0] = 0
    neighbors = neighbor.view(prob_t.size(0),-1)

    return neighbors


def calculate_prototypes(nodes_num, node_embeddings, labels):
    unique_classes = torch.unique(labels)
    prototypes = []
    class_to_indices = {}

    for cls in unique_classes:

        class_indices = [node for node in nodes_num if labels[node] == cls]
        class_to_indices[cls.item()] = class_indices


        class_embeddings = node_embeddings[class_indices]
        prototype = torch.mean(class_embeddings, dim=0)
        prototypes.append(prototype.unsqueeze(0))

    prototypes = torch.cat(prototypes, dim=0)
    return prototypes


def neighbor_contrastive_loss(source_embedding, target_embedding, neighbor_mask, temperature=0.5):

    source_embedding_nor = F.normalize(source_embedding, dim=1)
    target_embedding_nor = F.normalize(target_embedding, dim=1)
    similarity = F.log_softmax(torch.exp(torch.matmul(source_embedding_nor, target_embedding_nor.transpose(0, 1)) / temperature), dim=1) * -1
    neighbor_similarity = similarity * neighbor_mask.transpose(0,1)

    loss = neighbor_similarity.mean()
    return loss


def neighbor_center_contrastive_loss(source_embedding, target_embedding, unknown_index, labels, device, temperature=0.5):
    nodes_num = torch.arange(source_embedding.shape[0]).tolist()
    source_prototypes = calculate_prototypes(nodes_num, source_embedding, labels)

    if unknown_index[0].size(0) < 2:
        neighbor_centers = source_prototypes
    else:
        Kmeans = KMeans(n_clusters = 2, random_state = 0)
        Kmeans.fit(target_embedding[unknown_index].cpu().detach().numpy())
        target_centers = Kmeans.cluster_centers_
        target_centers = torch.tensor(target_centers).to(device)
        neighbor_centers = torch.cat((source_prototypes,target_centers), dim=0)

    neighbor_centers_nor  =F.normalize(neighbor_centers, dim=1)
    target_embedding_nor = F.normalize(target_embedding, dim=1)

    loss_nc = F.log_softmax(torch.exp(torch.matmul(target_embedding_nor[unknown_index], neighbor_centers_nor.transpose(0,1)) / temperature), dim=1) * -1
    loss_nc = loss_nc.mean()
    return loss_nc
