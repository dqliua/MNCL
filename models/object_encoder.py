from datapreparation.kitti360pose.utils import COLOR_NAMES
from datapreparation.kitti360pose.imports import Object3d
from models.pointcloud.pointnet2 import PointNet2
from models.language_encoder import get_mlp
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import torch_geometric.nn.conv.point_conv
from torch_geometric.nn.conv.point_conv import PointNetConv
torch_geometric.nn.conv.point_conv.PointConv = PointNetConv


class ObjectEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, known_classes: List[str], known_colors: List[str], args):
        """Module to encode a set of objects

        Args:
            embed_dim (int): Embedding dimension
            known_classes (List[str]): List of known classes, only used for embedding ablation
            known_colors (List[str]): List of known colors, only used for embedding ablation
            args: Global training arguments
        """
        super(ObjectEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.args = args

        # Set idx=0 for padding
        self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
        self.known_classes["<unk>"] = 0
        self.class_embedding = nn.Embedding(
            len(self.known_classes), embed_dim, padding_idx=0)

        self.known_colors = {c: i for i, c in enumerate(COLOR_NAMES)}
        self.known_colors["<unk>"] = 0
        self.color_embedding = nn.Embedding(
            len(self.known_colors), embed_dim, padding_idx=0)

        # OPTION: pos_encoder layers
        self.pos_encoder = get_mlp([3, 64, embed_dim])
        # OPTION: color_encoder layers
        self.color_encoder = get_mlp([3, 64, embed_dim])
        self.num_encoder = get_mlp([1, 64, embed_dim])

        self.num_mean = 1826.6844940968194
        self.num_std = 2516.8905096993817
        self.pointnet = torch.load(
            open('./checkpoints/pointnet_acc0.86_lr1_p256_model.pth', 'rb'))
        # self.pointnet = PointNet2(
        #     len(known_classes), len(known_colors), args
        # )  # The known classes are all the same now, at least for K360
        # self.pointnet.load_state_dict(torch.load(args.pointnet_path))
        # # self.pointnet_dim = self.pointnet.lin2.weight.size(0)

        if args.pointnet_freeze:
            print("CARE: freezing PN")
            self.pointnet.requires_grad_(False)

        if args.pointnet_features == 0:
            self.mlp_pointnet = get_mlp([self.pointnet.dim0, self.embed_dim])
        elif args.pointnet_features == 1:
            self.mlp_pointnet = get_mlp([self.pointnet.dim1, self.embed_dim])
        elif args.pointnet_features == 2:
            self.mlp_pointnet = get_mlp([self.pointnet.dim2, self.embed_dim])
        self.mlp_merge = get_mlp(
            [len(args.use_features) * embed_dim, embed_dim])
        self.bias = True
        self.num_heads = 4
        self.embedding_dim = embed_dim
        self.dropout = 0.1
        self.attention = nn.MultiheadAttention(
            self.embedding_dim, self.num_heads, dropout=self.dropout)
        self.linear_object = torch.nn.Linear(3, embed_dim, bias=self.bias)
        self.linear_q_object = torch.nn.Linear(
            embed_dim, embed_dim, bias=self.bias)
        self.linear_k_object = torch.nn.Linear(
            embed_dim, embed_dim, bias=self.bias)
        self.linear_v_object = torch.nn.Linear(
            embed_dim, embed_dim, bias=self.bias)

    def forward(self, objects: List[Object3d], object_points):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """
        if ("class_embed" in self.args and self.args.class_embed) or (
            "color_embed" in self.args and self.args.color_embed
        ):
            class_indices = []
            color_indices = []
            for i_batch, objects_sample in enumerate(objects):
                for obj in objects_sample:
                    class_idx = self.known_classes.get(obj.label, 0)
                    class_indices.append(class_idx)
                    color_idx = self.known_colors[obj.get_color_text()]
                    color_indices.append(color_idx)

        if "class_embed" not in self.args or self.args.class_embed == False:
            # Void all colors for ablation
            if "color" not in self.args.use_features:
                for pyg_batch in object_points:
                    pyg_batch.x[:] = 0.0  # x is color, pos is xyz

            object_features = [
                self.pointnet(pyg_batch.to(self.get_device())).features2
                for pyg_batch in object_points
            ]  # [B, obj_counts, PN_dim]
            # [total_objects, PN_dim]
            object_features = torch.cat(object_features, dim=0)
            object_features = self.mlp_pointnet(object_features)

        embeddings = []
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args and self.args.class_embed
            ):  # Use fixed embedding (ground-truth data!)
                class_embedding = self.class_embedding(
                    torch.tensor(np.array(class_indices),
                                 dtype=torch.long, device=self.get_device())
                )
                embeddings.append(F.normalize(class_embedding, dim=-1))
            else:
                pc_features = object_features
                pc_features = self.relation_enhance_object(
                    pc_features, objects)
                pc_features = F.normalize(pc_features, dim=-1)
                embeddings.append(
                    pc_features
                )

        if "color" in self.args.use_features:
            if "color_embed" in self.args and self.args.color_embed:
                color_embedding = self.color_embedding(
                    torch.tensor(np.array(color_indices),
                                 dtype=torch.long, device=self.get_device())
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))
            else:
                colors = []
                for objects_sample in objects:
                    colors.extend([obj.get_color_rgb()
                                  for obj in objects_sample])
                color_embedding = self.color_encoder(
                    torch.tensor(np.array(colors), dtype=torch.float,
                                 device=self.get_device())
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))

        if "position" in self.args.use_features:
            positions = []
            for objects_sample in objects:
                positions.extend([obj.get_center() for obj in objects_sample])
            pos_positions = torch.tensor(
                np.array(positions), dtype=torch.float, device=self.get_device())
            pos_embedding = self.pos_encoder(pos_positions)
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        if "num" in self.args.use_features:
            num_points = []
            for objects_sample in objects:
                num_points.extend([len(obj.xyz) for obj in objects_sample])
            num_points_embedding = self.num_encoder(
                (torch.tensor(np.array(num_points), dtype=torch.float, device=self.get_device(
                )).unsqueeze(-1) - self.num_mean) / self.num_std
            )
            embeddings.append(F.normalize(num_points_embedding, dim=-1))

        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]

        return embeddings, pos_positions

    @property
    def device(self):
        return next(self.class_embedding.parameters()).device

    def get_device(self):
        return next(self.class_embedding.parameters()).device

    def split_heads(self, tensor, num_heads):
        batch_size, num_of_texts, feature_dim = tensor.size()
        head_dim = feature_dim // num_heads
        output = tensor.view(batch_size, num_of_texts,
                             num_heads, head_dim).transpose(1, 2)
        return output

    def combine_heads(self, tensor, num_heads):
        batch_size, num_heads, num_of_texts, head_dim = tensor.size()
        feature_dim = num_heads * head_dim
        output = tensor.transpose(1, 2).contiguous().view(
            batch_size, num_of_texts, feature_dim)
        return output

    def relation_enhance_object(self, object_encodings, objects: List[Object3d]):
        device = self.device
        pc_features = torch.zeros(
            (len(object_encodings), object_encodings.size(-1))).to(device)
        num_of_object_list = [len(object) for object in objects]
        all_centers = torch.zeros((len(object_encodings), 3)).to(device)
        cnt_obj = 0
        for i in range(len(objects)):
            for o in objects[i]:
                all_centers[cnt_obj] = torch.tensor(o.get_center())
                cnt_obj += 1
        sum = 0
        for i in range(len(num_of_object_list)):
            num = num_of_object_list[i]
            features = object_encodings[sum:sum+num, :]
            centers = all_centers[sum:sum+num, :]
            pc_features[sum:sum+num,
                        :] = self.relation_enhance_object_sa(features, centers)
            sum += num

        return pc_features

    def relation_enhance_object_sa(self, object_encodings, centers):
        device = self.device
        object_features = object_encodings.unsqueeze(
            0).to(device)  # [B,N_OBJECT,D]
        relation = centers.unsqueeze(1)-centers.unsqueeze(0)
        relation = relation.unsqueeze(0).to(device)  # [B,N_OBJECT,N_OBJECT,3]
        row = len(relation[0])

        relation_embeded = self.linear_object(relation.reshape(-1, 3)).reshape(
            1, row, row, self.embed_dim).to(device)  # [B,N_OBJECT,N_OBJECT,3]
        relation_embeded_pooling = torch.max(relation_embeded, dim=1)[
            0].to(device)  # (B,N_OBJECT,D)

        Q_object = self.linear_q_object(object_features)
        K_object = self.linear_k_object(object_features)
        V_object = self.linear_v_object(object_features)
        attn_outputs_object, attn = self.attention(
            Q_object, K_object, V_object+relation_embeded_pooling)

        return attn_outputs_object.squeeze(0)
