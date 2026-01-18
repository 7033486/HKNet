import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50
import numpy as np

block = 32



class CrossAttentionModel(nn.Module):
    def __init__(self, d, d_ff, n_heads, n_layers):
        super().__init__()
        self.dim = d
        self.dropout1 = CustomDropout(p=0.5)
        self.dropout2 = CustomDropout(p=0.5)
        self.fc_kv = nn.Linear(d_ff, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.bn2 = nn.BatchNorm1d(d)
        self.fc_down = nn.Linear(d, d_ff)
        self.fc_up = nn.Linear(d_ff, d)
        assert d % n_heads == 0, "d must be divisible by n_heads"


    def forward(self, x):

        if x.size(1) == 2:
            batch_size, _, dim = x.shape
            x_split = x.view(batch_size, 2, 8, 2048 // 8)
            mode = 'train'
        else:
            batch_size, dim = x.shape
            x_split = x.view(batch_size, 8, 2048 // 8)
            mode = 'test'

        x_head = self.fc_kv(x_split)
        x_head = self.bn1(x_head.view(-1, dim))
        if mode == 'train':
            x_head = x_head.view(batch_size, 2, 8, -1)
        else:
            x_head = x_head.view(batch_size, 8, -1)

        x_weight = (x_head + x.unsqueeze(-2)) / 2

        weight = self.fc_up(F.relu(self.fc_down(x_weight)))
        weight = self.bn2(weight.view(-1, dim))
        if mode == 'train':
            weight = weight.view(batch_size, 2, 8, -1)
        else:
            weight = weight.view(batch_size, 8, -1)

        x_drop = self.dropout1(x_head)
        weight = self.dropout2(weight)
        output = x_head * weight + x_drop

        if mode == 'train':
            output = output.view(batch_size, 2, x_head.size(-2), -1)
        return output


def max_min_hash(x, mode):

    if mode == 'train':
        x_reshaped = x.view(x.size(0), x.size(1), x.size(2), -1, block)
        x_max = F.softmax(x_reshaped, dim=-1)
        x_min = F.softmax(-x_reshaped, dim=-1)
        x_max = x_max.view(x.size(0), x.size(1), -1)
        x_min = x_min.view(x.size(0), x.size(1), -1)
        result = torch.cat((x_max, x_min), dim=-1)
    else:
        x_reshaped = x.view(x.size(0), -1, block)
        max_indices = torch.max(x_reshaped, dim=-1)[1]
        min_indices = torch.min(x_reshaped, dim=-1)[1]
        result = torch.cat((max_indices, min_indices), dim=-1)

    return result

class tomaxmin(nn.Module):

    def __init__(self):
        super(tomaxmin, self).__init__()

    def forward(self, x, mode):
        x = max_min_hash(x, mode)
        return x

class CustomDropout(nn.Module):
    def __init__(self, p=0.2):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.p > 0:
            mask = torch.bernoulli((1 - self.p) * torch.ones_like(x)).float()
            return mask * x
        return x

class HK_layer(nn.Module):
    def __init__(self, N, c1, c2, m):
        super(HK_layer_pos, self).__init__()
        self.N = N
        self.c1 = c1
        self.c2 = c2
        self.m = m
        self.embedding = nn.Parameter(torch.randn(N, c2))
        self.to_q = nn.Linear(c1, m * c2)
        self.to_v = nn.Linear(c2, c2)
        self.fc_up = nn.Linear(c2, c2 * 8)
        self.fc_down = nn.Linear(c2 * 8, c2)

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 2:
            x = x.view(-1, self.c1)  
            batch_size = x.size(0) // 2
            is_double = True
        else:
            batch_size = x.size(0)
            is_double = False

        q = self.to_q(x).view(-1, self.m, self.c2)  
        v = self.to_v(self.embedding)
        k = self.embedding.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)
        v = v.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)
        k = k.view(-1, self.c2, self.N) 

        d_k = k.size(-2)
        attn_weights = torch.bmm(q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if is_double:
            attn_weights = attn_weights.view(batch_size, 2, -1)
        else:
            attn_weights = attn_weights.view(batch_size, -1)

        return attn_weights

class Decoder(nn.Module):
    def __init__(self, N, c1, c2, m):
        super(Decoder, self).__init__()
        self.N = N
        self.c1 = c1
        self.c2 = c2
        self.m = m
        self.mapping = nn.Linear(c1, c2)
        self.embedding = nn.Parameter(torch.randn(N, c2))
        self.to_q = nn.Linear(c2, m * c2)
        self.bn1 = nn.BatchNorm1d(c2)
        self.bn2 = nn.LayerNorm(c2)
        self.to_v = nn.Linear(c2, c2)
        
    def forward(self, x):
        
        
        if x.dim() == 3 and x.size(1) == 2:
           
            x = x.view(-1, self.c1) 
            batch_size = x.size(0) // 2
            is_double = True
        else:
            batch_size = x.size(0)
            is_double = False
        
        
        
        q = x.view(-1, self.m, self.c2)
        q = self.to_q(q)
        q = q.view(-1, self.c2)
        q = self.bn1(q)
        q = q.view(x.size(0), -1, self.c2) 
        v = self.to_v(self.embedding)
        k = self.bn2(self.embedding)
        k = k.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)
        v = v.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)
        k = k.view(-1, self.c2, self.N)  

        d_k = k.size(-2)
        attn_weights = torch.bmm(q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_applied = torch.bmm(attn_weights, v)
        x_s = []
        x_s.append(q)
        x_s.append(attn_applied)

        if is_double:
            attn_applied = attn_applied.view(batch_size, 2, -1)
            attn_weights = attn_weights.view(batch_size, 2, -1)
            
        else:
            attn_weights = attn_weights.view(batch_size, -1)
            attn_applied = attn_applied.view(batch_size, -1)
    
        keys = self.embedding   
        
        return attn_weights, x_s, attn_applied

class HK_layer(nn.Module):
    def __init__(self, N, c1, c2, m):
        super(HK_layer, self).__init__()
        self.N = N
        self.c1 = c1
        self.c2 = c2
        self.m = m
        self.mapping = nn.Linear(c1, c2)
        self.embedding = nn.Parameter(torch.randn(N, c2))
        self.to_q = nn.Linear(c2, m * c2)
        self.bn1 = nn.BatchNorm1d(c2)
        self.bn2 = nn.LayerNorm(c2)
        self.to_v = nn.Linear(c2, c2)
        self.fc_up = nn.Linear(c2, c1)
        self.fc_down = nn.Linear(c1, c2)
        
        self.linear1 = nn.Linear(c1, c2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(c2, c1)
        self.dropout = nn.Dropout(0.1)
       


    def forward(self, x):
    
        if x.dim() == 3 and x.size(1) == 2:
            x = x.view(-1, self.c1) 
            batch_size = x.size(0) // 2
            is_double = True
        else:
            batch_size = x.size(0)
            is_double = False
        q = x.view(-1, self.m, self.c2)
        q = self.to_q(q)
        q = q.view(-1, self.c2)
        q = self.bn1(q)
        q = q.view(x.size(0), -1, self.c2)
        
        v = self.to_v(self.embedding)
        k = self.bn2(self.embedding)
        k = k.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)     
        v = v.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)
        k = k.view(-1, self.c2, self.N)  

        d_k = k.size(-2)
        attn_weights = torch.bmm(q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_applied = torch.bmm(attn_weights, v)
        x_s = []
        x_s.append(q)
        attn_applied = attn_applied
        x_s.append(attn_applied)

        if is_double:
            attn_applied = attn_applied.view(batch_size, 2, -1)
            attn_weights = attn_weights.view(batch_size, 2, -1)
            
            
        else:
            attn_weights = attn_weights.view(batch_size, -1)
            attn_applied = attn_applied.view(batch_size, -1)
    
        keys = self.embedding   
        
        return attn_weights, x_s, attn_applied
        
class HK_layer_shuffle(nn.Module):
    def __init__(self, N, c1, c2, m):
        super(HK_layer_shuffle, self).__init__()
        self.N = N
        self.c1 = c1
        self.c2 = c2
        self.m = m
        self.mapping = nn.Linear(c1, c2)
        self.embedding = nn.Parameter(torch.randn(N, c2))
        self.to_q = nn.Linear(c2, m * c2)
        self.bn1 = nn.BatchNorm1d(c2)
        self.bn2 = nn.LayerNorm(c2)
        self.to_v = nn.Linear(c2, c2)
        self.fc_up = nn.Linear(c2, c1)
        self.fc_down = nn.Linear(c1, c2)
        
        self.linear1 = nn.Linear(c1, c2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(c2, c1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
              
        if x.dim() == 3 and x.size(1) == 2:        
            x = x.view(-1, self.c1) 
            batch_size = x.size(0) // 2
            is_double = True
        else:
            batch_size = x.size(0)
            is_double = False

        q = x.view(-1, self.m, self.c2)
        q = self.to_q(q)
        q = q.view(-1, self.c2)
        q = self.bn1(q)
        q = q.view(x.size(0), -1, self.c2)       
        v = self.to_v(self.embedding)
        k = self.bn2(self.embedding)
        k = k.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)             
        v = v.unsqueeze(0).repeat(batch_size * (2 if is_double else 1), 1, 1)
        k = k.view(-1, self.c2, self.N) 
        d_k = k.size(-2)
        
        attn_weights = torch.bmm(q, k) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_applied = torch.bmm(attn_weights, v)
        topk_values, topk_indices = torch.topk(attn_weights, 40, dim=-1)
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(-1, topk_indices, topk_values)
        attn_weights = mask
              
        if not is_double:
            topk_indices = topk_indices.view(batch_size, -1)
            avg_topk_indices = topk_indices.float().mean(dim=-1)
            quantized_value = torch.floor(avg_topk_indices / (self.N / 4)).long()
            random_seeds = quantized_value % 4  
        
            permutation_matrix = []
            for seed in random_seeds:
                torch.manual_seed(seed.item())
                perm = torch.randperm(attn_weights.size(-1), dtype=torch.int64)  
                permutation_matrix.append(perm)
            permutation_matrix = torch.stack(permutation_matrix, dim=0)
            permutation_matrix = permutation_matrix.cuda()
            attn_weights = torch.gather(attn_weights, dim=-1, index=permutation_matrix.unsqueeze(1).expand(-1, attn_weights.size(1), -1))

        
        x_s = []
        x_s.append(q)
        attn_applied = attn_applied 
        # + q
        x_s.append(attn_applied)

        if is_double:
            attn_applied = attn_applied.view(batch_size, 2, -1)
            attn_weights = attn_weights.view(batch_size, 2, -1)
            
            
        else:
            attn_weights = attn_weights.view(batch_size, -1)
            attn_applied = attn_applied.view(batch_size, -1)
    
        keys = self.embedding   
        
        return attn_weights, x_s, attn_applied
        
    def positional_encoding(self, H, W, d, device):
        """Generate a positional encoding matrix of shape [1, 1, H, W, d] using cosine functions."""
        y_position = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_position = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)
        
        i = torch.arange(d, device=device)
        div_term = 10000 ** (2 * (i // 2) / d)
        
        pos_y = y_position[:, :, None] / div_term
        pos_x = x_position[:, :, None] / div_term
        
        pos_y = torch.sin(pos_y)[:, :, 0::2] + torch.cos(pos_y)[:, :, 1::2]
        pos_x = torch.sin(pos_x)[:, :, 0::2] + torch.cos(pos_x)[:, :, 1::2]
        
        pos = torch.zeros((H, W, d), device=device)
        pos[:, :, 0::2] = pos_y
        pos[:, :, 1::2] = pos_x
        
        pos = pos.unsqueeze(0)  
        return pos

class res50_cls_hash(nn.Module):
    """
        EMBEDDING_SIZE: It is the dimension of the embedding feature vector (i.e. 1024).
        CLASS_SIZE: Total number of classes in training and validation (i.e. 100),
                    it is compulsary if the ONLY_EMBEDDINGS is FALSE.
        PRETRAINED: TRUE or FALSE to flag the Imagenet pretrained model.
        ONLY_EMBEDDINGS: if TRUE, dimensions of both training/validation and test outputs are equal to EMBEDDING SIZE.
                         if FALSE, training/validation outputs = CLASS_SIZE while test outputs = EMBEDDING SIZE.
                         TRUE is necessary if custom penalty functions are utilized such as AAMP, LMCP
                         FALSE is necessary if standard softmax is preferred for training.
        L2_NORMED: The output of test pairs are normalized if it is TRUE. For AAMP, LMCP (Margin losses) it should be TRUE

    """

    def __init__(self, class_size=250, pretrained=True, only_embeddings=True, l2_normed=True):
        super(res50_cls_hash, self).__init__()
        self.only_embeddings = only_embeddings
        self.l2_normed = l2_normed
        self.model = resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, class_size),
            nn.BatchNorm1d(class_size)
        )

    def forward(self, x, train=True):
        b = x.size(0)
        if train:
            x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        out = self.features(x)
        # out = F.relu(out, inplace=True)
        out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        # print(out.shape)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        # out = self.model.classifier(out)
        # out = F.relu(out)
        if train:
            out = out.view(b, 2, -1)

        return out






