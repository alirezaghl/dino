import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import argparse

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            stride=patch_size,
            kernel_size=patch_size,
            bias=True
        )
    
    def forward(self, x):
        x = self.proj(x)
        out = rearrange(x, 'b e h w -> b (h w) e')
        return out

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.dropout_p = dropout
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        out = rearrange(attn, 'b h n d -> b n (h d)')
        out = self.dropout(self.proj(out))
        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_heads, dropout, depth):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(embed_dim, num_heads, dropout),
                    MLP(embed_dim, mlp_dim, dropout)
                ])
            )
    
    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return self.norm(x)

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embedding_dim,
                 mlp_dim, num_heads, dropout, depth, avg='cls'):
        super(VisionTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.pe = PatchEmbedding(img_size, patch_size, in_channels, embedding_dim)
        self.num_patches = self.pe.num_patches 
        self.avg = avg
        
        self.encoder = Encoder(embedding_dim, mlp_dim, num_heads, dropout, depth)
        
        if self.avg == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.pe(x)
        
        if self.avg == 'cls':
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding
            x = self.dropout(x)
            x = self.encoder(x)
            x = x[:, 0]  
        else:
            x = x + self.pos_embedding
            x = self.dropout(x)
            x = self.encoder(x)
            x = x.mean(dim=1)  
        return x

class DinoHead(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, bottleneck_dim, out_dim, norm_last_layer=False):
        super(DinoHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DinoVIT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embedding_dim, num_heads, 
                 depth, mlp_dim, hidden_dim, bottleneck_dim, dropout, out_dim):
        super().__init__()
        
        self.backbone = VisionTransformer(
            img_size, patch_size, in_channels, embedding_dim,
            mlp_dim, num_heads, dropout, depth
        )
        
        self.head = DinoHead(
            embedding_dim,
            hidden_dim,
            bottleneck_dim,
            out_dim
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_size', type=int, default=32)  
#     parser.add_argument('--patch_size', type=int, default=4)  
#     parser.add_argument('--in_channels', type=int, default=3)
#     parser.add_argument('--embedding_dim', type=int, default=192)
#     parser.add_argument('--mlp_dim', type=int, default=768)
#     parser.add_argument('--num_heads', type=int, default=3)
#     parser.add_argument('--dropout', type=float, default=0.0)
#     parser.add_argument('--depth', type=int, default=6)
#     parser.add_argument('--batch_size', type=int, default=4)
#     parser.add_argument('--avg', type=str, default='cls')  
#     parser.add_argument('--bottleneck_dim', type=int, default=256)
#     parser.add_argument('--hidden_dim', type=int, default=2048)
#     parser.add_argument('--out_dim', type=int, default=8192)
    
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
    
#     x = torch.rand(args.batch_size, args.in_channels, args.img_size, args.img_size)

#     model = DinoVIT(
#         args.img_size,
#         args.patch_size,
#         args.in_channels,
#         args.embedding_dim,
#         args.num_heads,
#         args.depth,
#         args.mlp_dim,
#         args.hidden_dim,
#         args.bottleneck_dim,
#         args.dropout,
#         args.out_dim,
#     )

#     dino_out = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Dino output: {dino_out.shape}")
#     print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")