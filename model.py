import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b 1 n l -> b n l')
        # [batch_size, num_patches+1, inner_dim*3] --> ([batch_size, num_patches+1, inner_dim], -->(q,k,v)
        #                                               [batch_size, num_patches+1, inner_dim],
        #                                               [batch_size, num_patches+1, inner_dim])
        #将x转变为qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对tensor进行分块

        q, k, v = \
            map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b n l -> b 1 n l')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, ResNetBlock()),
                # PreNorm(dim//(2**i), ConvAttention(dim//(2**i), heads, dim_head//(2**i), dropout)),
                # PreNorm(dim//(2**(i+1)), FeedForward(dim//(2**(i+1)), mlp_dim, dropout))
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class TimeTransformer(nn.Module):
    def __init__(self, *, input_dim,  num_patches=16, dim, depth, heads, mlp_dim,
                 pool='cls', channels=1, dim_head, emb_dropout=0., dropout=0.):
        super(TimeTransformer, self).__init__()

        # self.to_patch_embedding = Embedding(input_dim, dim)
        self.to_patch_embedding = self.to_patch_embedding = nn.Sequential(
            Rearrange('b 1 (n d) -> b 1 n d', n=num_patches),
            nn.Linear(input_dim//num_patches, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))  # [1, 1, 1, dim] 随机数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # [1, num_patches+1, dim] 随机数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()  # 这个恒等函数，就如同名字占位符，并没有实际操作

    def forward(self, rawdata):
        TimeSignals = rawdata   # Get Time Domain Signals
        TimeSignals = rearrange(TimeSignals, 'b l -> b 1 l')
        # print(TimeSignals.shape, rawdata.shape)

        x = self.to_patch_embedding(TimeSignals)
        b, _, n, _ = x.shape      # x: [batch_size, channels, num_patches, dim]

        cls_tokens = repeat(self.cls_token, '() c n d -> b c n d', b=b)  # cls_tokens: [batch_size, c, num_patches, dim]
        x = torch.cat((cls_tokens, x), dim=2)  # x: [batch_size, c, num_patches+1, dim]
        # print(x.shape)
        #x += self.pos_embedding[:, :(n + 1)]  # 添加位置编码：x: [batch_size, c, num_patches+1, dim]
        x = self.dropout(x)

        x = self.transformer(x)     # x: [batch_size, c, num_patches+1, dim]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, :, 0]  # x: [batch_size, c, 1, dim]
        x = self.to_latent(x)
        return x


class DSCTransformer(nn.Module):
    def __init__(self, *, input_dim, dim, depth, heads, mlp_dim, pool='cls',
                 num_classes, channels=1, dim_head, emb_dropout=0., dropout=0.):
        super(DSCTransformer, self).__init__()
        self.in_dim_time = input_dim
        self.TimeTrans = TimeTransformer(input_dim=self.in_dim_time, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                         pool=pool, dim_head=dim_head, emb_dropout=emb_dropout, dropout=dropout)

        self.mlp_head = nn.Sequential(
            Rearrange('b c l -> b (l c)'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        TimeSignals = x
        TimeFeature = self.TimeTrans(TimeSignals)
        y = self.mlp_head(TimeFeature)

        return y  # [batch_size, 1, num_classes]
