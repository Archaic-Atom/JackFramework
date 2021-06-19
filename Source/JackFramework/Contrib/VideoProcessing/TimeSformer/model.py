# -*- coding: utf-8 -*-
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


class PreNorm(nn.Module):
    # classes
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    # feedforward
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    return einsum('b i j, b j d -> b i d', attn, v)


class Attention(nn.Module):
    # attention
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims),
                         (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # combine heads out
        return self.to_out(out)


class TimeSformer(nn.Module):
    # main classes
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_size=224,
        patch_size=16,
        channels=3,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_emb = nn.Embedding(num_positions + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, \
            f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        tokens = self.to_patch_embedding(video)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, tokens), dim=1)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        cls_token = x[:, 0]
        return self.to_out(cls_token)


class TimeSformer_v2(nn.Module):
    # main classes
    def __init__(
        self,
        *,
        in_channels,
        bottleneck_channels,
        num_frames,
        image_height,
        image_width,
        patch_size=16,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()
        assert image_height % patch_size == 0 and image_width % patch_size == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_size) * (image_width // patch_size)
        num_positions = num_frames * num_patches
        patch_dim = in_channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, bottleneck_channels)
        self.pos_emb = nn.Embedding(num_positions + 1, bottleneck_channels)
        self.cls_token = nn.Parameter(torch.randn(1, bottleneck_channels))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(bottleneck_channels, Attention(bottleneck_channels,
                                                       dim_head=dim_head,
                                                       heads=heads, dropout=attn_dropout)),
                PreNorm(bottleneck_channels, Attention(bottleneck_channels,
                                                       dim_head=dim_head,
                                                       heads=heads, dropout=attn_dropout)),
                PreNorm(bottleneck_channels, FeedForward(bottleneck_channels, dropout=ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(bottleneck_channels),
            nn.Linear(bottleneck_channels, patch_dim)
            #nn.Linear(bottleneck_channels, patch_size*patch_size)
        )

    def forward(self, video):
        video = rearrange(video, 'b c f h w  -> b f c h w')
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, \
            f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        tokens = self.to_patch_embedding(video)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, tokens), dim=1)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        #cls_token = x[:, 0]
        x = self.to_out(x)
        x = x[:, 1:, :]  # remove cls_token
        x = rearrange(x, 'b (f h w) (p1 p2 c) -> b f c (h p1) (w p2)', p1=p, p2=p,
                      f=f, h=h // p, w=w // p)
        x = rearrange(x, 'b f c h w -> b c f h w')
        return x


class TimeSformer_v3(nn.Module):
    # main classes
    def __init__(
        self,
        *,
        in_channels,
        bottleneck_channels,
        num_frames,
        image_height,
        image_width,
        patch_size=16,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()
        assert image_height % patch_size == 0 and image_width % patch_size == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_size) * (image_width // patch_size)
        num_positions = num_frames * num_patches
        patch_dim = in_channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, bottleneck_channels)
        self.pos_emb = nn.Embedding(num_positions + 1, bottleneck_channels)
        self.cls_token = nn.Parameter(torch.randn(1, bottleneck_channels))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(bottleneck_channels, Attention(bottleneck_channels,
                                                       dim_head=dim_head,
                                                       heads=heads, dropout=attn_dropout)),
                PreNorm(bottleneck_channels, Attention(bottleneck_channels,
                                                       dim_head=dim_head,
                                                       heads=heads, dropout=attn_dropout)),
                PreNorm(bottleneck_channels, FeedForward(bottleneck_channels, dropout=ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(bottleneck_channels),
            #nn.Linear(bottleneck_channels, patch_dim)
            nn.Linear(bottleneck_channels, patch_size * patch_size)
        )

    def forward(self, x):
        x = rearrange(x, 'b c f h w  -> b f c h w')
        b, f, _, h, w, *_, device, p = *x.shape, x.device, self.patch_size
        assert h % p == 0 and w % p == 0, \
            f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        x = rearrange(x, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        x = self.to_patch_embedding(x)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        #cls_token = x[:, 0]
        x = self.to_out(x)
        x = x[:, 1:, :]  # remove cls_token
        x = rearrange(x, 'b (f h w) (p1 p2) -> b f 1 (h p1) (w p2)', p1=p, p2=p,
                      f=f, h=h // p, w=w // p)
        x = rearrange(x, 'b f c h w -> b c f h w')
        return x


class TimeSformer_v4(nn.Module):
    # main classes
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        bottleneck_channels,
        num_frames,
        image_height,
        image_width,
        patch_size=16,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()
        assert image_height % patch_size == 0 and image_width % patch_size == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_size) * (image_width // patch_size)
        num_positions = num_frames * num_patches
        patch_dim = in_channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, bottleneck_channels)
        self.pos_emb = nn.Embedding(num_positions + 1, bottleneck_channels)
        self.cls_token = nn.Parameter(torch.randn(1, bottleneck_channels))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(bottleneck_channels, Attention(bottleneck_channels,
                                                       dim_head=dim_head,
                                                       heads=heads, dropout=attn_dropout)),
                PreNorm(bottleneck_channels, Attention(bottleneck_channels,
                                                       dim_head=dim_head,
                                                       heads=heads, dropout=attn_dropout)),
                PreNorm(bottleneck_channels, FeedForward(bottleneck_channels, dropout=ff_dropout))
            ]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(bottleneck_channels),
            nn.Linear(bottleneck_channels, out_channels * patch_size ** 2)
            #nn.Linear(bottleneck_channels, patch_size*patch_size)
        )

    def forward(self, video):
        video = rearrange(video, 'b c f h w  -> b f c h w')
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, \
            f'height {h} and width {w} of video must be divisible by the patch size {p}'

        n = (h // p) * (w // p)

        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        tokens = self.to_patch_embedding(video)

        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, tokens), dim=1)
        x += self.pos_emb(torch.arange(x.shape[1], device=device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f) + x
            x = ff(x) + x

        #cls_token = x[:, 0]
        x = self.to_out(x)
        x = x[:, 1:, :]  # remove cls_token
        x = rearrange(x, 'b (f h w) (p1 p2 c) -> b f c (h p1) (w p2)', p1=p, p2=p,
                      f=f, h=h // p, w=w // p)
        x = rearrange(x, 'b f c h w -> b c f h w')
        return x


def debug_main():
    model = TimeSformer_v3(
        bottleneck_channels=256, image_height=64, image_width=128,
        num_frames=64, in_channels=96, patch_size=8, depth=11,
        heads=8, dim_head=16, attn_dropout=0, ff_dropout=0)
    video = torch.randn(2, 96, 64, 64, 128)
    print(video.size())
    # print(model)
    pred = model(video)  # (2, 10)

    import time

    for _ in range(2):
        start_time = time.time()
        pred = model(video)
        duration = time.time() - start_time
        print(duration)

    print(pred.size())
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)


if __name__ == '__main__':
    debug_main()
