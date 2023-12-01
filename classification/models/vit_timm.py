import timm
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, DropPath
from timm.models.helpers import named_apply


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class PatchEmbedConv(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, norm_layer=None, stem_channel=32):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size_origin = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size_origin
        self.grid_size = (img_size[0] // patch_size_origin[0], img_size[1] // patch_size_origin[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.stem_conv1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.proj = nn.Conv2d(stem_channel, embed_dim, kernel_size=7, stride=2, padding=3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)
        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)
        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bn=0):
        super().__init__()
        self.bn = bn
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # IRFFN
        B, N, C = x.shape
        bottleneck = x[:, -self.bn:, :]
        x = x[:, :-self.bn, :] if self.bn != 0 else x
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = torch.cat([x, bottleneck], dim=1) if self.bn != 0 else x
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1, bn=0):
        super().__init__()
        self.bn = bn
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            # adapt to cross phase tokens
            bottleneck = x[:, -self.bn:, :]
            x = x[:, :-self.bn, :] if self.bn != 0 else x
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = torch.cat([x_, bottleneck], dim=1) if self.bn != 0 else x_
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1, bn=0, dwflag=True):
        super().__init__()
        self.bn = bn
        self.dwflag = dwflag
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio, bn=bn)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bn=bn)
        if dwflag:
            self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W, relative_pos):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # a part of conv down sampler
        if self.dwflag:
            bottleneck = x[:, -self.bn:, :]
            x = x[:, :-self.bn, :] if self.bn != 0 else x
            B, N, C = x.shape
            cnn_feat = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat
            x = x.flatten(2).permute(0, 2, 1).contiguous()
            x = torch.cat([x, bottleneck], dim=1) if self.bn != 0 else x
        return x


class MultiPhaseVisionTransformer(VisionTransformer):

    def __init__(self,
                 use_bottleneck=True,
                 bottleneck_n=4,  # cross phase tokens number
                 phase_num=4,  # phase number
                 fusion_layer=8,  # fusion layer
                 depth=13,  # depth of the last 2 stages
                 drop_rate=0.,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.2,
                 num_classes=7,
                 num_heads=8,
                 img_size=224,
                 patch_size=16,
                 embed_dim=512,
                 in_chans=1,
                 pre_norm=False,
                 *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(depth=depth, drop_rate=drop_rate, num_classes=num_classes, num_heads=num_heads, img_size=img_size,
                         patch_size=patch_size, embed_dim=embed_dim, in_chans=in_chans, *args, **kwargs, weight_init='')
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.bottleneck_n = bottleneck_n
        self.phase_num = phase_num
        self.fusion_layer = fusion_layer
        self.depth = depth
        self.in_chans = in_chans
        self.img_size = img_size
        sr_ratios = [8, 4, 2, 1]

        # init the patch embedding layer, which includes the conv encoder
        del self.patch_embed
        self.patch_embeds = []
        for i in range(phase_num):
            self.patch_embeds.append(PatchEmbedConv(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=64, norm_layer=norm_layer))
        self.patch_embeds = nn.ModuleList(self.patch_embeds)

        self.temporal_dims = [0]*phase_num

        # transliver does not use absolute position embedding and class token
        del self.pos_embed
        del self.cls_token

        self.norm_pres = []
        for i in range(phase_num):
            self.norm_pres.append(norm_layer(self.embed_dim) if pre_norm else nn.Identity())
        self.norm_pres = nn.ModuleList(self.norm_pres)

        # init cross phase tokens
        if use_bottleneck:
            # remember tile (bs,1,1) when forward
            self.bottleneck = nn.Parameter(torch.Tensor(1, bottleneck_n, self.embed_dim))
        else:
            self.bottleneck = None

        # init conv downsampler
        self.down_sample_a = []
        self.down_sample_b = []
        self.down_sample_c = []
        for i in range(phase_num):
            self.down_sample_a.append(PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=64, embed_dim=128, norm_layer=norm_layer))
            self.down_sample_b.append(PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=128, embed_dim=256, norm_layer=norm_layer))
            self.down_sample_c.append(PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=256, embed_dim=embed_dim, norm_layer=norm_layer))
        self.down_sample_a = nn.ModuleList(self.down_sample_a)
        self.down_sample_b = nn.ModuleList(self.down_sample_b)
        self.down_sample_c = nn.ModuleList(self.down_sample_c)

        # init relative position embedding in each stage
        self.relative_pos_a = []
        self.relative_pos_b = []
        self.relative_pos_c = []
        self.relative_pos_d = []
        for i in range(phase_num):
            self.relative_pos_a.append(nn.Parameter(torch.randn(
                1, self.patch_embeds[0].num_patches*16,
                self.patch_embeds[0].num_patches*16 // sr_ratios[0] // sr_ratios[0])))
            self.relative_pos_b.append(nn.Parameter(torch.randn(
                2, self.patch_embeds[0].num_patches*4,
                self.patch_embeds[0].num_patches*4 // sr_ratios[1] // sr_ratios[1])))
            self.relative_pos_c.append(nn.Parameter(torch.randn(
                4, self.patch_embeds[0].num_patches,
                self.patch_embeds[0].num_patches // sr_ratios[2] // sr_ratios[2])))
            # only use cross phase tokens in the last stage
            self.relative_pos_d.append(nn.Parameter(torch.randn(
                8, self.patch_embeds[0].num_patches // 4 + bottleneck_n,
                self.patch_embeds[0].num_patches // 4 // sr_ratios[3] // sr_ratios[3] + bottleneck_n)))
        self.relative_pos_a = nn.ParameterList(self.relative_pos_a)
        self.relative_pos_b = nn.ParameterList(self.relative_pos_b)
        self.relative_pos_c = nn.ParameterList(self.relative_pos_c)
        self.relative_pos_d = nn.ParameterList(self.relative_pos_d)

        # init transformer blocks of the first 2 stages
        self.depth_b = 2  # depth of stage a and b
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth+2*self.depth_b)]  # stochastic depth decay rule
        self.blocks_a = []
        self.blocks_b = []
        for l in range(self.depth_b):
            self.blocks_a.append([])
            self.blocks_b.append([])
            for i in range(phase_num):
                self.blocks_a[l].append(Block(dim=64, num_heads=1, norm_layer=norm_layer,
                                                  drop=drop_rate, qkv_bias=True, attn_drop=attn_drop_rate,
                                                  drop_path=dpr[l], sr_ratio=sr_ratios[0]))
                self.blocks_b[l].append(Block(dim=128, num_heads=2, norm_layer=norm_layer,
                                                  drop=drop_rate, qkv_bias=True, attn_drop=attn_drop_rate,
                                                  drop_path=dpr[l+self.depth_b], sr_ratio=sr_ratios[1]))
            self.blocks_a[l] = nn.ModuleList(self.blocks_a[l])
            self.blocks_b[l] = nn.ModuleList(self.blocks_b[l])
        self.blocks_a = nn.ModuleList(self.blocks_a)
        self.blocks_b = nn.ModuleList(self.blocks_b)

        # init transformer blocks of the last 2 stages
        del self.blocks
        self.blocks = []
        for l in range(depth):
            self.blocks.append([])
            for i in range(phase_num):
                if l < fusion_layer:
                    self.blocks[l].append(Block(dim=256, num_heads=4, norm_layer=norm_layer,
                                                drop=drop_rate, qkv_bias=True, attn_drop=attn_drop_rate,
                                                drop_path=dpr[l + 2 * self.depth_b], sr_ratio=sr_ratios[2]))
                else:
                    self.blocks[l].append(Block(dim=self.embed_dim, num_heads=8, norm_layer=norm_layer,
                                                drop=drop_rate, qkv_bias=True, attn_drop=attn_drop_rate,
                                                drop_path=dpr[l + 2 * self.depth_b], sr_ratio=sr_ratios[3],
                                                bn=bottleneck_n, dwflag=False))
            self.blocks[l] = nn.ModuleList(self.blocks[l])
        self.blocks = nn.ModuleList(self.blocks)

        # 2 fc
        del self.head
        fc_mid_dim = 64
        self.head0 = nn.Linear(self.embed_dim, fc_mid_dim)
        self.head1 = nn.Linear(fc_mid_dim, num_classes)

        self.init_weights_custom()

    def init_weights_custom(self):
        if self.bottleneck is not None:
            nn.init.normal_(self.bottleneck, std=.02)
        named_apply(init_weights_vit_timm, self)

    def forward_features(self, x):
        # patch embedding
        for i in range(self.phase_num):
            x[i] = self.patch_embeds[i](x[i])

        # stage a
        for l in range(self.depth_b):
            for i in range(self.phase_num):
                x[i] = self.blocks_a[l][i](x[i], self.img_size//4, self.img_size//4, self.relative_pos_a[i])
        # downsample
        for i in range(self.phase_num):
            x[i] = x[i].reshape(x[i].shape[0], self.img_size//4, self.img_size//4, -1).permute(0, 3, 1, 2).contiguous()
            x[i] = self.down_sample_a[i](x[i])

        # stage b
        for l in range(self.depth_b):
            for i in range(self.phase_num):
                x[i] = self.blocks_b[l][i](x[i], self.img_size//8, self.img_size//8, self.relative_pos_b[i])
        # downsample
        for i in range(self.phase_num):
            x[i] = x[i].reshape(x[i].shape[0], self.img_size//8, self.img_size//8, -1).permute(0, 3, 1, 2).contiguous()
            x[i] = self.down_sample_b[i](x[i])

        # cross phase tokens
        if self.bottleneck is not None:
            batch_bottleneck = self.bottleneck.expand(x[0].shape[0], -1, -1)
        else:
            batch_bottleneck = None

        for i in range(self.phase_num):
            x[i] = self.norm_pres[i](x[i])

        for l in range(self.depth):
            # stage c
            if l < self.fusion_layer:
                for i in range(self.phase_num):
                    x[i] = self.blocks[l][i](x[i], self.img_size//16, self.img_size//16, self.relative_pos_c[i])
                    if l == self.fusion_layer-1:
                        x[i] = x[i].reshape(x[i].shape[0], self.img_size // 16, self.img_size // 16, -1).permute(0, 3, 1, 2).contiguous()
                        x[i] = self.down_sample_c[i](x[i])
                        self.temporal_dims[i] = x[i].shape[1]
            # stage d
            else:
                bottle = []
                for i in range(self.phase_num):
                    t_mod = x[i].shape[1]
                    in_mod = torch.cat([x[i], batch_bottleneck], dim=1)
                    out_mod = self.blocks[l][i](in_mod, self.img_size // 32, self.img_size // 32,
                                                self.relative_pos_d[i])
                    x[i] = out_mod[:, :t_mod, ...]
                    bottle.append(out_mod[:, t_mod:, ...])
                batch_bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)

        x_out = torch.cat(x, dim=1)
        encoded = self.norm(x_out)

        return encoded

    def forward_head(self, x, pre_logits: bool = False):
        x_out = []
        counter = 0
        # global average for each phase
        for i in range(self.phase_num):
            x_out.append(x[:, counter:counter+self.temporal_dims[i], ...].mean(dim=1))
            counter += self.temporal_dims[i]

        if pre_logits:
            return x_out
        for i in range(self.phase_num):
            # 2 fc
            x_out[i] = self.head0(x_out[i])
            x_out[i] = self.head1(x_out[i])
        x_pool = torch.zeros_like(x_out[0])
        for i in range(self.phase_num):
            x_pool += x_out[i]
        return x_pool / len(x_out)

    def forward(self, x):
        xs = []
        for i in range(self.phase_num):
            xs.append(x[:, i, ...])
        x = xs
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def mbt_base_phase4_bottleneck4_vit(pretrained=False, pretrain_path=None, pretrained_cfg=None, **kwargs):
    def find_keys(d, s):
        keys = []
        for key in d:
            if s in key:
                keys.append(key)
        return keys

    def find_phase_position(s, dots=1):
        cnt = 0
        for ind in range(len(s)):
            if s[ind] == '.':
                if cnt == dots:
                    return ind+1
                else:
                    cnt += 1
        return len(s)

    phase_num = 4
    bottleneck_n = 4
    use_bottleneck = True
    fusion_layer = 10
    model = MultiPhaseVisionTransformer(phase_num=phase_num, bottleneck_n=bottleneck_n, use_bottleneck=use_bottleneck,
                                        fusion_layer=fusion_layer, **kwargs)

    # load the pretrain model into the new model
    if pretrained:
        pre_model = timm.create_model("vit_small_patch16_224", pretrained=True, **kwargs)
        model_dict = model.state_dict()
        new_dict = {}
        pre_dict = pre_model.state_dict()
        pre_dict_cmt = torch.load(pretrain_path)["model"]  # may be changed for your pretrain model path
        para_dict = {}
        for k in pre_dict:
            if k in model_dict and model_dict[k].shape == pre_dict[k].shape:
                para_dict[k] = k
        for k in pre_dict_cmt:
            if "stem_conv" in k or "stem_norm" in k:
                for mk in find_keys(model_dict, k):
                    if model_dict[mk].shape == pre_dict_cmt[k].shape:
                        new_dict[mk] = pre_dict_cmt[k]
                        para_dict[mk] = k
                    else:
                        new_dict[mk] = torch.sum(pre_dict_cmt[k], dim=1).unsqueeze(1)
                        para_dict[mk] = k
            elif "patch_embed" in k:
                for mk in find_keys(model_dict, "down_sample")+find_keys(model_dict, "patch_embed"):
                    if mk.split('.')[2] == k.split('.')[1] and mk.split('.')[3] == k.split('.')[2]:
                        if model_dict[mk].shape == pre_dict_cmt[k].shape:
                            new_dict[mk] = pre_dict_cmt[k]
                            para_dict[mk] = k
            elif "block" in k:
                for i in range(phase_num):
                    if "blocks_c" in k:
                        nk = k.replace("blocks_c", "blocks")
                    elif "blocks_d" in k:
                        nk = k.replace("blocks_d", "blocks")
                        l = int(k.split('.')[1])
                        nl = l + fusion_layer
                        nk = nk.replace(str(l), str(nl), 1)
                    else:
                        nk = k
                    pos = find_phase_position(nk)
                    mk = nk[0:pos] + str(i) + "." + nk[pos:]
                    if len(find_keys(model_dict, mk)) > 0:
                        if model_dict[mk].shape == pre_dict_cmt[k].shape:
                            new_dict[mk] = pre_dict_cmt[k]
                            para_dict[mk] = k
            elif "relative_pos" in k:
                for mk in find_keys(model_dict, k):
                    if model_dict[mk].shape == pre_dict_cmt[k].shape:
                        new_dict[mk] = pre_dict_cmt[k]
                        para_dict[mk] = k

        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

    return model


def create_mbt(model_name, pretrained, pretrain_path, **kwargs):
    return timm.create_model(model_name, pretrained=pretrained, pretrain_path=pretrain_path, **kwargs)
