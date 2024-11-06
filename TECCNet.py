# -*- coding: utf-8 -*-
# Transformer-enhanced two-stream complementary convolutional neural network
# for hyperspectral image classification

import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

class Residual(nn.Module):   #
    def __init__(self, fn):   #
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs): #
        return self.fn(x, **kwargs) + x;

class LayerNormalize(nn.Module): #
    def __init__(self,
                 dim, fn       #
                 ):
        super(LayerNormalize, self).__init__()
        self.norm = nn.LayerNorm(dim)  #
        self.fn = fn
        #
        #
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):  #
    def __init__(self,
                 dim, heads = 8, dropout = 0.1  #
                 ):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5 #

        self.to_qkv = nn.Linear(dim, dim*3, bias=True)
        self.nn1 =nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
    def forward(self, x, mask=None):   # x = (32,5,64)
        b, n, _, h = *x.shape, self.heads  # self.heads=8
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale   # A = softmax(Z)*V  中的   Z
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None: #
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper   # A = softmax(Z)*V  中的  softmax(Z)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax # A = softmax(Z)*V 的 softmax(Z)*V
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block # softmax(Z)*V 的最终结果A的变形
        out = self.nn1(out)  # self.nn1 =nn.Linear(dim, dim)
        out = self.do1(out)  # self.do1 = nn.Dropout(dropout)
        return out

class MLP_Block(nn.Module):   #
    def __init__(self, #
                 dim,hidden_dim,dropout=0.1 # dim=64  hidden_dim=8
                 ):
        super(MLP_Block, self).__init__()
        self.net =nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
      return self.net(x)  #

class Transformer(nn.Module):  #
    def __init__(self,
        dim, depth, heads, mlp_dim, dropout, # hsi: dim=h_dim,dropout=dropout,depth=h_enc_depth,heads=h_enc_heads,mlp_dim=h_enc_mlp_dim
                 ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  #
            self.layers.append(  #
         #
         #
                nn.ModuleList([
                  Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),  # Residual
                    # Attention
                  Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))       # LayerNormalize
                    # MLP_Block
                ])
            )
    def forward(self, x, mask=None):
        for attention, mlp in self.layers:  #
            x = attention(x, mask = mask)   #
            x = mlp(x) # go to MLP_Block    #
        return x   #

class ProjectInOut(nn.Module):  #
    def __init__(self,
            dim_in,dim_out,fn
                 ):
        super(ProjectInOut, self).__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in =nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()  #
        self.project_out =nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)  #
        x = self.project_out(x)
        return x

class CT_Transformer(nn.Module):
    def __init__(self,
                 depth, h_dim, l_dim, heads, dim_head, dropout  # depth=ct_atten_depth=1 h_dim=h_dim=64 l_dim=l_dim=64 heads=ct_attn_heads=8
                 # dim_head=ct_attn_dim_head=48  dropout=dropout=0.
                 ):
        super(CT_Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # depth = 1
            self.layers.append(nn.ModuleList([
                ProjectInOut(h_dim, l_dim, LayerNormalize(l_dim, CTAttention(l_dim, heads=heads, dim_head=dim_head,
                                                                             dropout=dropout))),
                ProjectInOut(l_dim, h_dim,LayerNormalize(h_dim, CTAttention(h_dim, heads=heads, dim_head=dim_head,
                                                                            dropout=dropout)))
            ]))

    def forward(self, h_tokens, l_tokens):
        (h_cls, h_patch_tokens), (l_cls, l_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (h_tokens, l_tokens))
        # h_tokens(2,5,48) l_tokens(2,5,48)
        for h_attend_lg, l_attend_h in self.layers:

            h_cls = h_attend_lg(h_cls, context=l_cls, kv_include_self=True) + h_cls  #葛老师建议
            l_cls = l_attend_h(l_cls, context=h_cls, kv_include_self=True) + l_cls

        h_tokens = torch.cat((h_cls, h_patch_tokens), dim=1)
        l_tokens = torch.cat((l_cls, l_patch_tokens), dim=1)
        return h_tokens, l_tokens

def exists(val):
    return val is not None

class FusionEncoder(nn.Module):  #
    def __init__(self,
                *, depth, h_dim, l_dim, dropout=0., h_enc_params, l_enc_params, ct_attn_depth, ct_attn_heads, ct_attn_dim_head = 64  # 64->32
                ):
        #
        super(FusionEncoder, self).__init__()

        self.layers =nn.ModuleList([])
        for _ in range(depth):  # depth =1
            self.layers.append(nn.ModuleList([    #
            Transformer(dim = h_dim, dropout = dropout, **h_enc_params),  #
            Transformer(dim = l_dim, dropout = dropout, **l_enc_params),
            CT_Transformer(h_dim=h_dim, l_dim=l_dim, depth=ct_attn_depth, heads=ct_attn_heads,
                               dim_head=ct_attn_dim_head, dropout=dropout)
               ])
            )

    def forward(self, h_tokens, l_tokens):
        for h_enc, l_enc, cross_attend in self.layers:
            h_tokens, l_tokens = h_enc(h_tokens), l_enc(l_tokens) #
            h_tokens, l_tokens = cross_attend(h_tokens, l_tokens)

        return h_tokens, l_tokens

def default(val, d):
    return val if exists(val) else d   #

class CTAttention(nn.Module):   # transformer
    def __init__(self,
          dim, heads=8, dim_head=64, dropout=0.1,  # dim = h_dim = 64
                 ):
        super(CTAttention, self).__init__()

        inner_dim = dim_head * heads  # 64*8

        self.heads = heads  # heads = 8
        self.scale = dim_head ** -0.5  # dim_head = 64

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out =nn.Sequential(
        nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context),
                                dim=1)  # cross token attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# BATCH_SIZE_TRAIN = 1
NUM_CLASS = 9 # PU-9,HongHu-16,Houston-15
Depth = 1  # the depth encoder

class HCTnet(nn.Module):
    def __init__(
            self,
            in_dim= 48,
            num_classes=NUM_CLASS,
            num_tokens=1, #4,
            dim=64,
            heads=8,
            mlp_dim=8,
            h_dim=64,
            l_dim=64,
            depth=1,
            dropout=0.1,
            emb_dropout=0.1,
            h_enc_depth=1,
            h_enc_heads=8,
            h_enc_mlp_dim=8,
            # h_enc_dim_head = 64,
            l_enc_depth=1,
            l_enc_heads=8,
            l_enc_mlp_dim=8,
            # l_enc_dim_head = 64,
            ct_attn_depth=1,
            ct_attn_heads=8,
            ct_attn_dim_head=64,
            patch_size=1
    ):
        super(HCTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim


        self.proj = nn.ConvTranspose2d(in_dim, dim, kernel_size= patch_size, stride= patch_size)   # apply for data preprocessing
        # self.proj = nn.Conv2d(in_dim, dim, kernel_size= patch_size, stride= patch_size)

        self.pos_embedding = nn.Parameter(torch.empty(1, 1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.fusion_encoder = FusionEncoder(
            depth=depth,
            h_dim=h_dim,
            l_dim=l_dim,
            ct_attn_heads=ct_attn_heads,
            ct_attn_dim_head=ct_attn_dim_head,
            ct_attn_depth=ct_attn_depth,
            h_enc_params=dict(
                depth=h_enc_depth,
                heads=h_enc_heads,
                mlp_dim=h_enc_mlp_dim,
                # dim_head = h_enc_dim_head
            ),
            l_enc_params=dict(
                depth=l_enc_depth,
                heads=l_enc_heads,
                mlp_dim=l_enc_mlp_dim,
                # dim_head = l_enc_dim_head
            ),
            dropout=dropout
        )
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        self.GB = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(64, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64, NUM_CLASS)
        )
    def forward(self, x1, x2, mask=None):  #

        cls_tokens1 = self.cls_token.expand(x1.shape[0], -1, -1) #
        x1 = self.proj(x1)  # (32, 4, 48)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')  # (32, 49, 48)
        x1 = torch.cat((cls_tokens1, x1), dim=1) # concatenated which is used to perform the classification task.
        pos_embedding = self.pos_embedding.expand(-1, x1.shape[1], -1)
        x1 += pos_embedding  # to annotate position information
        x1 = self.dropout(x1)  # dropout = 0.1

        cls_tokens2 = self.cls_token.expand(x2.shape[0], -1, -1)  #
        x2 = self.proj(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')  # (2,48, 49)
        x2 = torch.cat((cls_tokens2, x2), dim=1) # concatenated which is used to perform the classification task.
        pos_embedding = self.pos_embedding.expand(-1, x2.shape[1], -1)
        x2 += pos_embedding # to annotate position information
        x2 = self.dropout(x2)

        x1, x2 = self.fusion_encoder(x1, x2)  # x1(64,5,64)  x2(64 5 64)

        x1 = rearrange(x1, 'b h w -> b w h')
        x2 = rearrange(x2, 'b h w -> b w h')
        out_1 = self.GB(x1) # (1,64,82)--globalaveragepooling-->(1,64,82)

        out_2 = self.GB(x2)
        out = out_1 + out_2
        out = out.squeeze(-1)
        out = self.full_connection(out)
        return out

class sum_network(nn.Module):
    def __init__(self, bands, classes):
        super(sum_network, self).__init__()   #
        self.name = 'Transconv64'
        inter_bands = ((bands - 7) // 2) + 1   #
        # initial layer
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=48, kernel_size=(1, 1, 7), stride=(1, 1, 2)),
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        # spectral branch
        self.layer1_1 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=24, out_channels=12, kernel_size=(1, 1, 1),# padding=(0, 0, 3),
                               stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_3 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(0, 0, 3), kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_6 = nn.Sequential(
            nn.Conv3d(in_channels=96, out_channels=48, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer1_7 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=(1, 1, inter_bands), stride=(1, 1, 1)),
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        # spatial branch
        self.layer2_0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=48, kernel_size=(1, 1, bands), stride=(1, 1, 1)),
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_1 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=24, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=24, out_channels=12, kernel_size=(1, 1, 1), stride=(1, 1, 1)),#  padding=(1, 1, 0),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )
        self.layer2_3 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, padding=(1, 1, 0), kernel_size=(3, 3, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        self.layer2_6 = nn.Sequential(
            nn.Conv3d(in_channels=96, out_channels=48, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        self.fusionNET_1 = HCTnet(
            num_classes=NUM_CLASS,  #
            num_tokens=1,  #
            dim=64,  #
            heads=8,  # Attention head
            mlp_dim=8,  #
            h_dim=64,  #
            l_dim=64,  #
            depth=1,  # Depth of encoder,not the hsi cube depth
            dropout=0.1,
            emb_dropout=0.1,  # embed dropout:0.1
            h_enc_depth=1,  # hyperspectral encoder depth:1
            h_enc_heads=8,  # hyperspectral encoder heads:8
            h_enc_mlp_dim=8,  # hyperspectral encoder mlp dim:8
            # h_enc_dim_head = 64,
            l_enc_depth=1,  # Lidar encoder depth
            l_enc_heads=8,  # Lider encoder depth
            l_enc_mlp_dim=8,  # Lidar encoder MLP dimension
            # l_enc_dim_head = 64,
            ct_attn_depth=1,  # CTAttention depth
            ct_attn_heads=8,  # CTAttention head
            ct_attn_dim_head=64,  # CTAttention dimension head
        )

    def forward(self, x):
        x_01 = self.layer0(x)

        # spectral stream

        x_11 = self.layer1_1(x_01)
        # x_11(2,24,7,7,49)
        x_12 = self.layer1_2(x_11)

        x_13 = self.layer1_3(x_12)

        x_16 = torch.cat((x_01, x_11, x_12, x_13), dim=1)     #

        x_17 = self.layer1_6(x_16)

        x_18 = x_01 + x_17        #

        x_19 = self.layer1_7(x_18)    #
        x_19 = torch.squeeze(x_19, dim=4)     #


        del x_01, x_11, x_12, x_13, x_16, x_17, x_18#, x_19

        # spatial stream
        x_20 = self.layer2_0(x) #

        x_21 = self.layer2_1(x_20)
        # x_21(2,24,7,7,1)
        x_22 = self.layer2_2(x_21)

        x_23 = self.layer2_3(x_22)

        x_26 = torch.cat((x_20, x_21, x_22, x_23), dim=1) #

        x_27 = self.layer2_6(x_26)

        x_28 = x_20 + x_27
        x_28 = torch.squeeze(x_28, dim=4) # x_28(2,24,7,7)

        del x_20, x_21, x_22, x_23, x_26, x_27, #  x_28

        output = self.fusionNET_1(x_19, x_28)

        return output


# from torchsummary import summary
# model = sum_network(103, 9)
# summary(model, (1, 11, 11, 103), device="cpu")
# exit(0)
# from ptflops import get_model_complexity_info
# model = sum_network(103, 9)
# flops, params = get_model_complexity_info(model, (1, 9, 9, 103), as_strings=True, print_per_layer_stat=True)
# print(params)
# print(flops)
# exit(0)
# x = torch.randn(32, 1, 11, 11, 176)
# net = sum_network(176, 9)
# out = net(x)
# print(out.shape)
