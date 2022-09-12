import torch.nn.functional
from functools import partial
from torchvision import models
from models.TransformerBaseNetworks import *
from models.base_model import ASPP_v1, OverlapPatchEmbed, EdgeAddSementic, SementicAddEdge, BilinearUp, Block, EdgeFusion
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class Encoder(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1], edge_channel=[32, 64, 128, 256]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.edge_channel = edge_channel

        # resnet34
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        # Transformer
        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # CNN
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # feature aggregator
        self.FA1 = nn.Conv2d(self.embed_dims[0] * 2, self.embed_dims[0], kernel_size=1, stride=1, padding=0)
        self.FA2 = nn.Conv2d(self.embed_dims[1] * 2, self.embed_dims[1], kernel_size=1, stride=1, padding=0)
        self.FA3 = nn.Conv2d(self.embed_dims[2] * 2, self.embed_dims[2], kernel_size=1, stride=1, padding=0)
        self.FA4 = nn.Conv2d(self.embed_dims[3] * 2, self.embed_dims[3], kernel_size=1, stride=1, padding=0)

        self.aspp = ASPP_v1(self.embed_dims[3])
        self.leakyRelu = nn.LeakyReLU(inplace=True)

        self.convDown = nn.Conv2d(self.edge_channel[0], self.embed_dims[0], kernel_size=3, stride=2, padding=1)
        self.convDown1 = nn.Conv2d(self.embed_dims[0], self.embed_dims[1], kernel_size=3, stride=2, padding=1)
        self.convDown2 = nn.Conv2d(self.embed_dims[1], self.embed_dims[2], kernel_size=3, stride=2, padding=1)
        self.convDown3 = nn.Conv2d(self.embed_dims[2], self.embed_dims[3], kernel_size=3, stride=2, padding=1)

        self.conv = nn.Conv2d(self.edge_channel[0], self.embed_dims[0], kernel_size=3, stride=1, padding=1)
        self.decoder4 = BilinearUp(self.embed_dims[3], self.embed_dims[2])
        self.decoder3 = BilinearUp(self.embed_dims[2], self.embed_dims[1])
        self.decoder2 = BilinearUp(self.embed_dims[1], self.embed_dims[0])
        self.decoder1 = BilinearUp(self.embed_dims[0], self.embed_dims[0])

        # ESA
        self.esa1 = EdgeAddSementic(sem_channel=self.embed_dims[3], edge_channel=self.edge_channel[2], out_channel=self.edge_channel[2], stride=4)
        self.esa2 = EdgeAddSementic(sem_channel=self.edge_channel[2], edge_channel=self.edge_channel[1], out_channel=self.edge_channel[1], stride=2)
        self.esa3 = EdgeAddSementic(sem_channel=self.edge_channel[1], edge_channel=self.edge_channel[1], out_channel=self.edge_channel[0], stride=2)

        # EEM
        self.eem1 = SementicAddEdge(sem_channel=self.embed_dims[0], edge_channel=self.edge_channel[0], out_channel=self.edge_channel[0], stride=1)
        self.eem2 = SementicAddEdge(sem_channel=self.embed_dims[0], edge_channel=self.edge_channel[1], out_channel=self.edge_channel[1], stride=1)
        self.eem3 = SementicAddEdge(sem_channel=self.embed_dims[1], edge_channel=self.edge_channel[2], out_channel=self.edge_channel[2], stride=1)
        self.eem4 = SementicAddEdge(sem_channel=self.embed_dims[2], edge_channel=self.edge_channel[3], out_channel=self.edge_channel[3], stride=1)
        self.eem5 = SementicAddEdge(sem_channel=self.embed_dims[3], edge_channel=self.edge_channel[0], out_channel=self.edge_channel[0], stride=16)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        x_tr = x
        x_cnn = x
        B = x.shape[0]

        x_cnn = self.firstconv(x_cnn)
        x_cnn = self.firstbn(x_cnn)
        x_cnn_edge = self.firstrelu(x_cnn)
        x_cnn = self.firstmaxpool(x_cnn_edge)

        # stage 1
        x_tr1, H1, W1 = self.patch_embed1(x_tr)
        for i, blk in enumerate(self.block1):
            x_tr1 = blk(x_tr1, H1, W1)
        x_tr1 = self.norm1(x_tr1)
        x_tr1 = x_tr1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        x_cnn1 = self.encoder1(x_cnn)
        x_cat1 = self.FA1(torch.cat([x_tr1, x_cnn1], dim=1))

        # stage 2
        x_tr2, H1, W1 = self.patch_embed2(x_cat1)
        for i, blk in enumerate(self.block2):
            x_tr2 = blk(x_tr2, H1, W1)
        x_tr2 = self.norm2(x_tr2)
        x_tr2 = x_tr2.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn2 = self.encoder2(x_cat1)
        x_cat2 = self.FA2(torch.cat([x_tr2, x_cnn2], dim=1))

        # stage 3
        x_tr3, H1, W1 = self.patch_embed3(x_cat2)
        for i, blk in enumerate(self.block3):
            x_tr3 = blk(x_tr3, H1, W1)
        x_tr3 = self.norm3(x_tr3)
        x_tr3 = x_tr3.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn3 = self.encoder3(x_cat2)
        x_cat3 = self.FA3(torch.cat([x_tr3, x_cnn3], dim=1))

        # stage 4
        x_tr4, H1, W1 = self.patch_embed4(x_cat3)
        for i, blk in enumerate(self.block4):
            x_tr4 = blk(x_tr4, H1, W1)
        x_tr4 = self.norm4(x_tr4)
        x_tr4 = x_tr4.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn4 = self.encoder4(x_cat3)
        x_cat4 = self.FA4(torch.cat([x_tr4, x_cnn4], dim=1))

        e4 = self.aspp(x_cat4)

        # ESA
        x_cat2_addSem = self.esa1(e4, x_cat2)
        x_cat1_addSem = self.esa2(x_cat2_addSem, x_cat1)
        x_cnn_edge_addSem = self.esa3(x_cat1_addSem, x_cnn_edge)
        edge1 = x_cat2_addSem
        edge2 = x_cat1_addSem
        edge3 = x_cnn_edge_addSem

        # IMD
        x_cnn_edge_addSem_down = self.leakyRelu(self.convDown(x_cnn_edge_addSem))
        x_cat1_down = self.leakyRelu(self.convDown1(x_cat1 + x_cnn_edge_addSem_down))
        x_cat2_down = self.leakyRelu(self.convDown2(x_cat2 + x_cat1_down))
        x_cat3_down = self.leakyRelu(self.convDown3(x_cat3 + x_cat2_down))

        d4 = self.decoder4(e4 + x_cat3_down) + x_cat3 + x_cat2_down
        d3 = self.decoder3(d4) + x_cat2 + x_cat1_down
        d2 = self.decoder2(d3) + x_cat1 + x_cnn_edge_addSem_down
        d1 = self.decoder1(d2) + self.conv(x_cnn_edge_addSem)

        x_cnn_edge_addSem_down2 = self.leakyRelu(self.convDown1(x_cnn_edge_addSem_down))
        x_cnn_edge_addSem_down3 = self.leakyRelu(self.convDown2(x_cnn_edge_addSem_down2))
        sem1 = self.eem1(d1, x_cnn_edge_addSem)
        sem2 = self.eem2(d2, x_cnn_edge_addSem_down)
        sem3 = self.eem3(d3, x_cnn_edge_addSem_down2)
        sem4 = self.eem4(d4, x_cnn_edge_addSem_down3)
        sem5 = self.eem5(e4, x_cnn_edge_addSem)

        sem_res = [sem1, sem2, sem3, sem4, sem5]
        edge_res = [edge1, edge2, edge3]
        return sem_res, edge_res

    def forward(self, x):
        x = self.forward_features(x)
        return x

#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

class Decoder(nn.Module):
    def __init__(self, num_classes=1, embedding_dim=32, output_nc=2, edge_channel=[32, 64, 128, 256]):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.num_classes = num_classes
        self.edge_channel = edge_channel

        # convolutional Difference Modules
        self.diff_c5 = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c4 = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        self.ef3 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)
        self.ef2 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)
        self.ef1 = EdgeFusion(in_chn=self.embedding_dim, out_chn=self.embedding_dim)

        self.downconv1 = nn.Conv2d(in_channels=self.edge_channel[1], out_channels=self.embedding_dim, kernel_size=1, stride=1, padding=0)
        self.downconv2 = nn.Conv2d(in_channels=self.edge_channel[2], out_channels=self.embedding_dim, kernel_size=1, stride=1, padding=0)
        self.downconv3 = nn.Conv2d(in_channels=self.edge_channel[3], out_channels=self.embedding_dim, kernel_size=1, stride=1, padding=0)
        self.downconv4 = nn.Conv2d(in_channels=self.edge_channel[0], out_channels=self.embedding_dim, kernel_size=1,
                                   stride=1, padding=0)
        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 5, out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        self.edge_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.edgeOut = BilinearUp(self.embedding_dim, self.embedding_dim, 2)
        self.finalcat_edge = nn.Conv2d(self.embedding_dim, num_classes, 1, 1)

        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)

        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.change_edge = ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1, c5_1 = inputs1[0]
        c1_2, c2_2, c3_2, c4_2, c5_2 = inputs2[0]

        e1_1, e2_1, e3_1 = inputs1[1]
        e1_2, e2_2, e3_2 = inputs2[1]

        outputs = []

        # Edge Stage 1
        e1_1_down = self.downconv2(e1_1)
        e1_2_down = self.downconv2(e1_2)
        _e1 = self.ef1(e1_1_down, e1_2_down)
        _e1_up = resize(_e1, size=e3_2.size()[2:], mode='bilinear', align_corners=False)

        # Edge Stage 2
        e2_1_down = self.downconv1(e2_1)
        e2_2_down = self.downconv1(e2_2)
        _e2 = self.ef2(e2_1_down, e2_2_down) + F.interpolate(_e1, scale_factor=2, mode="bilinear")
        _e2_up = resize(_e2, size=e3_2.size()[2:], mode='bilinear', align_corners=False)

        # Edge Stage 3
        e3_1_down = self.downconv4(e3_1)
        e3_2_down = self.downconv4(e3_2)
        _e3 = self.ef3(e3_1_down, e3_2_down) + F.interpolate(_e2, scale_factor=2, mode="bilinear")
        # EMFF
        _e = self.leakyRelu(self.edge_fuse(torch.cat((_e1_up, _e2_up, _e3), dim=1)))

        e_out = self.edgeOut(_e)
        e_out = self.finalcat_edge(e_out)
        outputs.append(e_out)

        # Sem Stage 5
        c5_1_down = self.downconv4(c5_1)
        c5_2_down = self.downconv4(c5_2)
        _c5 = self.diff_c5(torch.cat((c5_1_down, c5_2_down), dim=1))
        _c5_up = resize(_c5, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 4
        c4_1_down = self.downconv3(c4_1)
        c4_2_down = self.downconv3(c4_2)
        _c4 = self.diff_c4(torch.cat((c4_1_down, c4_2_down), dim=1))
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)


        # Sem Stage 3
        c3_1_down = self.downconv2(c3_1)
        c3_2_down = self.downconv2(c3_2)
        _c3 = self.diff_c3(torch.cat((c3_1_down, c3_2_down), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 2
        c2_1_down = self.downconv1(c2_1)
        c2_2_down =self.downconv1(c2_2)
        _c2 = self.diff_c2(torch.cat((c2_1_down, c2_2_down), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Sem Stage 1
        c1_1_down = self.downconv4(c1_1)
        c1_2_down = self.downconv4(c1_2)
        _c1 = self.diff_c1(torch.cat((c1_1_down, c1_2_down), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")

        # SEMM
        _c = self.linear_fuse(torch.cat((_c5_up,_c4_up, _c3_up, _c2_up, _c1), dim=1))

        c = self.convd1x(_c)
        c = self.dense_1x(c)
        cp = self.change_probability(c)

        outputs.append(cp)
        return outputs


class EGCTNet(nn.Module):
    def __init__(self, img_size=256, input_nc=3, output_nc=2, embed_dim=32, num_classes=2):
        super(EGCTNet, self).__init__()
        #Encoder
        self.embed_dims = [64, 128, 256, 512]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1
        self.num_classes = num_classes
        self.img_size = img_size

        self.FE_IMD = Encoder(img_size=self.img_size, patch_size=3, in_chans=input_nc, num_classes=self.num_classes, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1], edge_channel=[32, 64, 128, 256])

        self.CD_ED = Decoder(num_classes=1, embedding_dim=self.embedding_dim, output_nc=output_nc, edge_channel=[32, 64, 128, 256])

    def forward(self, x1, x2):
        [fx1, fx2] = [self.FE_IMD(x1), self.FE_IMD(x2)]

        cp = self.CD_ED(fx1, fx2)
        return cp
