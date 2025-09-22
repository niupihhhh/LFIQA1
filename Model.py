import torch
import torch.nn as nn
from VAN import van_b0, van_b1, van_b2, van_b3
from VSSM import VSSBlock
from HCConv2 import HCConvModule
import math
from einops import rearrange
from torch.nn import functional as F


class ISSB(nn.Module):

    def __init__(self, inchannel, outchannel):
        super(ISSB, self).__init__()

        self.mamba = VSSBlock(inchannel)
        self.HCConv = HCConvModule(inchannel, outchannel)

    def forward(self, x):
        x1 = self.mamba(x)
        x2 = self.HCConv(x1)
        out = torch.cat((x1, x2), 1)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Squeeze
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)  # Excitation
        return x * y.expand_as(x)

class CNNBranch3D(nn.Module):
    def __init__(self, channels):
        super(CNNBranch3D, self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0),  # 1×1×1 Conv
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1,  # 3×3×3 depthwise
                      groups=channels, bias=False),
            nn.LeakyReLU(inplace=True),



            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)  # 1×1×1 pointwise
        )

        self.se_block = SEBlock(channels)

    def forward(self, x):
        residual = x  # Save the input for residual connection

        x = self.conv_branch(x)  # Apply the convolution sequence
        x = self.se_block(x)  # Apply SEBlock for channel-wise feature enhancement

        x += residual

        return x


class AngTrans(nn.Module):
    def __init__(self, channels, angRes, MHSA_params):
        super(AngTrans, self).__init__()
        self.angRes = angRes
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):
        buffer_token = rearrange(buffer, 'b c a h w -> a (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token):
        buffer = rearrange(buffer_token, '(a) (b h w) (c) -> b c a h w', a=self.angRes, h=self.h, w=self.w)
        return buffer

    def forward(self, buffer):
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(self.ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token)

        return buffer


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)



class Network(nn.Module):

    # def __init__(self,loss_weights):
    def __init__(self):
        super(Network, self).__init__()

        # self.loss_weights = nn.Parameter(torch.Tensor(loss_weights))

        self.div_h = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(7, 1, 1), stride=(7, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.div_v = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(7, 1, 1), stride=(1, 1, 1), dilation=(7, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        '''horizontal'''
        self.h_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.h_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.h_conv4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        '''vertical'''
        self.v_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.v_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.v_conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.v_conv4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        '''regression'''
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 1)
        self.linear_3 = nn.Linear(832, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.van = van_b0(pretrained=True)

        self.relu = nn.ReLU()
        self.gap_3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        self.issb_1 = ISSB(inchannel=32, outchannel=32)
        self.issb_2 = ISSB(inchannel=64, outchannel=64)
        self.issb_3 = ISSB(inchannel=160, outchannel=160)
        self.issb_4 = ISSB(inchannel=256, outchannel=256)

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.task_main = nn.Sequential(
            nn.Linear(1088, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.task_aux1 = nn.Sequential(
            nn.Linear(1088, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        self.task_aux2 = nn.Sequential(
            nn.Linear(1088, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化权重参数alpha
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.angRes = 7
        self.channels_1 = 128
        self.channels_2 = 256
        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.trans_1 = AngTrans(self.channels_1, self.angRes, self.MHSA_params)
        self.trans_2 = AngTrans(self.channels_2, self.angRes, self.MHSA_params)

        self.CNNBranch = CNNBranch3D(128)
        self.CNNBranch1 = CNNBranch3D(256)

    def forward(self, x, y):
        v_x = self.div_v(x)
        v_x = self.v_conv1(v_x)
        v_x = self.v_conv2(v_x)

        for m in self.modules():
            m.h = v_x.size(-2)
            m.w = v_x.size(-1)
        ang_position = self.pos_encoding(v_x, dim=[2], token_dim=self.channels_1)
        for m in self.modules():
            m.ang_position = ang_position
        v_x1 = self.trans_1(v_x)
        v_x2 = self.CNNBranch(v_x)
        v_x = v_x1 + v_x2

        v_x = self.v_conv3(v_x)
        v_x = self.v_conv4(v_x)

        v_x = self.gap_3D(v_x)
        v_x = self.flat(v_x)

        q1 = self.linear_1(v_x)
        q1 = self.linear_2(q1)

        y = self.conv(y)
        y1, y2, y3, y4 = self.van(y)

        y3 = self.issb_3(y3)

        y3 = self.avg_pool(y3)
        y3 = self.flat(y3)

        y4 = self.issb_4(y4)
        y4 = self.avg_pool(y4)
        y4 = self.flat(y4)
        q2 = torch.cat([y3, y4], dim=1)
        q2 = self.linear_3(q2)
        q2 = self.linear_4(q2)
        weight1= torch.sigmoid(self.alpha)
        weight2= torch.sigmoid(self.beta)
        q = weight1 * q1 + weight2 * q2

        out = torch.cat([y3, y4, v_x], dim=1)
        c1 = self.task_aux1(out)
        c2 = self.task_aux2(out)
        return q, c1, c2

if __name__ == "__main__":
    net = Network().cuda()
    from thop import profile
    input1 = torch.randn(2, 1, 49, 32, 32).cuda()
    input2 = torch.randn(2, 1, 224, 224).cuda()
    q, c1, c2 = net(input1, input2)
    flops, params = profile(net, inputs=(input1,input2))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
    print(q.shape)
    print(c1.shape)
    print(c2.shape)



