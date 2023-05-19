"""
Source code from

@misc{PFL,
  title={{PyTorch Face Landmark}: A Fast and Accurate Facial Landmark Detector},
  url={https://github.com/cunjian/pytorch_face_landmark},
  note={Open-source software available at https://github.com/cunjian/pytorch_face_landmark},
  author={Cunjian Chen},
  year={2021},
}
"""

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import get_perspective_transform, warp_perspective

##################################  Original Arcface Model #############################################################


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


##################################  MobileFaceNet #############################################################


class Conv_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(
        self,
        in_c,
        out_c,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(
            in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride
        )
        self.project = Linear_block(
            groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(
        self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
    ):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(
                    c,
                    c,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class GNAP(Module):
    def __init__(self, embedding_size):
        super(GNAP, self).__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(
            512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        # self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class MobileFaceNet(Module):
    def __init__(self, input_size, embedding_size=512, output_name="GDC"):
        super(MobileFaceNet, self).__init__()
        assert output_name in ["GNAP", "GDC"]
        assert input_size[0] in [112]
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(
            64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
        )
        self.conv_23 = Depth_Wise(
            64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128
        )
        self.conv_3 = Residual(
            64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_34 = Depth_Wise(
            64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256
        )
        self.conv_4 = Residual(
            128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_45 = Depth_Wise(
            128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512
        )
        self.conv_5 = Residual(
            128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_6_sep = Conv_block(
            128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)

        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features).detach()
        return out, conv_features

    @torch.no_grad()
    def get_face_landmark(self, x):
        """
        이미지에서 face 의 68개 landmark 를 찾는다.

        :param x: 이미지
        :return: landmark
        """

        size = x.size(2)
        x = F.interpolate(x, size=(112, 112), mode="bilinear")
        landmark = self.forward(x)[0]
        #print(landmark.max())
        landmark = landmark * size
        landmark = landmark.view(landmark.size(0), -1, 2)  # N x 68 x 2

        return landmark

    def align_face(self, inputs, scale=1.25, inverse=False, target_size=224):
        lm = self.get_face_landmark(inputs)

        # ref quad
        ref_quad = (
            torch.tensor(
                [
                    [0, 0],
                    [0, target_size - 1],
                    [target_size - 1, target_size - 1],
                    [target_size - 1, 0],
                ]
            )
            .float()
            .to(inputs.device)
            .unsqueeze(0)
        )

        # Get left, right, top, bottom
        #breakpoint()
        l = lm[:, :, 0].min(dim=1, keepdim=True)[0]
        r = lm[:, :, 0].max(dim=1, keepdim=True)[0]
        t = lm[:, :, 1].min(dim=1, keepdim=True)[0]
        b = lm[:, :, 1].max(dim=1, keepdim=True)[0]

        # Calcualte new img size with margin
        old_size = (r - l + b - t) / 2 #* 1.1
        #print('dnon')
        new_size = old_size * scale

        # Calculate center
        center_x = (r + l) / 2
        center_y = (b + t) / 2

        # Calculate new l, r, t, b
        l = (center_x - (new_size / 2)).unsqueeze(dim=1)
        r = (center_x + (new_size / 2)).unsqueeze(dim=1)
        t = (center_y - (new_size / 2)).unsqueeze(dim=1)
        b = (center_y + (new_size / 2)).unsqueeze(dim=1)

        # Calcualte quads
        tl = torch.cat([l, t], dim=2)
        bl = torch.cat([l, b], dim=2)
        br = torch.cat([r, b], dim=2)
        tr = torch.cat([r, t], dim=2)
        quads = torch.cat([tl, bl, br, tr], dim=1)

        # Calculate transform matrix
        if inverse:
            mats = get_perspective_transform(
                ref_quad.repeat(quads.shape[0], 1, 1), quads.float()
            )
        else:
            mats = get_perspective_transform(
                quads.float(), ref_quad.repeat(quads.shape[0], 1, 1)
            )
            target_size = (target_size, target_size)

        # Warping
        outs = warp_perspective(inputs.float(), mats, target_size, padding_mode="zeros")

        return outs

    def landmark_dist(self, src, tgt):
        predefined_idx = 17
        landmark_src = self.get_face_landmark(src)[
            :, :predefined_idx, :
        ]  # N x 17 x 2 (1 x 17 x 2)
        landmark_dst = self.get_face_landmark(tgt)[
            :, :predefined_idx, :
        ]  # N x 17 x 2 (1 x 17 x 2)

        dist = torch.mean(
            torch.norm(landmark_src - landmark_dst, p="fro", dim=-1)
        )  # (N x 17 x 2) -> (N x 17) -> 1, (1 x 17 x 2) -> (1 x 17) -> 1, 두 이미지 사이의 landmark distance

        return dist


def load_face_landmark_detector():
    """
    Face landmark detection 모델을 불러온다.

    :return: Face landmark detection 모델
    """

    model = MobileFaceNet([112, 112], 136)
    checkpoint = torch.load(
        "/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/mobilefacenet_model_best.pth.tar",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval()
    return model
