import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
import torch
import math


def creat_norm_layer(norm_layer, channel, is_token=False):
    """
    Args:
        norm_layer (str): Normalization layer type, use 'BN' or 'LN'.
        channel (int): Input channels.
        is_token (bool): Whether to process token. Default: False
    """
    if not is_token:
        if norm_layer == 'LN':
            norm = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(channel),
                Rearrange('b h w c -> b c h w')
            )
        elif norm_layer == 'BN':
            norm = nn.BatchNorm2d(channel)
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    else:
        if norm_layer == 'LN':
            norm = nn.LayerNorm(channel)
        elif norm_layer == 'BN':
            norm = nn.Sequential(
                Rearrange('b d n -> b n d'),
                nn.BatchNorm1d(channel)
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


class Spatial_attention(nn.Module):
    """
    Build spatial attention

    Args:
        encoder_chans (int): Input channels from encoder.
        decoder_chans (int): Input channels from decoder.
        act_layer (nn.Module): Act layer. Default: nn.ReLU

    Return shape: (b c h w)
    """

    def __init__(self, encoder_chans, decoder_chans, attn_chans=None, act_layer=nn.ReLU):
        super().__init__()
        attn_chans = attn_chans or decoder_chans
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(encoder_chans, attn_chans, kernel_size=1),
            nn.BatchNorm2d(attn_chans)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_chans, attn_chans, kernel_size=1),
            nn.BatchNorm2d(attn_chans)
        )
        self.attn = nn.Sequential(
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(attn_chans, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x_en, x_de):
        """
        x_en: feature map from encoder
        x_de: Feature map from decoder
        """
        x_en = self.conv1(x_en)
        x_de = self.conv2(x_de)

        return x_de * self.attn(x_en + x_de)


class Dw_spatial_attention(nn.Module):
    """
    Build spatial attention by downscaling convolution

    Args:
        in_chans (int): Input channels.

    Return shape: (b c h w)
    """

    def __init__(self, in_chans):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_chans, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, _, x):
        return x * self.attn(x)


class SE_channel_attention(nn.Module):
    """
    Build channel attention based on SE

    Args:
        in_chans (int): Number of input image channels.
        ratio (int): Scaling ratio. Default: 4
        act_layer (nn.Module): Act layer. Default: nn.ReLU6

    Return shape: (b c h w)
    """

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class CBAM_channel_attention(nn.Module):
    """
    Build channel attention based on CBAM

    Args:
        in_chans (int): Number of input image channels.
        ratio (int): Scaling ratio. Default: 4

    Return shape: (b c h w)
    """

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.conv2 = nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.act_layer(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.act_layer(self.conv1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight

class Build_decode_gate(nn.Module):
    """
    Build decode gate

    Args:
        in_chans (int): Number of input image channels.
        n_classes (int): Number of probabilities you want to get per pixel.
        norm_layer (str): Normalization layer.
        act_layer (optional): Act layer.
        head_chans (int | None): Number of decoder head image channels.
        chan_ratio (int): Scaling ratio of 'CBAM' or 'SE' channel attention. Default: 16
        chan_attn_type (str): Channnel attention method, using 'CBAM' or 'SE'. Default: 'SE'
        dw_spac_attn (bool): Whether to use 'dw' spatial attention.
        en_chans (int | None): Number of encoder feature channels.

    Return shape: (b c H W)
    """

    def __init__(self, in_chans, n_classes, norm_layer, act_layer, head_chans=None,
                 chan_ratio=16, chan_attn_type='SE', dw_spac_attn=False, en_chans=None):
        super().__init__()
        head_chans = head_chans or in_chans // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, head_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, head_chans)
        )

        self.spat_attn = Spatial_attention(encoder_chans=en_chans,
                                           decoder_chans=head_chans,
                                           attn_chans=None,
                                           act_layer=act_layer
                                           ) if not dw_spac_attn else Dw_spatial_attention(head_chans)

        self.dwconv = nn.Sequential(
            nn.Conv2d(head_chans, head_chans, kernel_size=3, padding=1, groups=head_chans),
            creat_norm_layer(norm_layer, head_chans),
            nn.Conv2d(head_chans, in_chans, kernel_size=1, bias=False)
        )

        self.out = nn.Sequential(
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans, n_classes, kernel_size=1)
        )

        if chan_attn_type == 'CBAM':
            self.chan_attn = CBAM_channel_attention(in_chans=head_chans, ratio=chan_ratio)
        elif chan_attn_type == 'SE':
            self.chan_attn = SE_channel_attention(in_chans=head_chans, ratio=chan_ratio)
        else:
            raise NotImplementedError(f"Build channel attention does not support {chan_attn_type}")


    def forward(self, x, x1):
        """
        x: Feature map from encoder
        x1: Feature map from decoder
        """
        short_cut = x1
        x1 = self.conv(x1)

        spat_x = self.spat_attn(x, x1)
        chan_x = self.chan_attn(x1)
        fuse_attn_x = self.dwconv(spat_x + chan_x)

        x = short_cut + fuse_attn_x
        x = self.out(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        # _, _, h, w = feats.shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        # self.stride = torch.zeros(self.nl)  # strides computed during build
        self.stride = torch.Tensor([32, 16, 8, 4])  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            # x = torch.cat((self.cv2[i](x), self.cv3[i](x)), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        # print(shape)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        # x_cat = x.view(shape[0], self.no, -1)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides