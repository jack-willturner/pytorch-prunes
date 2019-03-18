import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ZeroMake(nn.Module):
    def __init__(self, channels, spatial):
        super(ZeroMake, self).__init__()
        self.spatial = spatial
        self.channels = channels

    def forward(self, x):
        return torch.zeros([x.size()[0], self.channels, x.size()[2]//self.spatial, x.size()[3]//self.spatial], dtype=x.dtype, layout=x.layout, device=x.device)


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
    def forward(self, x):
            return x * 0

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MaskBlock(nn.Module):
    expansion = 1
    '''Wrap-round block for resnets. Doesnt incorporate dropout yet.'''

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(MaskBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.activation = Identity()
        self.activation.register_backward_hook(self._fisher)
        self.register_buffer('mask', None)

        self.input_shape = None
        self.output_shape = None
        self.flops = None
        self.params = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.got_shapes = False

        # Fisher method is called on backward passes
        self.running_fisher = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)


        if self.mask is not None:
            out = out * self.mask[None, :, None, None]

        else:
            self._create_mask(x, out)

        out = self.activation(out)
        self.act = out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out



    def _create_mask(self, x, out):
        """This takes an activation to generate the exact mask required. It also records input and output shapes
        for posterity."""
        self.mask = x.new_ones(out.shape[1])
        self.input_shape = x.size()
        self.output_shape = out.size()

    def _fisher(self, blargh, blergh, grad_output):
        act = self.act.detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher += del_k


    def reset_fisher(self):
        self.running_fisher = 0 * self.running_fisher

    def update(self, previous_mask):
        return None

    def cost(self):

        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size = self.conv1.weight.size()
        conv2_size = self.conv2.weight.size()

        self.params = in_channels * middle_channels * conv1_size[2] * conv1_size[3] + middle_channels * out_channels * \
                      conv2_size[2] * conv2_size[3]
        self.flops = self.output_shape[2] * self.output_shape[3] * conv1_size[2] * conv1_size[
            3] * in_channels * middle_channels \
                     + self.output_shape[2] * self.output_shape[3] * conv2_size[2] * conv2_size[
                         3] * middle_channels * out_channels

        # Ignore batch-norm for now as it's ridiculously small. Also ignoring the skip connection conv.

        self.flops_vector = self.mask * (self.flops / self.out_channels)

    def compress_weights(self):

        # This all hinges on the channel dimension between the two convs
        middle_dim = int(self.mask.sum().item())
        print(middle_dim)
        # BN1 doesn't change.

        if middle_dim is not 0:

            conv1 = nn.Conv2d(self.in_channels, middle_dim, kernel_size=3, stride=self.stride, padding=1, bias=False)
            conv1.weight = nn.Parameter(self.conv1.weight[self.mask == 1, :, :, :])

            # BN1 changes
            bn1 = nn.BatchNorm2d(middle_dim)
            bn1.weight = nn.Parameter(self.bn1.weight[self.mask == 1])
            bn1.bias = nn.Parameter(self.bn1.bias[self.mask == 1])
            bn1.running_mean = self.bn1.running_mean[self.mask == 1]
            bn1.running_var = self.bn1.running_var[self.mask == 1]

            # Batch norm 2 DOESN'T change

            conv2 = nn.Conv2d(middle_dim, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            conv2.weight = nn.Parameter(self.conv2.weight[:, self.mask == 1, :, :])



        if middle_dim is 0:
            conv1 = Zero()
            bn2 = Zero()
            conv2 = ZeroMake(channels=self.out_channels, spatial=self.stride)



        # The big one
        self.conv1 = conv1
        self.conv2 = conv2
        self.bn1 = bn1
        self.activation = Identity()


        if middle_dim is not 0:
            self.mask = torch.ones(middle_dim)
        else:
            self.mask = torch.ones(1)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, mask=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock if not mask else MaskBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, mask=False,**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock if not mask else MaskBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
