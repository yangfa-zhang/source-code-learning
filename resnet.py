import torch.nn as nn
from typing import Optional,Callable,Type,Union,List
from torch import Tensor
import torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,#输入输出通道数，如rgb图像每个像素点的通道数为3
        kernel_size=3,
        stride=stride,
        padding=dilation,#填充0的层数
        groups=groups,#为1时，所有输入通道共享同一个卷积核
        bias=False,#偏置项，wx+b
        dilation=dilation,#为1时，3x3卷积->5x5卷积
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


#basicblock和bottleneck的区别，basicblock两个卷积层，用于较浅的resnet18/34，bottleneck三个卷积层，用于较深的resnet50/101
class BasicBlock(nn.Module):
    expansion: int = 1 #输出特征图宽度out_channels/输入特征图维度inplanes

    def __init__(
        self,
        inplanes: int,
        planes: int,#输出通道数out_planes
        stride: int = 1,
        downsample: Optional[nn.Module] = None,#解决维度不匹配问题，通过卷积操作进行维度转换，optional就是进行/不进行downsample操作
        groups: int = 1,#为1时，所有输入通道共享同一个卷积核
        base_width: int = 64,#每个残差块的宽度，调整此值，控制网络中参数数量
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,#可调用callable对象，如batchnorm，归一化操作，防止过拟合
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d #none使用批量归一化
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion:int=4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,#将残差连接的权重设为0
        groups: int = 1,#卷积核数量，e.g.32个输入通道（输入数据每个位置的特征数量为32）分为4组每组8个，4个卷积核就与对应一个组的通道进行交互
        width_per_group: int = 64,#每组的卷积核的宽度
        replace_stride_with_dilation: Optional[List[bool]] = None,#是否用dilation来代替stride
        norm_layer: Optional[Callable[..., nn.Module]] = None,#设为none时则不使用正则化层
    ) -> None:
        super().__init__()#调用父类的构造函数
        _log_api_usage_once(self)#自定义的日志记录函数，在：https://github.com/pytorch/vision/blob/main/torchvision/utils.py 中找到
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    #构建并放回又多个basicblock/bottleneck组成的层序列用于构建resnet
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer #在_init_中已定义
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        '''
        # 假设我们有一系列层
        conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        relu2 = nn.ReLU()

        # 使用 nn.Sequential 创建一个顺序容器
        sequential_model = nn.Sequential(*[conv1, relu1, conv2, relu2])
        '''
        return nn.Sequential(*layers)
        
    
    def forward(self,x:Tensor) -> Tensor:
         # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x