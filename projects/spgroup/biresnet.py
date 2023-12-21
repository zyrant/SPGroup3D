# Adapted from https://github.com/Haiyang-W/CAGroup3D/blob/main/pcdet/models/backbones_3d/biresnet.py
# by --zyrant

import torch.nn as nn
import MinkowskiEngine as ME
from mmdet.models import BACKBONES
from mmcv.runner import BaseModule
import torch
import math
BatchNorm = ME.MinkowskiBatchNorm
bn_mom = 0.1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 no_relu=False,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 no_relu=True,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, stride=1, bias=False, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, bias=False, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1,
                               bias=False, dimension=dimension)
        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, stride=1, bias=False, dilation=dilation, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class BaseBottleneck(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 no_relu=True,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BaseBottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, stride=1, bias=False, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, bias=False, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1,
                               bias=False, dimension=dimension)
        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, stride=1, bias=False, dilation=dilation, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)
        
        

class Bottle2neck(nn.Module):

    # https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
    expansion = 1

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 no_relu=False,
                 bn_momentum=0.1,
                 baseWidth=26, 
                 dimension=-1,
                 scale = 4, 
                 stype='normal'):
        super(Bottle2neck, self).__init__()
        assert dimension > 0
        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = ME.MinkowskiConvolution(inplanes, width*scale, kernel_size=1, bias=False, dimension=dimension)
        self.bn1 = ME.MinkowskiBatchNorm(width*scale, momentum=bn_momentum)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = ME.MinkowskiAvgPooling(kernel_size=3, stride = stride)
        convs = []
        norms = []
        for i in range(self.nums):
            convs.append(ME.MinkowskiConvolution(width, width, kernel_size=3, stride = stride, bias=False, dimension=dimension))
            norms.append(ME.MinkowskiBatchNorm(width, momentum=bn_momentum))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.conv3 = ME.MinkowskiConvolution(width*scale, planes * self.expansion, kernel_size=1, bias=False, dimension=dimension)
        self.bn3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out.F, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = ME.SparseTensor(
            spx[i],
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager)
          else:
            sp = ME.SparseTensor(
                sp.features_at_coordinates(x.C.float()) + spx[i],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager)
          sp = self.convs[i](sp)
          sp = self.relu(self.norms[i](sp))
          if i==0:
            out = sp
          else:
            out = ME.cat((out, sp))
        if self.scale != 1 and self.stype=='normal':
            spx_nums =  ME.SparseTensor(
                            spx[self.nums],
                            coordinate_map_key=x.coordinate_map_key,
                            coordinate_manager=x.coordinate_manager)
            out = ME.cat((out, ME.SparseTensor(
                            spx_nums.features_at_coordinates(out.C.float()),
                            coordinate_map_key=out.coordinate_map_key,
                            coordinate_manager=out.coordinate_manager)))
        elif self.scale != 1 and self.stype=='stage':
            spx_nums =  ME.SparseTensor(
                            spx[self.nums],
                            coordinate_map_key=x.coordinate_map_key,
                            coordinate_manager=x.coordinate_manager)
            out = ME.cat((out, self.pool(ME.SparseTensor(
                            spx_nums.features_at_coordinates(out.C.float()),
                            coordinate_map_key=out.coordinate_map_key,
                            coordinate_manager=out.coordinate_manager))))

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, dimension=-1):
        assert dimension > 0
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=5, stride=2, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale2 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=9, stride=4, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale3 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=17, stride=8, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale4 = nn.Sequential(ME.MinkowskiAvgPooling(kernel_size=33, stride=16, dimension=dimension),
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.scale0 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, branch_planes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.process1 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )
        self.process2 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )
        self.process3 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )
        self.process4 = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes, branch_planes, kernel_size=3, bias=False, dimension=dimension),
                                    )        
        self.compression = nn.Sequential(
                                    ME.MinkowskiBatchNorm(branch_planes * 5, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        branch_planes * 5, outplanes, kernel_size=1, bias=False, dimension=dimension),
                                    )
        self.shortcut = nn.Sequential(
                                    ME.MinkowskiBatchNorm(inplanes, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(
                                        inplanes, outplanes, kernel_size=1, bias=False, dimension=dimension),
                                    )

    def forward(self, x):
        x_list = []
        x_coords = x.C.float()

        x_list.append(self.scale0(x))

        x_scale1_tensor = self.scale1(x).features_at_coordinates(x_coords)
        x_scale1 = ME.SparseTensor(features=x_scale1_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process1(x_scale1+x_list[0]))

        x_scale2_tensor = self.scale2(x).features_at_coordinates(x_coords)
        x_scale2 = ME.SparseTensor(features=x_scale2_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process2(x_scale2+x_list[1]))

        x_scale3_tensor = self.scale3(x).features_at_coordinates(x_coords)
        x_scale3 = ME.SparseTensor(features=x_scale3_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process3(x_scale3+x_list[2]))

        x_scale4_tensor = self.scale4(x).features_at_coordinates(x_coords)
        x_scale4 = ME.SparseTensor(features=x_scale4_tensor,
            coordinate_manager=x.coordinate_manager, coordinate_map_key=x.coordinate_map_key)
        x_list.append(self.process4(x_scale4+x_list[3]))

        out = self.compression(ME.cat(*x_list)) + self.shortcut(x)
        return out 


class segmenthead(nn.Module):

    def __init__(self,
                 inplanes,
                 interplanes,
                 outplanes,
                 dimension=-1):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm(inplanes, momentum=bn_mom)
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, interplanes, kernel_size=3, bias=False, dimension=dimension)
        self.bn2 = BatchNorm(interplanes, momentum=bn_mom)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            interplanes, outplanes, kernel_size=1, bias=True, dimension=dimension)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out

@BACKBONES.register_module()
class BiResNet(nn.Module):

    def __init__(self,
                 in_channels =3,
                 out_channels = 64, 
                 layers = [2, 2, 2, 2],
                 planes = 64, 
                 spp_planes = 128,
                 head_planes = 128, 
                 dimension = 3,
                 block=BasicBlock,
                 return_4x = False):
        super(BiResNet, self).__init__()
        highres_planes = planes * 2
        spp_planes = planes * 2
        self.return_4x = return_4x

        self.conv1 =  nn.Sequential(
                          ME.MinkowskiConvolution(
                              in_channels, planes, kernel_size=3, stride=1, dimension=dimension), # ori: stride=2
                          BatchNorm(planes, momentum=bn_mom),
                          ME.MinkowskiReLU(inplace=True),
                          ME.MinkowskiConvolution(
                              planes,planes,kernel_size=3, stride=1, dimension=dimension), # ori: stride=2
                          BatchNorm(planes, momentum=bn_mom),
                          ME.MinkowskiReLU(inplace=True),
                      )
        # self.conv1 = nn.Sequential(
        #     ME.MinkowskiConvolution(
        #         in_channels, planes, kernel_size=3, stride=1, dimension=dimension
        #     ),
        #     ME.MinkowskiInstanceNorm(planes),
        #     ME.MinkowskiReLU(inplace=True),
        #     # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
        # )

        self.relu = ME.MinkowskiReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0], stride=2, dimension=dimension)
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2, dimension=dimension)
        self.layer3 = self._make_layer(block, planes * 2, planes * 2, layers[2], stride=2, dimension=dimension)
        self.layer4 = self._make_layer(block, planes * 2, planes * 2, layers[3], stride=2, dimension=dimension)

        self.compression3 = nn.Sequential(
                                          ME.MinkowskiConvolution(
                                              planes * 2, highres_planes, kernel_size=1, bias=False, dimension=dimension),
                                          BatchNorm(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          ME.MinkowskiConvolution(
                                              planes * 2, highres_planes, kernel_size=1, bias=False, dimension=dimension),
                                          BatchNorm(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   ME.MinkowskiConvolution(
                                       highres_planes, planes * 2, kernel_size=3, stride=2, bias=False, dimension=dimension),
                                   BatchNorm(planes * 2, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   ME.MinkowskiConvolution(
                                       highres_planes, planes * 2, kernel_size=3, stride=2, bias=False, dimension=dimension),
                                   BatchNorm(planes * 2, momentum=bn_mom),
                                   ME.MinkowskiReLU(inplace=True),
                                   ME.MinkowskiConvolution(
                                       planes * 2, planes * 2, kernel_size=3, stride=2, bias=False, dimension=dimension),
                                   BatchNorm(planes * 2, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2, dimension=dimension)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2, dimension=dimension)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1, dimension=dimension)

        self.layer5 =  self._make_layer(Bottleneck, planes * 2, planes * 2, 1, stride=2, dimension=dimension)

        self.spp = DAPPM(planes * 4, spp_planes, planes * 4, dimension=dimension)

        # MinkowskiConvolutionTranspose会返回上原来下采样那边一样的坐标和特征数
        # MinkowskiGenerativeConvolutionTranspose会插值, 产生更多的坐标和特征, 怎么插值的不知道
        if self.return_4x:
            self.out = nn.Sequential(
                                    ME.MinkowskiConvolution(planes * 4, planes * 4, kernel_size=3, bias=False, dimension=dimension),
                                    ME.MinkowskiBatchNorm(planes * 4, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(planes * 4, out_channels, kernel_size=1, bias=False, dimension=dimension),
                                    ME.MinkowskiBatchNorm(out_channels, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True))
        else:
            self.out = nn.Sequential(
                                    ME.MinkowskiConvolutionTranspose(planes * 4, planes * 4, kernel_size=2, stride=2, dimension=dimension),
                                    ME.MinkowskiBatchNorm(planes * 4, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True),
                                    ME.MinkowskiConvolution(planes * 4, out_channels, kernel_size=1, bias=False, dimension=dimension),
                                    ME.MinkowskiBatchNorm(out_channels, momentum=bn_mom),
                                    ME.MinkowskiReLU(inplace=True))
        # self.final_layer = segmenthead(planes * 4, head_planes, out_channels, dimension=dimension) # NOTE: we dont need this layer anymore

        self.num_point_features = out_channels
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dimension=-1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dimension=dimension),
                BatchNorm(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, downsample=downsample, dimension=dimension))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True, dimension=dimension))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, dimension=dimension))

        return nn.Sequential(*layers)


    def forward(self, input):
        x = input
        layers = []
        init_layer = []

        x = self.conv1(x) # 1
        init_layer.append(x)

        x = self.layer1(x) # 2
        layers.append(x)

        x = self.layer2(self.relu(x)) # 4
        layers.append(x)
  
        x = self.layer3(self.relu(x)) # 8
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1])) # 4

        x = x + self.down3(self.relu(x_)) # 8 
        x_f = x_.F + self.compression3(self.relu(layers[2])).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        x = self.layer4(self.relu(x)) # 16
        layers.append(x)
        x_ = self.layer4_(self.relu(x_)) # 4

        x = x + self.down4(self.relu(x_)) # 16
        x_f = x_.F + self.compression4(self.relu(layers[3])).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        x_ = self.layer5_(self.relu(x_)) # 4
        x_f = x_.F + self.spp(self.layer5(self.relu(x))).features_at_coordinates(x_.C.float())
        x_ = ME.SparseTensor(features=x_f,
            coordinate_manager=x_.coordinate_manager, coordinate_map_key=x_.coordinate_map_key)

        x_ = self.out(x_) # 2

        # x_f = init_layer[0].F + self.last_out(x_).features_at_coordinates(init_layer[0].C.float())
        # x_ = ME.SparseTensor(features=x_f,
        #     coordinate_manager=init_layer[0].coordinate_manager, coordinate_map_key=init_layer[0].coordinate_map_key) 

        return x_


if __name__ == '__main__':
    import torch
    f = torch.rand(2048, 3).float().cuda()
    c1 = torch.randint(-64, 64, (1024, 4)).cuda().float()
    c2 = torch.randint(-64, 64, (1024, 4)).cuda().float()
    c1[:,0] = 0
    c2[:,0] = 1
    c = torch.cat((c1, c2),dim=0)
    x = ME.SparseTensor(coordinates=c, features=f)
    # model_cfg = {'out_channels':18, 'planes':132, 'spp_planes':128, 'head_planes':64, 'augment':True, 'in_channels':3}
    net = BiResNet().cuda()
    y = net(x)
    y_decomposed_features = y.decomposed_features
    print(len(y_decomposed_features[0]), len(y_decomposed_features[1]))
    y_slice = y.slice(x)
    y_slice_decomposed_features = y_slice.decomposed_features
    print(len(y_slice_decomposed_features[0]), len(y_slice_decomposed_features[1]))
    print(x.F.shape, y.F.shape, y.coordinate_map_key)
    print(y.F[100:106])