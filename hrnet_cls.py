'''
Modified by mengtianjian
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Ke Sun, Yang Zhao, Borui Jiang, Tianheng Cheng, Bin Xiao, Dong Liu, Yadong Mu, Xinggang Wang,
Wenyu Liu, Jingdong Wang. 'High-Resolution Representations for Labeling Pixels and Regions'

'''
import numpy as np
import mxnet as mx
import hrnet


def residual_unit(data,
                  output_channels,
                  stride,
                  dim_match,
                  name,
                  bottle_neck=True,
                  bn_mom=0.9,
                  workspace=256):
    '''Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    output_channels : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to output_channels
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    '''
    if bottle_neck:
        output_channels_act1 = int(output_channels * 0.25)
        conv1 = mx.sym.Convolution(
            data=data,
            num_filter=output_channels_act1,
            kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            no_bias=True,
            workspace=workspace,
            name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(
            data=conv1,
            fix_gamma=False,
            eps=2e-5,
            momentum=bn_mom,
            name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        output_channels_act2 = int(output_channels * 0.25)
        conv2 = mx.sym.Convolution(
            data=act1,
            num_filter=output_channels_act2,
            kernel=(3, 3),
            stride=stride,
            pad=(1, 1),
            no_bias=True,
            workspace=workspace,
            name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(
            data=conv2,
            fix_gamma=False,
            eps=2e-5,
            momentum=bn_mom,
            name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = mx.sym.Convolution(
            data=act2,
            num_filter=output_channels,
            kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            no_bias=True,
            workspace=workspace,
            name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(
            data=conv3,
            fix_gamma=False,
            eps=2e-5,
            momentum=bn_mom,
            name=name + '_bn3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(
                data=data,
                num_filter=output_channels,
                kernel=(1, 1),
                stride=stride,
                no_bias=True,
                workspace=workspace,
                name=name + '_sc')
            shortcut = mx.sym.BatchNorm(
                data=shortcut,
                fix_gamma=False,
                eps=2e-5,
                momentum=bn_mom,
                name=name + '_sc_bn')
        out = mx.sym.Activation(data=bn3+shortcut, act_type='relu', name=name + '_relu3')
        return out
    else:
        conv1 = mx.sym.Convolution(
            data=data,
            num_filter=output_channels,
            kernel=(3, 3),
            stride=stride,
            pad=(1, 1),
            no_bias=True,
            workspace=workspace,
            name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(
            data=conv1,
            fix_gamma=False,
            momentum=bn_mom,
            eps=2e-5,
            name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(
            data=act1,
            num_filter=output_channels,
            kernel=(3, 3),
            stride=stride,
            pad=(1, 1),
            no_bias=True,
            workspace=workspace,
            name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(
            data=conv2,
            fix_gamma=False,
            momentum=bn_mom,
            eps=2e-5,
            name=name + '_bn2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(
                data=data,
                num_filter=output_channels,
                kernel=(1, 1),
                stride=stride,
                no_bias=True,
                workspace=workspace,
                name=name + '_sc')
            shortcut = mx.sym.BatchNorm(
                data=shortcut,
                fix_gamma=False,
                eps=2e-5,
                momentum=bn_mom,
                name=name + '_sc_bn')
        out = mx.sym.Activation(data=bn2+shortcut, act_type='relu', name=name + '_relu2')
        return out


def conv3x3(data,
            output_channel,
            stride,
            name,
            act=True,
            bn_mom=0.9,
            workspace=256):
    out = mx.sym.Convolution(
        data=data,
        num_filter=output_channel,
        kernel=(3, 3),
        stride=stride,
        pad=(1, 1),
        no_bias=True,
        name=name+'_conv',
        workspace=workspace)
    out = mx.sym.BatchNorm(
        data=out,
        fix_gamma=False,
        eps=2e-5,
        momentum=bn_mom,
        name=name+'_bn')
    if act:
        out = mx.sym.Activation(data=out, act_type='relu', name=name+'_relu')
    return out


def conv1x1(data,
            output_channel,
            stride,
            name,
            act=True,
            bn_mom=0.9,
            workspace=256):
    out = mx.sym.Convolution(
        data=data,
        num_filter=output_channel,
        kernel=(1, 1),
        stride=stride,
        pad=(0, 0),
        no_bias=True,
        name=name+'_conv',
        workspace=workspace)
    out = mx.sym.BatchNorm(
        data=out,
        fix_gamma=False,
        eps=2e-5,
        momentum=bn_mom,
        name=name+'_bn')
    if act:
        out = mx.sym.Activation(data=out, act_type='relu', name=name+'_relu')
    return out


def get_cls_head(data, pre_stage_channels, num_classes, dtype='float32'):
    head_channels = [128, 256, 512, 1024]

    output = residual_unit(
        data[0],
        head_channels[0],
        (1, 1),
        pre_stage_channels[0] == head_channels[0],
        name='cls_head_1')
    for i in range(1, len(pre_stage_channels)):
        output = residual_unit(data[i],
                               head_channels[i],
                               (1, 1),
                               pre_stage_channels[i] == head_channels[i],
                               name='cls_head_%d'%(i+1)) + conv3x3(output,
                                                                   head_channels[i],
                                                                   (2, 2),
                                                                   'cls_head_down%d'%(i+1))
    output = conv1x1(output, 2048, (1, 1), 'cls_head_final')
    output = mx.sym.Pooling(
        data=output,
        global_pool=True,
        kernel=(7, 7),
        pool_type='avg',
        name='global_pool')
    output = mx.sym.Flatten(data=output)
    output = mx.sym.FullyConnected(
        data=output,
        num_hidden=num_classes,
        name='fc')
    if dtype == 'float16':
        output = mx.sym.Cast(data=output, dtype=np.float32)
    output = mx.sym.SoftmaxOutput(data=output, name='softmax')
    return output


def get_symbol(num_classes,
               config,
               image_shape,
               conv_workspace=256,
               dtype='float32',
               **kwargs):
    '''
      Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
      Original author Wei Wu
      '''
    image_shape = [int(l) for l in image_shape.split(',')]
    (channels, height, width) = image_shape

    sym, channels = hrnet.get_symbol(config,
                                     image_shape,
                                     conv_workspace=conv_workspace,
                                     dtype=dtype)

    return get_cls_head(sym,
                        channels,
                        num_classes)


if __name__ == '__main__':
    sym = get_symbol(1000, 'w18', '3,224,224')
    vis = mx.viz.plot_network(sym)
    vis.render('hrnet_example')
