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

def hr_unit(data,
            num_branches,
            input_channels,
            output_channels,
            num_blocks,
            bottleneck,
            name,
            multi_scale_output):
    branches = []
    expansion = 4 if bottleneck else 1
    for i in range(num_branches):
        branch = residual_unit(
            data[i],
            output_channels[i] * expansion,
            (1, 1),
            input_channels[i] == output_channels[i] * expansion,
            name=name+"_branch%d_block%d" % (i + 1, 1),
            bottle_neck=bottleneck
        )
        input_channels[i] = output_channels[i] * expansion
        for j in range(1, num_blocks[i]):
            branch = residual_unit(
                branch,
                output_channels[i] * expansion,
                (1, 1),
                True,
                name=name+"_branch%d_block%d" % (i + 1, j + 1),
                bottle_neck=bottleneck
            )
        branches.append(branch)

    output = fuse_layer(branches,
                        input_channels,
                        num_branches,
                        multi_scale_output,
                        name)

    return output


def fuse_layer(data,
               input_channels,
               num_branches,
               multi_scale_output,
               name):
    outputs = []

    for i in range(num_branches if multi_scale_output else 1):
        output = data[0]
        if i != 0:
            for k in range(i):
                if k == i - 1:
                    output = conv3x3(output,
                                     input_channels[i],
                                     (2, 2),
                                     name+'_branch%d_down%d'%(i+1, k+1),
                                     False)
                else:
                    output = conv3x3(output,
                                     input_channels[0],
                                     (2, 2),
                                     name+'_branch%d_down%d'%(i+1, k+1),
                                     True)
        for j in range(1, num_branches):
            if i == j:
                output = output + data[j]
            elif i < j:
                output = output + mx.sym.UpSampling(conv1x1(data[j],
                                                            input_channels[i],
                                                            (1, 1),
                                                            name+'_branch%d_up%d'%(i+1, j+1)),
                                                    scale=2 ** (j - i),
                                                    sample_type='nearest')
            else:
                conv = data[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        conv = conv3x3(conv,
                                       input_channels[i],
                                       (2, 2),
                                       name+'_branch%d_down%d_conv%d'%(i+1, j+1, k+1),
                                       False)
                    else:
                        conv = conv3x3(conv,
                                       input_channels[j],
                                       (2, 2),
                                       name+'_branch%d_down%d_conv%d'%(i+1, j+1, k+1),
                                       True)
                output = output + conv
        output = mx.sym.Activation(data=output, act_type='relu', name=name+'_branch%d_relu'%(i+1))
        outputs.append(output)

    return outputs, input_channels


def hrnet(config, dtype):
    data = mx.sym.Variable(name='data')
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)

    stem1 = conv3x3(data, 64, (2, 2), 'stem1', True)
    stem2 = conv3x3(stem1, 64, (2, 2), 'stem2', True)

    stage1_cfg = config['stage1']
    expansion = 4 if stage1_cfg['block'] == 'BOTTLENECK' else 1
    stage1_output_channels = stage1_cfg['num_channels'][0] * expansion
    layer1 = residual_unit(
        stem2,
        stage1_output_channels,
        (1, 1),
        stage1_output_channels == 64,
        name='layer1_unit1',
        bottle_neck=stage1_cfg['block'] == 'BOTTLENECK')
    for i in range(1, stage1_cfg['num_blocks'][0]):
        layer1 = residual_unit(
            layer1,
            stage1_output_channels,
            (1, 1),
            True,
            name='layer1_unit%d' % (i+1),
            bottle_neck=stage1_cfg['block'] == 'BOTTLENECK')
    pre_stage_channels = [stage1_output_channels]
    output = [layer1]# * config['stage'][1]['num_branches']

    for i in range(1, 4):
        stage_cfg = config['stage'+str(i+1)]
        expansion = 4 if stage_cfg['block'] == 'BOTTLENECK' else 1
        num_channels = [stage_cfg['num_channels'][j] * expansion for j in range(len(stage_cfg['num_channels']))]
        output = transition_layer(output, pre_stage_channels, num_channels, 'layer%d_trans'%(i+1))
        output, pre_stage_channels = hr_stage(output, stage_cfg, num_channels, 'layer%d'%(i+1))

    return output, pre_stage_channels


def transition_layer(data, num_channels_pre_layer, num_channels_cur_layer, name):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    outputs = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                outputs.append(conv3x3(data[-1],
                                       num_channels_cur_layer[i],
                                       (1, 1),
                                       name+'_match_branch%d'%(i+1),
                                       True))
            else:
                outputs.append(data[i])
        else:
            output = data[-1]
            for j in range(i + 1 - num_branches_pre):
                if j == i - num_branches_pre:
                    output_channels = num_channels_cur_layer[i]
                else:
                    output_channels = num_channels_pre_layer[-1]
                output = conv3x3(output,
                                 output_channels,
                                 (2, 2),
                                 name+'_down%d_branch%d'%(j+1, i+1),
                                 True)
            outputs.append(output)
    return outputs


def hr_stage(data, config, input_channels, name, multi_scale_output=True):
    num_modules = config['num_modules']
    output = data
    for i in range(num_modules):
        if not multi_scale_output and i == num_modules - 1:
            reset_multi_scale_output = False
        else:
            reset_multi_scale_output = True

        output, input_channels = hr_unit(output,
                                         config['num_branches'],
                                         input_channels,
                                         config['num_channels'],
                                         config['num_blocks'],
                                         config['block'] == 'BOTTLENECK',
                                         name+'_unit%d'%(i+1),
                                         reset_multi_scale_output)
    return output, input_channels

def get_symbol(config,
               image_shape,
               conv_workspace=256,
               dtype='float32',
               **kwargs):
    '''
      Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
      Original author Wei Wu
      '''
    image_shape = [int(l) for l in image_shape.split(',')
                   ] if type(image_shape) is str else image_shape
    (channels, height, width) = image_shape

    if config == 'w18':
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
                fuse_method='SUM'),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36),
                fuse_method='SUM'),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
                fuse_method='SUM'),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                fuse_method='SUM'))
    elif config == 'w32':
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
                fuse_method='SUM'),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64),
                fuse_method='SUM'),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128),
                fuse_method='SUM'),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                fuse_method='SUM'))

    return hrnet(extra, dtype)


if __name__ == '__main__':
    sym = get_symbol(2, 101, '3,224,224')
    vis = mx.viz.plot_network(sym)
    vis.render('hrnet_example')
