# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.vision.models as models
from paddle.vision.models.resnet import BasicBlock, BottleneckBlock
from passl.utils import logger
from passl.nn import init, freeze


# class ResNet(models.ResNet):
#     """ResNet model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         Block (BasicBlock|BottleneckBlock): block module of model.
#         depth (int): layers of resnet, default: 50.
#         num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
#                             will not be defined. Default: 1000.
#         with_pool (bool): use pool before the last fc layer or not. Default: True.

#     Examples:
#         .. code-block:: python

#             from paddle.vision.models import ResNet
#             from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

#             resnet50 = ResNet(BottleneckBlock, 50)

#             resnet18 = ResNet(BasicBlock, 18)

#     """
#     def __init__(self,
#                  depth,
#                  num_classes=0,
#                  with_pool=False,
#                  zero_init_residual=False,
#                  frozen_stages=-1,
#                  pretrained=None):

#         block = BasicBlock if depth in [18, 34] else BottleneckBlock

#         super(ResNet, self).__init__(block, depth, num_classes=num_classes, with_pool=with_pool)
#         self.zero_init_residual = zero_init_residual
#         self.frozen_stages = frozen_stages
#         self.init_parameters()

#         if pretrained is not None:
#             state_dict = paddle.load(pretrained)
#             if 'state_dict' in state_dict:
#                 state_dict = state_dict['state_dict']

#             self.set_state_dict(state_dict)
#             logger.info(
#                 'Load pretrained backbone weight from {} success!'.format(
#                     pretrained))

#         self._freeze_stages()

#     def init_parameters(self):
#         for m in self.sublayers():
#             if isinstance(m, nn.Conv2D):
#                 init.kaiming_init(m, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.layer.norm._BatchNormBase, nn.GroupNorm)):
#                 init.constant_init(m, 1)

#         if self.zero_init_residual:
#             for m in self.sublayers():
#                 if isinstance(m, BottleneckBlock):
#                     init.constant_init(m.bn3, 0)
#                 elif isinstance(m, BasicBlock):
#                     init.constant_init(m.bn2, 0)

#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             freeze.freeze_batchnorm_statictis(self.bn1)
#             for m in [self.conv1, self.bn1]:
#                 for param in m.parameters():
#                     param.trainable = False

#         for i in range(1, self.frozen_stages + 1):
#             m = getattr(self, 'layer{}'.format(i))
#             freeze.freeze_batchnorm_statictis(m)
#             for param in m.parameters():
#                 param.trainable = False

#         if self.frozen_stages >= 0:
#             logger.info(
#                 'Frozen layer before stage {}'.format(self.frozen_stages + 1))
# =======
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle

from paddle.vision.models.resnet import ResNet as PDResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

from passl.models.base_model import Model

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext50_64x4d",
    "resnext101_32x4d",
    "resnext101_64x4d",
    "resnext152_32x4d",
    "resnext152_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

class ResNet(PDResNet, Model):
    def __init__(
        self,
        block,
        depth=50,
        width=64,
        class_num=1000,
        with_pool=True,
        groups=1,
    ):
        super().__init__(block, depth=depth, width=width, num_classes=class_num, with_pool=with_pool, groups=groups)

    def load_pretrained(self, path, rank=0, finetune=False):
        if not os.path.exists(path + '.pdparams'):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(path + ".pdparams")

        # for FP16 saving pretrained weight
        for key, value in param_state_dict.items():
            if key in param_state_dict and key in state_dict and param_state_dict[
                    key].dtype != state_dict[key].dtype:
                param_state_dict[key] = param_state_dict[key].astype(
                    state_dict[key].dtype)

        self.set_dict(param_state_dict)

    def save(self, path, local_rank=0, rank=0):
        paddle.save(self.state_dict(), path + ".pdparams")

def resnet18(**kwargs):
    """ResNet 18-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    model = ResNet(BasicBlock, 18, **kwargs)
    return model

def resnet34(**kwargs):
    """ResNet 34-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    model = ResNet(BasicBlock, 34, **kwargs)
    return model

def resnet50(**kwargs):
    """ResNet 50-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    model = ResNet(BottleneckBlock, 50, **kwargs)
    return model


def resnet101(**kwargs):
    """ResNet 101-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    model = ResNet(BottleneckBlock, 101, **kwargs)
    return model

def resnet152(**kwargs):
    """ResNet 152-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """

    model = ResNet(BottleneckBlock, 152, **kwargs)
    return model


def resnext50_32x4d(**kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """

    kwargs['groups'] = 32
    kwargs['width'] = 4
    model = ResNet(BottleneckBlock, 50, **kwargs)
    return model

def resnext50_64x4d(**kwargs):
    """ResNeXt-50 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """

    kwargs['groups'] = 64
    kwargs['width'] = 4
    model = ResNet(BottleneckBlock, 50, **kwargs)
    return model

def resnext101_32x4d(**kwargs):
    """ResNeXt-101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """

    kwargs['groups'] = 32
    kwargs['width'] = 4
    model = ResNet(BottleneckBlock, 101, **kwargs)
    return model

def resnext101_64x4d(**kwargs):
    """ResNeXt-101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """

    kwargs['groups'] = 64
    kwargs['width'] = 4
    model = ResNet(BottleneckBlock, 101, **kwargs)
    return model


def resnext152_32x4d(**kwargs):
    """ResNeXt-152 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """

    kwargs['groups'] = 32
    kwargs['width'] = 4
    model = ResNet(BottleneckBlock, 152, **kwargs)
    return model

def resnext152_64x4d(**kwargs):
    """ResNeXt-152 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """

    kwargs['groups'] = 64
    kwargs['width'] = 4
    model = ResNet(BottleneckBlock, 152, **kwargs)
    return model

def wide_resnet50_2(**kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    """

    kwargs['width'] = 64 * 2
    model = ResNet(BottleneckBlock, 50, **kwargs)
    return model

def wide_resnet101_2(**kwargs):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    """

    kwargs['width'] = 64 * 2
    model = ResNet(BottleneckBlock, 101, **kwargs)
    return model
# >>>>>>> simsiam_lp
