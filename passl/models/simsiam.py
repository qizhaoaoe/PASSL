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
# Ref: https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

import os
import paddle
import paddle.nn as nn
from passl.models import Model
from passl.models import ResNet
from passl.models.resnet import BottleneckBlock, BasicBlock
from passl.nn.init import xavier_init, constant_, normal_init
import paddle.nn as nn
from passl.nn import init


__all__ = [
    'SimSiamLinearProbe',
    'simsiam_resnet50_linearprobe',
    'SimSiam',
]


class SimSiam(Model):
    """
    Build a SimSiam model.
    https://arxiv.org/abs/2011.10566
    """
    def __init__(self,
                 depth=50,
                 dim=2048,
                 hid_channels=512,
                 use_synch_bn=True
                ):
        """
        Args:
            backbone (dict): config of backbone.
            head (dict): config of head.
            predictor (dict): config of predictor.
            use_synch_bn (bool): whether apply apply sync bn.
        """
        super(SimSiam, self).__init__()

        # Create the encoder
        # number classes is the output fc dimension, zero-initialize last BNs
        self.encoder = ResNet(
            block=BottleneckBlock,
            depth=depth,
            with_pool=True,
            class_num=dim)
        
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[0]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias_attr=False),
                                        nn.BatchNorm1D(prev_dim),
                                        nn.ReLU(),
                                        nn.Linear(prev_dim, prev_dim, bias_attr=False),
                                        nn.BatchNorm1D(prev_dim),
                                        nn.ReLU(),
                                        self.encoder.fc,
                                        nn.BatchNorm1D(dim, weight_attr=False, bias_attr=False))
        self.encoder.fc[6].bias.stop_gradient = True
        
        self.predictor = nn.Sequential(nn.Linear(dim, hid_channels, bias_attr=False),
                                 nn.BatchNorm1D(hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, dim))
        self.criterion = nn.CosineSimilarity(axis=1)

        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
        # initialize parameters of encoder
        self.init_parameters()
        
    def init_parameters(self):
        for m in self.encoder.sublayers():
            if isinstance(m, nn.Conv2D):
                init.kaiming_init(m, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.layer.norm._BatchNormBase, nn.GroupNorm)):
                init.constant_init(m, 1)
        # from torch
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.encoder.sublayers():
            if isinstance(m, BottleneckBlock):
                init.constant_init(m.bn3, 0)
            elif isinstance(m, BasicBlock):
                init.constant_init(m.bn2, 0)
                    
    def train_iter(self, inputs):
        x1, x2 = inputs

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        # print("encoder z1: ", z1.detach().sum().cpu().numpy())
        z2 = self.encoder(x2)  # NxC
        # print("encoder z2: ", z2.detach().sum().cpu().numpy())

        p1 = self.predictor(z1)  # NxC
        # print("encoder p1: ", p1.detach().sum().cpu().numpy())

        p2 = self.predictor(z2)  # NxC
        # print("encoder p2: ", p2.detach().sum().cpu().numpy())
        outputs = dict()
        outputs['loss'] = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))

    def load_pretrained(self, path, rank=0, finetune=False):
        # load pretrained model
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
        if not finetune:
            self.set_dict(param_state_dict)

    def save(self, path, local_rank=0, rank=0):
        if local_rank == 0:
            paddle.save(self.state_dict(), path + ".pdparams")



class SimSiamLinearProbe(ResNet, Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.stop_gradient = True

        # optimize only the linear classifier
        parameters = list(
            filter(lambda p: not p.stop_gradient, self.parameters()))
        assert len(parameters) == 2  # weight, bias

        init.normal_(self.fc.weight, mean=0.0, std=0.01)
        init.zeros_(self.fc.bias)

        self.apply(self._freeze_norm)

    def _freeze_norm(self, layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True
    
    def load_pretrained(self, path, rank=0, finetune=False):
        # load pretrained model
        if not os.path.exists(path):
            raise ValueError("Model pretrain path {} does not "
                             "exists.".format(path))

        state_dict = self.state_dict()
        param_state_dict = paddle.load(path)
        if "state_dict" in param_state_dict:
            param_state_dict = param_state_dict['state_dict']
        new_state_dict = {}
        for key, value in param_state_dict.items():
            k = key.replace('encoder.', '')
            if k in state_dict:
                new_state_dict[k] = param_state_dict[key]
            else:
                print(key, ' not in current model')
        if not finetune:
            self.set_state_dict(new_state_dict)


def simsiam_resnet50_linearprobe(**kwargs):
    model = SimSiamLinearProbe(block=BottleneckBlock, depth=50, **kwargs)
    return model

def simsiam_resnet50_pretrain(**kwargs):
    model = SimSiam()
    return model

# if __name__ == '__main__':
#     model = SimSiam()
#     for name, param in model.named_parameters():
#         print(name, param.stop_gradient, param.detach().sum().cpu().numpy())