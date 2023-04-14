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

import os
import paddle
import paddle.nn as nn
from passl.models import Model
from passl.models import ResNet
from passl.nn.init import xavier_init, constant_


class NonLinearNeckV2(nn.Layer):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_bias=True,
                 with_avg_pool=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.mlp = nn.Sequential(nn.Linear(in_channels, hid_channels, bias_attr=with_bias),
                                 nn.BatchNorm1D(hid_channels), nn.ReLU(),
                                 nn.Linear(hid_channels, out_channels))

        # init_backbone_weight(self.mlp)
        # self.init_parameters()

    def init_parameters(self, init_linear='kaiming'):
        # _init_parameters(self, init_linear)
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm,
                                nn.SyncBatchNorm)):
                if m.weight is not None:
                    constant_(m.weight, 1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x.reshape([x.shape[0], -1]))


class SimSiamContrastiveHead(nn.Layer):
    """Head for simsiam contrastive learning."""
    def __init__(self):
        super(SimSiamContrastiveHead, self).__init__()
        self.criterion = nn.CosineSimilarity(axis=1)

    def forward(self, p1, p2, z1, z2):
        """Forward head.

        Args:
            p1 (Tensor): output of predictor1.
            p2 (Tensor): output of predictor2.
            z1 (Tensor): output of encoder1.
            z2 (Tensor): output of encoder2.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        outputs = dict()
        outputs['loss'] = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return outputs


class SimSiam(Model):
    """
    Build a SimSiam model.
    https://arxiv.org/abs/2011.10566
    """
    def __init__(self,
                 dim=2048,
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
            depth=50,
            with_pool=True,
            num_classes=2048,
            zero_init_residual=True)
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

        self.predictor = NonLinearNeckV2(
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_bias=False,
            with_avg_pool=False)
        self.head = SimSiamContrastiveHead()

        # Convert BatchNorm*d to SyncBatchNorm*d
        if use_synch_bn:
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.predictor = nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

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

        outputs = self.head(p1, p2, z1.detach(), z2.detach())
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


# if __name__ == '__main__':
#     seed = 2023
#     import numpy as np
#     np.random.seed(seed)
#     paddle.seed(seed)
#     model = SimSiam()
#     paddle.save(model.state_dict(), "ss_model.pd")
#     np.save("fc0_weight.npy", model.encoder.fc[0].weight.cpu().numpy())
#     data0 = paddle.to_tensor(np.random.random([1,3,224,224]).astype(np.float32)).cuda()
#     data1 = paddle.to_tensor(np.random.random([1,3,224,224]).astype(np.float32)).cuda()
#     out = model([data0, data1])
