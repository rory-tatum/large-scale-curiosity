import numpy as np
import torch

from curious.auxiliary_tasks import JustPixels
from curious.utils import small_convnet, flatten_two_dims, unflatten_first_dim, unet


class Dynamics(object):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None):
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=False)
        else:
            self.features = torch.stop_gradient(self.auxiliary_task.features)

        self.out_features = self.auxiliary_task.next_features

        self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = torch.nn.LeakyReLU
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = torch.shape(x)
            x = flatten_two_dims(x)
        x = (torch.cast(x, dtype=torch.float32) - self.ob_mean) / self.ob_std
        x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        ac = torch.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = torch.shape(ac)
        ac = flatten_two_dims(ac)

        def add_ac(x):
            return torch.concat([x, ac], axis=-1)

        x = flatten_two_dims(self.features)
        x = torch.nn.Linear(add_ac(x), self.hidsize, activation=torch.nn.leaky_relu)

        def residual(x):
            res = torch.nn.Linear(add_ac(x), self.hidsize, activation=torch.nn.leaky_relu)
            res = torch.nn.Linear(add_ac(res), self.hidsize, activation=None)
            return x + res

        for _ in range(4):
            x = residual(x)
        n_out_features = self.out_features.get_shape()[-1].value
        x = torch.nn.Linear(add_ac(x), n_out_features, activation=None)
        x = unflatten_first_dim(x, sh)
        return torch.reduce_mean((x - torch.stop_gradient(self.out_features)) ** 2, -1)

    def calculate_loss(self, ob, last_ob, acs):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        return np.concatenate([getsess().run(self.loss,
                                             {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)


class UNet(Dynamics):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = torch.nn.leaky_relu
        ac = torch.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = torch.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = torch.expand_dims(torch.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return torch.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = torch.shape(x)
                return torch.concat(
                    [x, ac_four_dim + torch.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], torch.float32)],
                    axis=-1)

        x = flatten_two_dims(self.features)
        x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
        x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean
        return torch.reduce_mean((x - torch.stop_gradient(self.out_features)) ** 2, [2, 3, 4])
