import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_generic import NNBase
import numpy as np


class ConvNet4(nn.Module):
    def __init__(self, input_shape, num_channels=64, output_size=512):
        super(ConvNet4, self).__init__()

        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.drop_prob = 0.2
        self.num_channels = num_channels
        self.output_size = output_size
        self.input_shape = input_shape
        self.flattened_size = num_channels * (input_shape[1] - 4) * (input_shape[2] - 4)

        self.conv1 = nn.Conv2d(input_shape[0], num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        #x = F.dropout(F.relu(self.fc1(x)), p=self.drop_prob, training=self.training)
        out = self.fc2(x)

        return out


class PommNet(NNBase):
    def __init__(self, obs_shape, image_shape, recurrent=False, hidden_size=512):
        super(PommNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.obs_shape = obs_shape
        assert len(image_shape) == 3
        self.image_shape = image_shape
        assert np.prod(obs_shape) >= np.prod(image_shape)
        self.other_shape = obs_shape - np.prod(image_shape)

        self.common_conv = ConvNet4(input_shape=self.image_shape, output_size=hidden_size)

        self.common_mlp = nn.Sequential(
            nn.Linear(self.other_shape[0], hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_size + hidden_size//4, hidden_size)

        self.critic = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//4, 1),
            nn.Tanh()
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        inputs_image = inputs[:, :-self.other_shape[0]].view([-1] + self.image_shape)
        inputs_other = inputs[:, -self.other_shape[0]:]

        x_conv = self.common_conv(inputs_image)
        x_mlp = self.common_mlp(inputs_other)
        x = torch.cat([x_conv, x_mlp], dim=1)
        #x = x_conv + x_mlp

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        out_actor = self.actor(x)
        out_value = self.critic(x)

        return out_value, out_actor, rnn_hxs


