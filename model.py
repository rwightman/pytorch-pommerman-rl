import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import get_distribution


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        All classes that inheret from Policy are expected to have
        a feature exctractor for actor and critic (see examples below)
        and modules called linear_critic and dist. Where linear_critic
        takes critic features and maps them to value and dist
        represents a distribution of actions.        
        """
        
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, hidden_actor, states = self(inputs, states, masks)
        
        action = self.dist.sample(hidden_actor, deterministic=deterministic)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor, action)
        
        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):        
        value, _, _ = self(inputs, states, masks)
        return value
    
    def evaluate_actions(self, inputs, states, masks, actions):
        value, hidden_actor, states = self(inputs, states, masks)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor, actions)
        
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(Policy):
    def __init__(self, num_inputs, action_space, use_gru):
        super(CNNPolicy, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU()
        )
        
        if use_gru:
            self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        self.dist = get_distribution(512, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        def mult_gain(m):
            relu_gain = nn.init.calculate_gain('relu')
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.mul_(relu_gain)
    
        self.main.apply(mult_gain)

        if hasattr(self, 'gru'):
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)


        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return self.critic_linear(x), x, states


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(Policy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.critic_linear = nn.Linear(64, 1)
        self.dist = get_distribution(64, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
    
    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor, states
