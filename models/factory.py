from .model_pomm import PommNet
from .model_generic import CNNBase, MLPBase
from .policy import Policy


def create_policy(obs_space, action_space, name='basic', nn_kwargs={}):
    nn = None
    obs_shape = obs_space.shape
    if name.lower() == 'basic':
        if len(obs_shape) == 3:
            nn = CNNBase(obs_shape[0], **nn_kwargs)
        elif len(obs_shape) == 1:
            nn = MLPBase(obs_shape[0], **nn_kwargs)
        else:
            raise NotImplementedError
    elif name.lower() == 'pomm':
        nn = PommNet(
            obs_shape=obs_shape,
            **nn_kwargs)
    else:
        assert False and "Invalid policy name"

    policy = Policy(nn, action_space=action_space)

    return policy
