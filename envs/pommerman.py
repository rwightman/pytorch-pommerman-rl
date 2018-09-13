import pommerman
import pommerman.characters
import pommerman.envs
import numpy as np
import gym
import random


def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def featurize(obs, agent_id, replace_agents=True, replace_powerups=True):
    max_item = pommerman.constants.Item.Agent3.value

    ob = obs["board"]
    ob_bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32) / pommerman.constants.AGENT_VIEW_SIZE
    ob_bomb_life = obs["bomb_life"].astype(np.float32) / pommerman.constants.DEFAULT_BOMB_LIFE

    # one hot encode the board items
    ob_values = max_item + 1
    ob_hot = np.eye(ob_values)[ob]

    # replace agent item channels with friend, enemy, self channels
    if replace_agents:
        self_value = pommerman.constants.Item.Agent0.value + agent_id
        enemies = np.logical_and(ob >= pommerman.constants.Item.Agent0.value, ob != self_value)
        self = (ob == self_value)
        friends = (ob == pommerman.constants.Item.AgentDummy.value)
        ob_hot[:, :, 9] = friends.astype(np.float32)
        ob_hot[:, :, 10] = self.astype(np.float32)
        ob_hot[:, :, 11] = enemies.astype(np.float32)
        ob_hot = np.delete(ob_hot, np.s_[12::], axis=2)

    if replace_powerups:
        # replace powerups with single channel
        powerup = ob_hot[:, :, 6] * 0.5 + ob_hot[:, :, 7] * 0.66667 + ob_hot[:, :, 8]
        ob_hot[:, :, 6] = powerup
        ob_hot = np.delete(ob_hot, [7, 8], axis=2)

    # replace bomb item channel with life
    ob_hot[:, :, 3] = ob_bomb_life

    # insert bomb blast strength channel next to bomb life
    #ob_hot = np.insert(ob_hot, 4, ob_bomb_blast_strength, axis=2)
    ob_hot[:, :, 5] = ob_bomb_blast_strength

    if True:
        #ob_hot[:, :, 0] = ob_hot[:, :, 0] * 0.5 + ob_hot[:, :, 2] * 0.66667 + ob_hot[:, :, 1]
        ob_hot[:, :, 1] = 0.5 * ob_hot[:, :, 2] + ob_hot[:, :, 1]
        ob_hot = np.delete(ob_hot, [2], axis=2)

    self_ammo = make_np_float([obs["ammo"]])
    self_blast_strength = make_np_float([obs["blast_strength"]])
    self_can_kick = make_np_float([obs["can_kick"]])

    ob_hot = ob_hot.transpose((2, 0, 1))

    if True:
        def _rescale(x):
            #return x
            return (x - 0.5) * 2.0
        ob_hot = _rescale(ob_hot)
        self_ammo = _rescale(self_ammo / 10)
        self_blast_strength = _rescale(self_blast_strength / pommerman.constants.AGENT_VIEW_SIZE)
        self_can_kick = _rescale(self_can_kick)

    return np.concatenate([
        np.reshape(ob_hot, -1), self_ammo, self_blast_strength, self_can_kick])


class PommermanEnvWrapper(gym.Wrapper):
    def __init__(self, env=None, original_features=False):
        super(PommermanEnvWrapper, self).__init__(env)
        self._original_features = original_features
        if not self._original_features:
            self._set_observation_space(channels=9)

    def _set_observation_space(self, channels):
        bss = self.env._board_size**2
        min_obs = [0] * bss * channels + [0] * 3
        max_obs = [1.0] * bss * channels + [1.0] * 3
        self.observation_space = gym.spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def step(self, actions):
        obs = self.env.get_observations()
        all_actions = [actions]
        all_actions += self.env.act(obs)
        state, reward, done, _ = self.env.step(all_actions)
        if self._original_features:
            agent_state = self.env.featurize(state[self.env.training_agent])
        else:
            agent_state = featurize(
                state[self.env.training_agent],
                self.env.training_agent)
        agent_reward = reward[self.env.training_agent]
        return agent_state, agent_reward, done, {}

    def reset(self):
        obs = self.env.reset()
        if self._original_features:
            agent_obs = self.env.featurize(obs[self.env.training_agent])
        else:
            agent_obs = featurize(obs[self.env.training_agent], self.env.training_agent)
        return agent_obs


class TrainingAgent(pommerman.agents.BaseAgent):

    def __init__(self, character=pommerman.characters.Bomber):
        super(TrainingAgent, self).__init__(character)

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions."""
        return None


def _ffa_partial_env():
    """Start up a FFA config with the competition settings."""
    env = pommerman.envs.v0.Pomme
    game_type = pommerman.constants.GameType.FFA
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeFFACompetition-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': pommerman.constants.BOARD_SIZE,
        'num_rigid': pommerman.constants.NUM_RIGID,
        'num_wood': pommerman.constants.NUM_WOOD,
        'num_items': pommerman.constants.NUM_ITEMS,
        'max_steps': pommerman.constants.MAX_STEPS,
        'render_fps': pommerman.constants.RENDER_FPS,
        'agent_view_size': pommerman.constants.AGENT_VIEW_SIZE,
        'is_partially_observable': True,
        'env': env_entry_point,
    }
    agent = pommerman.characters.Bomber
    return locals()


def _ffa_partial_fast_env():
    """Start up a FFA config with the competition settings."""
    env = pommerman.envs.v0.Pomme
    game_type = pommerman.constants.GameType.FFA
    env_entry_point = 'pommerman.envs.v0:Pomme'
    env_id = 'PommeFFACompetitionFast-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': pommerman.constants.BOARD_SIZE,
        'num_rigid': pommerman.constants.NUM_RIGID,
        'num_wood': pommerman.constants.NUM_WOOD,
        'num_items': pommerman.constants.NUM_ITEMS,
        'max_steps': pommerman.constants.MAX_STEPS,
        'render_fps': 1000,
        'agent_view_size': pommerman.constants.AGENT_VIEW_SIZE,
        'is_partially_observable': True,
        'env': env_entry_point,
    }
    agent = pommerman.characters.Bomber
    return locals()


def make_env(config):
    training_agent = TrainingAgent()
    agent_list = [
        training_agent,
        pommerman.agents.SimpleAgent(),
        pommerman.agents.SimpleAgent(),
        pommerman.agents.SimpleAgent(),
    ]
    if config == "PommeFFAPartialFast-v0":
        env_spec = _ffa_partial_fast_env()
        env = pommerman.envs.v0.Pomme(**env_spec['env_kwargs'])
        for id, agent in enumerate(agent_list):
            assert isinstance(agent, pommerman.agents.BaseAgent)
            # NOTE: This is IMPORTANT so that the agent character is initialized
            agent.init_agent(id, env_spec['game_type'])
        env.set_agents(agent_list)
        #env.set_init_game_state(game_state_file)
        #env.set_render_mode(render_mode)
    else:
        env = pommerman.make(config, agent_list)
    env.set_training_agent(training_agent.agent_id)
    env.set_init_game_state(None)
    return PommermanEnvWrapper(env)