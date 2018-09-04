# PyTorch Pommerman

This is a PyTorch starting point for experimenting with ideas for the Pommerman competitions (https://www.pommerman.com/)

The reinforcement learning codebase is based upon Ilya Kostrikov's awesome work (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

It requires the Pommerman `playground` (https://github.com/MultiAgentLearning/playground) to be installed in your Python environment, in addition to any dependencies of `pytorch-a2c-ppo-acktr`.

## Usage

Very few experiments have been done so far, you can get a model training for FFA play, with poor results, using the following command:

`python main.py --use-gae --env-name PommeFFACompetitionFast-v0 --no-norm --seed 42`

