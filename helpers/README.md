# OpenAI Baselines Helpers

This module contains the useful utility functions from OpenAI baselines (https://github.com/openai/baselines) including:
* Atari environment wrappers and pre-processing
* Vectored subprocess environments
* monitoring and logging utilities

The motivation for factoring this code out and including it here is to avoid a dependency on the full Baselines codebase, especially Tensorflow.
