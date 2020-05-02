# Deep Deterministic Policy Gradients

Deep deterministic policy gradients (DDPG) is an off-policy learning algorithm for continuous action spaces. It is an extension of deep Q-networks, where the selection of the maximum-value action is approximated using a neural network, rather than computed directly, due to the continuous action space.

## 🗂️ Directory Structure
* ``scripts/``
    * contains the models, loss functions, and other utilities (e.g. replay buffer) that are used in the DDPG algorithm.
* ``nb/``
    * contains the notebook with the implementation of the DDPG training routine, as well as code for testing an existing model.

## 🔗 Additional Resources
* [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html): Detailed explanation and high quality implementation from OpenAI of the DDPG algorithm.