# CLAP: **C**uriosity-Driven Reinforcment **L**earning **A**utomatic **P**enetration Testing Agent

`CLAP` is a reinforcement learning [PPO agent](https://arxiv.org/abs/1707.06347) performs [**Penetration Testing**](https://en.wikipedia.org/wiki/Penetration_test) in simulated computer network environment. The agent is trained to scan for vulnerabilities in the network and exploit them to gain access to various network resources. `CLAP` was initially poposed in our paper [*Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach*](https://arxiv.org/abs/2202.10630). 

![](https://files.catbox.moe/784yxg.jpg)


However, this 

- Developed based on CleanRL:
- Add LSTM for POMDP
  - [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://proceedings.mlr.press/v162/ni22a.html)
- Support 2D observation space
  - Extremely unstable to train


## Prerequisites:

To run this code, you will need to have the following installed on your system:

- Python 3.5 or later
- Pytorch 2.0 or later
- OpenAI Gym 0.21 (huge change 0.25)
- NASim 0.91


## Get Started

Use `Conda` to manage `python` environmnent and `Poetry` to manage packages.
Get Started

Clone this repo:

```bash
git clone https://github.com/yyzpiero/evo.git
```

Create conda environment:
```bash
conda create -p ./venv python==X.X
```
and use poetry to install all Python packages:
```bash
poetry install
```
## Train the agent
To train the agent, you can use the following command:

```python
python train_agent.py
```

This will start the training process, which will run until the agent reaches a satisfactory level of performance. The performance of the agent will be printed to the console at regular intervals, so you can monitor its progress.

## Contributing

> The ppo implementation is heavily based on [Costa Huang's](https://costa.sh/) fantasitc library [CleanRl](https://github.com/vwxyzjn/cleanrl)

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Citing `CLAP`

```latex
@article{yang2022behaviour,
  title={Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach},
  author={Yang, Yizhou and Liu, Xin},
  journal={arXiv preprint arXiv:2202.10630},
  year={2022}
}
```

## TODOs

## Limitations
This implementation of the PPO algorithm is not intended for use in real-world penetration testing. It is only meant for use in a simulated environment, and should not be used to perform actual penetration testing on real networks.
