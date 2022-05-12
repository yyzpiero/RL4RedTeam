## DQN-Agents
Modularized training of different DQN Algorithms.
> Modified From this repo: https://github.com/BY571/DQN-Atari-Agents
> This repository contains several Add-ons to the base DQN Algorithm. 
> With **multiprocessing to run several environments in parallel** for faster training.  

Following DQN versions are included:

- DDQN
- Dueling DDQN

Both can be enhanced with **Noisy layer**, **Per** (Prioritized Experience Replay), **Multistep Targets** and be trained in a **Categorical version (C51)**. Combining all these add-ons will lead to the *state-of-the-art* Algorithm of value-based methods called: **Rainbow**. 


**Please Note:** Rainbow is not currently compatible with **NASIM**.

[] Test ICM if posssible

To train the base DDQN simply run ``python run_atari_dqn.py``
To train and modify your own Atari Agent the following inputs are optional:

*example:* ``python run_atari_dqn.py -env BreakoutNoFrameskip-v4 -agent dueling -u 1 -eps_frames 100000 -seed 42 -info Breakout_run1``
- agent: Specify which type of DQN agent you want to train, default is DQN - baseline! **Following agent inputs are currently possible:** ``dqn``, ``dqn+per``, ``noisy_dqn``, ``noisy_dqn+per``, ``dueling``, ``dueling+per``, ``noisy_dueling``, ``noisy_dueling+per``, ``c51``, ``c51+per``, ``noisy_c51``, ``noisy_c51+per``, ``duelingc51``, ``duelingc51+per``, ``noisy_duelingc51``, ``noisy_duelingc51+per``, ``rainbow``
