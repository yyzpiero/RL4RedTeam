# CLAP

CLAP is a Python library for dealing with word pluralization.

From our paper [Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach](https://arxiv.org/abs/2202.10630)

## Installation

Use `Conda` to manage `python` environmnent and `Poetry` to manage packages.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

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