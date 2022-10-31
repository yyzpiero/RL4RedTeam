# Conda and Poetry configuration
```bash
conda create -n demo_env python==3.9
```


```bash
git submodule init
```


```bash
poetry install
```

```bash
poetry add --editable ./NetworkAttackSimulator
```

# Network Topology Configuration
Firstly，the network archetecture of our cyberrange on `192.167.253.190` consists of **4 Different Parts**:

![](https://pic1.zhimg.com/80/v2-9fb9539ce5b99a9a6a84ff99b9091c30_1440w.webp)

- Kali Attacker
- Firewall/Router (which is actually a host)
- Isolated Network
  - 3 Hosts
- `Windows` Active Domain
  - 1 Domain Controller
  - 2 Hosts

However, in [`NAsim`](https://networkattacksimulator.readthedocs.io/en/latest/reference/scenarios/benchmark_scenarios.html), network topology is described differently. 


### Generators details

- **num_services** : `int`
  - number of services running on network (minimum is 1)
- **num_os** : `int`, optional
  - number of OS running on network (minimum is 1) (default=2)
- **num_processes** : `int`, optional
  - number of processes running on hosts on network (minimum is 1)(default=2)
- **num_exploits** : `int`, optional
  - number of exploits to use. minimum is 1. If None will use num_services (default=None)
- **num_privescs**: `int`, optional
  - number of privilege escalation actions to use. minimum is 1. If None will use num_processes (default=None)