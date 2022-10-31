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

![](https://files.catbox.moe/sd6j2i.png)


Therefore in `NASim`'s language, we could actually interprete th

|NASim|CyberRange|Subnet ID| Hosts IDs|
|----------|:-------------:|------:|------:|
| External Network| `kali` | 0| 0| 
| DMZ |  `pfSense` Router | 1 | 1|
| Sensitive Subnet | Isolated Network | 2 |2,3,4|
| User Subnet | Windows AD |    3 |5,6,7|



<details>
    <summary> Always visible. Can **ONLY** be plaintext </summary>
<!-- empty line -->
  Collapsible content (Markdown-stylable)

  ```bash
  subnets: [1, 3, 3]
  subnets: [1, 3, 3]
topology: [[ 1, 1, 0, 0, 0],
           [ 1, 1, 1, 1, 0],
           [ 0, 1, 1, 1, 0],
           [ 0, 1, 1, 1, 1],
           [ 0, 0, 0, 1, 1]]
sensitive_hosts:
  (2, 0): 100
  (4, 0): 100
os:
  - linux
  - windows
services:
  - ssh
  - ftp
  - http
processes:
  - tomcat
  - daclsvc
  ```
</details>
<!-- empty line -->


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