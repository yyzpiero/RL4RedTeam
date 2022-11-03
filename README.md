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
    <summary> This is how we setup the `yaml` file </summary>
<!-- empty line -->
  In general, we ommit the `External Network` where `kali` attack machine is located.

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


In our NASim settings

｜ **What we have changed：** I personally believe that, exploits are deterministic 


Therefore，here we have 


# Optimal Pace

6,2,50,31,34

0-62 total 63 actions

SubnetScan: target=(1, 0), cost=1.00, prob=1.00, req_access=USER
Exploit: target=(1, 0), cost=2.00, prob=0.90, req_access=USER, os=None, service=http, access=1
Exploit: target=(2, 2), cost=3.00, prob=0.90, req_access=USER, os=linux, service=ssh, access=1
PrivilegeEscalation: target=(2, 2), cost=1.00, prob=1.00, req_access=USER, os=linux, process=tomcat, access=2
Exploit: target=(3, 1), cost=1.00, prob=0.60, req_access=USER, os=windows, service=ftp, access=2


The most trickiest part of this attack path is that, after


# "Huge" Reality Gap

- `NAsim` is not able to simulate router/firewall 
  - As a result, after exploit `host(1,0)` (which is suppose to be the `pfSense` router), the attacker `kali` agent can access 


# Problem with VMs(Boxs)

In our initial experiments, we performed PenTesting on two famous VulnHub CTF VMs:

- **Mr. Robots**
- **Metasploitable 2**

Following some well established *walkthroughs*.

<br/>

[Metasploitable Walkthrough 1](https://www.golinuxcloud.com/learn-hacking-using-metasploitable-2/)

[Apache Tomcat Exploitation](https://pentestlab.blog/2012/03/22/apache-tomcat-exploitation/)

### Pros&Cons of Those

![截图](https://files.catbox.moe/s51yq5.png)


==some human consiousiness/sentience==, which some part of PenTesting or CTF campain extremely hard to automate

### Next!

**Plus**, there are also some quick summaries 

[Easy Vulnhub Linux Machines](https://emaragkos.gr/recommended-machines/)


|VM Name|Feature|Walkthroughs|
|--|--|--|
|billu: b0x 2|Enumeration, Web, CVE, Common easy priv esc|https://www.hackingarticles.in/hack-the-billu-b0x-2-vm-boot-to-root/|
|TomGhost||https://steflan-security.com/tryhackme-tomghost-walkthrough/|

