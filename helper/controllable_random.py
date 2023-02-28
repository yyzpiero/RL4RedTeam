import random
random.seed = 23232
# Number of hosts
N = 1000
# Number of steps
T = 1000

# Initialize the list of hosts
hosts = [i for i in range(N)]
print(hosts)
hosts_on = [i for i in range(N)]
# # Initialize the list of hosts
# hosts = [1, 2, 3, ... , N]
# Initialize the number of hosts that are turned off
num_hosts_off = 0

# Calculate the number of hosts to turn off/on
num_off_on = int(N * 0.2)

# Iterate over each step
for t in range(T):
    # Randomly select number of hosts to turn off
    hosts_to_turn_off_on = random.sample(hosts, num_off_on)

    # Turn off/on the selected hosts
    for host in hosts_to_turn_off_on:
        
        # Generate a random number between 0 and 1
        rand_num = random.random()

        # Turn off the host if the random number is less than 0.5
        if rand_num < 0.2:
            if num_hosts_off >= N * 0.2:
                continue
            if host in hosts_on:
                hosts_on.remove(host)
                num_hosts_off += 1
                #print("Turning off host {} at time step {}".format(host, t))
                
        # Turn on the host if the random number is greater than 0.5
        else:
           
            if host not in hosts_on:
                hosts_on.append(host)
                hosts_on = sorted(hosts_on)
                num_hosts_off -= 1
                print("Turning on host {} at time step {}".format(host, t))
        # if num_hosts_off >= N * 0.2:
        #     break

    # Check the list of on hosts
    # print("List of on hosts at step {}: {}".format(t, hosts_on))
    print("Number of offlined host at step {}: {}".format(t, num_hosts_off))

   
    