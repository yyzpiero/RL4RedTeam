import numpy as np
import warnings
import random

from .action import ActionResult
from .utils import get_minimal_steps_to_goal, min_subnet_depth, draw_random_normal_int, AccessLevel

# column in topology adjacency matrix that represents connection between
# subnet and public
INTERNET = 0


class Network:
    """A computer network """

    def __init__(self, scenario):
        self.hosts = scenario.hosts
        self.host_num_map = scenario.host_num_map
        self.subnets = scenario.subnets
        self.topology = scenario.topology
        self.firewall = scenario.firewall
        self.address_space = scenario.address_space
        self.address_space_bounds = scenario.address_space_bounds
        self.sensitive_addresses = scenario.sensitive_addresses
        self.sensitive_hosts = scenario.sensitive_hosts

    def reset(self, state):
        """Reset the network state to initial state """
        next_state = state.copy()
        for host_addr in self.address_space:
            host = next_state.get_host(host_addr)
            host.running = True # ??? Is it okay to set to be true???
            host.compromised = False
            host.access = AccessLevel.NONE
            host.reachable = self.subnet_public(host_addr[0])
            host.discovered = host.reachable
        return next_state

    def perform_action(self, state, action):
        """Perform the given Action against the network.

        Arguments
        ---------
        state : State
            the current state
        action : Action
            the action to perform

        Returns
        -------
        State
            the state after the action is performed
        ActionObservation
            the result from the action
        """
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        next_state = state.copy()

        if action.is_noop():
            return next_state, ActionResult(True)
        
        if not state.host_running(action.target):
            result = ActionResult(False, 0.0, running_error=True)
            return next_state, result
        
        if not state.host_reachable(action.target) \
           or not state.host_discovered(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        has_req_permission = self.has_required_remote_permission(state, action)
        if action.is_remote() and not has_req_permission:
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        if action.is_exploit() \
           and not self.traffic_permitted(
                    state, action.target, action.service
           ):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        host_compromised = state.host_compromised(action.target)
        if action.is_privilege_escalation() and not host_compromised:
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if action.is_exploit() and host_compromised:
            # host already compromised so exploits don't fail due to randomness
            pass
        elif np.random.rand() > action.prob:
            return next_state, ActionResult(False, 0.0, undefined_error=True)

        if action.is_subnet_scan():
            return self._perform_subnet_scan(next_state, action)

        t_host = state.get_host(action.target)
        next_host_state, action_obs = t_host.perform_action(action)
        next_state.update_host(action.target, next_host_state)
        self._update(next_state, action, action_obs)
        return next_state, action_obs

    def _perform_subnet_scan(self, next_state, action):
        if not next_state.host_compromised(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if not next_state.host_has_access(action.target, action.req_access):
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        discovered = {}
        running = {}
        newly_discovered = {}
        discovery_reward = 0
        target_subnet = action.target[0]
        for h_addr in self.address_space:
            newly_discovered[h_addr] = False
            discovered[h_addr] = False
            if self.subnets_connected(target_subnet, h_addr[0]):
                host = next_state.get_host(h_addr)
                discovered[h_addr] = True
                running[h_addr] = host.running
                if not host.discovered:
                    newly_discovered[h_addr] = True
                    host.discovered = True
                    discovery_reward += host.discovery_value

        obs = ActionResult(
            True,
            discovery_reward,
            discovered=discovered,
            running=running,
            newly_discovered=newly_discovered
        )
        return next_state, obs

    def _update(self, state, action, action_obs):
        if action.is_exploit() and action_obs.success:
            self._update_reachable(state, action.target)

    def _update_reachable(self, state, compromised_addr):
        """Updates the reachable status of hosts on network, based on current
        state and newly exploited host
        """
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if state.host_reachable(addr):
                continue
            if self.subnets_connected(comp_subnet, addr[0]):
                state.set_host_reachable(addr)

    def get_sensitive_hosts(self):
        return self.sensitive_addresses

    def is_sensitive_host(self, host_address):
        return host_address in self.sensitive_addresses

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self.firewall[(src_subnet, dest_subnet)]

    def host_traffic_permitted(self, src_addr, dest_addr, service):
        dest_host = self.hosts[dest_addr]
        return dest_host.traffic_permitted(src_addr, service)

    def has_required_remote_permission(self, state, action):
        """Checks attacker has necessary permissions for remote action """
        if self.subnet_public(action.target[0]):
            return True

        for src_addr in self.address_space:
            if not state.host_compromised(src_addr):
                continue
            if action.is_scan() and \
               not self.subnets_connected(src_addr[0], action.target[0]):
                continue
            if action.is_exploit() and \
               not self.subnet_traffic_permitted(
                   src_addr[0], action.target[0], action.service
               ):
                continue
            if state.host_has_access(src_addr, action.req_access):
                return True
        return False

    def traffic_permitted(self, state, host_addr, service):
        """Checks whether the subnet and host firewalls permits traffic to a
        given host and service, based on current set of compromised hosts on
        network.
        """
        for src_addr in self.address_space:
            if not state.host_compromised(src_addr) and \
               not self.subnet_public(src_addr[0]):
                continue
            if not self.subnet_traffic_permitted(
                    src_addr[0], host_addr[0], service
            ):
                continue
            if self.host_traffic_permitted(src_addr, host_addr, service):
                return True
        return False

    def subnet_public(self, subnet):
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        return len(self.subnets)

    def all_sensitive_hosts_compromised(self, state):
        for host_addr in self.sensitive_addresses:
            if not state.host_has_access(host_addr, AccessLevel.ROOT):
                return False
        return True

    def get_total_sensitive_host_value(self):
        total = 0
        for host_value in self.sensitive_hosts.values():
            total += host_value
        return total

    def get_total_discovery_value(self):
        total = 0
        for host in self.hosts.values():
            total += host.discovery_value
        return total

    def get_minimal_steps(self):
        return get_minimal_steps_to_goal(
            self.topology, self.sensitive_addresses
        )

    def get_subnet_depths(self):
        return min_subnet_depth(self.topology)
    
    def get_num_running_hosts(self, state):
        """Return the number of runing hosts in the network"""
        total = 0
        for host_addr in self.address_space:
            total += state.get_host(host_addr).running
        return int(total)

    def get_num_inactive_hosts(self, state):
        """Return the number of runing hosts in the network"""
        total = 0
        for host_addr in self.address_space:
            total += state.get_host(host_addr).running
        return int(len(self.address_space) - total)
        
    
    def reset_hosts(self, address_list, state):
        """ Resetting hosts from a provided address list
        Args:
        ---------
            address_list (list) : A list of hosts address
            state (State) : The current state in the NASim Environment.

        Returns:
        ---------
            next_state (State) : The state after the defensive mechansim is performed 

        """
        # # copy the current state
        temp_state = state.copy()
        
        # Define the hosts' addresses to be reset
        if not isinstance(address_list, list):
            raise TypeError("Address list should be a list")
        elif len(address_list) == 0:
            warnings.warn("Nothing in the Address list")
            return temp_state

        # Iterate over the list of addresses.
        for host_addr in address_list:
            host = temp_state.get_host(host_addr)
            host.compromised = False
            host.access = AccessLevel.NONE
            host.reachable = self.subnet_public(host_addr[0]) # Only reachable if host in the public subnet??
            host.discovered = host.reachable
        return temp_state
        
    def reboot_hosts(self, address_list, state):
        """ Reboot hosts from a provided address list
        Args:
        ---------
            address_list (list) : A list of hosts address
            state (State) : The current state in the NASim Environment.

        Returns:
        ---------
            next_state (State) : The state after the defensive mechansim is performed 

        """
        # # copy the current state
        temp_state = state.copy()
        
        # Define the hosts' addresses to be reset
        if not isinstance(address_list, list):
            raise TypeError("Address list should be a list")
        elif len(address_list) == 0:
            warnings.warn("Nothing in the Address list")
            return temp_state

        # Iterate over the list of addresses.
        for host_addr in address_list:
            host = temp_state.get_host(host_addr)
            host.running = 1.0 if host.running == 0 else 0
            host.compromised = False
            #host.access = AccessLevel.NONE
            #host.reachable = self.subnet_public(host_addr[0]) # Only reachable if host in the public subnet??
            #host.discovered = host.reachable
        return temp_state

    def switch_single_hosts(self, address, state):
        """ Reboot a single hosts from a provided address 
        Args:
        ---------
            address_list (list) : A list of hosts address
            state (State) : The current state in the NASim Environment.

        Returns:
        ---------
            next_state (State) : The state after the defensive mechansim is performed 
        """
        # # copy the current state
        temp_state = state.copy()
        
        # Define the hosts' addresses to be reset
        if not isinstance(address, tuple):
            raise TypeError("Not Single Address")
        
        if len(address) !=2:
            warnings.warn("The input address is not in a correct format")

        # Iterate over the list of addresses.        
        host = temp_state.get_host(address)
        host.running = 1.0 if host.running == 0 else 0
        host.compromised = False
        #host.access = AccessLevel.NONE
        #host.reachable = self.subnet_public(host_addr[0]) # Only reachable if host in the public subnet??
        #host.discovered = host.reachable
        return temp_state
        
    def perform_defensive(self, state, def_type="reboot"):
        """Perform the defensive mechanism against the network.

        Arguments
        ---------
            state (State) : The current state in the NASim Environment.

        TODO:
            Defensive Level : High, Medium, Low
            Defensive Type: 1) Regular Maintance 2)
        
        Returns
        -------
            next_state (State) : The state after the defensive mechansim is performed

        """
        # testing
        #print(self.get_num_running_hosts(state))
        address_list = []
        if random.random() < 0.02:
            shutdown_num = draw_random_normal_int(low=0, high=1)
        else:
            shutdown_num = 0
            return state
        
        host_num = len(self.hosts)
        #print(host_num)
        #print(np.random.choice(host_num, shutdown_num, replace=False))
        
        idx = np.random.choice(host_num, shutdown_num, replace=False).astype(int)
        # print(idx)
        for i in idx:
            address_list.append(self.address_space[i])
        

        
        # copy the current state
        temp_state = state.copy()

        if def_type.lower() == "none":
            return final_state
        elif def_type.lower() == "reset":
            final_state = self.reset_hosts(address_list, temp_state)
        elif def_type.lower() == "reboot":
            final_state = self.reboot_hosts(address_list, temp_state)

        return final_state

    def perform_ctrl_defensive(self, step, state, def_type="reboot", off_limit=0.1, p_affect=0.2, p_def_opt=0.2):
        """Perform the controlled defensive mechanism against the network.
        NOTE: It is different from the random method, for controlled defensive operation:
                For a network with N total hosts
                20% of the total hosts are affected (turned on or off) at the operation step.
                Each affected host has a 20% chance of being turned off.
                    If it is not turned off, it is turned on.
                The total number of inactive hosts is less than 20%.

        Arguments
        ---------
            state (State) : The current state in the NASim Environment.

        Returns
        -------
            next_state (State) : The state after the defensive mechansim is performed

        """

        # copy the current state
        temp_state = state.copy()
        
        
        num_on_host = self.get_num_running_hosts(state=state)
        num_off_host = self.get_num_inactive_hosts(state=state)
        num_host = len(self.hosts)
        
        on_hosts_list = self.address_space.copy()
        # Calculate the number of hosts to turn off/on
        num_affected = int(num_host * p_affect)
        
        address_list = []

        if def_type == "reboot":
            # Randomly select number of hosts to turn off
            hosts_to_turn_off_on = random.sample(self.address_space, num_affected)

            for host in hosts_to_turn_off_on:
                # Generate a random number between 0 and 1
                rand_num = random.random()
                # Turn off the host if the random number is less than 0.5
                if rand_num < p_def_opt:
                    if num_off_host >= num_host * off_limit:
                        continue
                    if temp_state.get_host(host).running == True:
                        #on_hosts_list.remove(host)
                        #temp_state = self.switch_single_hosts(host, temp_state)
                        temp_state.switch_host(host)
                        num_off_host += 1
                        #print("Turning off host {} at time step {}".format(host, step))
                    
                # Turn on the host if the random number is greater than 0.5
                else:
                    #if host not in on_hosts_list:
                    if temp_state.get_host(host).running == False:
                        #temp_state = self.switch_single_hosts(host, temp_state)
                        temp_state.switch_host(host)
                        # on_hosts_list.append(host)
                        # on_hosts_list = sorted(on_hosts_list)
                        num_off_host -= 1
                        #print("Turning on host {} at time step {}".format(host, step))

            return temp_state            

        elif def_type == "reset":
            # Randomly select number of hosts to turn off
            hosts_to_reset = random.sample(self.address_space, num_affected)
            # Generate a random number between 0 and 1
            rand_num = random.random()
            # Turn off the host if the random number is less than 0.5
            if rand_num < p_def_opt:
                return self.reset_hosts(hosts_to_reset, temp_state)
            else:
                return temp_state
        


    def __str__(self):
        output = "\n--- Network ---\n"
        output += "Subnets: " + str(self.subnets) + "\n"
        output += "Topology:\n"
        for row in self.topology:
            output += f"\t{row}\n"
        output += "Sensitive hosts: \n"
        for addr, value in self.sensitive_hosts.items():
            output += f"\t{addr}: {value}\n"
        output += "Num_services: {self.scenario.num_services}\n"
        output += "Hosts:\n"
        for m in self.hosts.values():
            output += str(m) + "\n"
        output += "Firewall:\n"
        for c, a in self.firewall.items():
            output += f"\t{c}: {a}\n"
        return output
