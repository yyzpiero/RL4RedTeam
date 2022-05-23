# Copyright (C) 2013 Brian Wesley Baugh
# CSCE 6933: Social Network Analysis
# Created: January 22, 2013
# Updated: January 30, 2013
"""Generate a randomly connected graph with N nodes and E edges."""
import random
import argparse
import numpy as np
from pprint import pprint
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx


class Graph():
    def __init__(self, nodes, edges=None, loops=False, multigraph=False,
                 digraph=False):
        self.nodes = nodes
        if edges:
            self.edges = edges
        else:
            self.edges = []
            self.edge_set = set()
        self.loops = loops
        self.multigraph = multigraph
        self.digraph = digraph
    
    @property
    def size(self):
        return len(self.nodes)

    def add_edge(self, edge):
        """Add the edge if the graph type allows it."""
        if self.multigraph or edge not in self.edge_set:
            self.edges.append(edge)
            self.edge_set.add(edge)
            if edge[1] not in self.nodes:
                self.nodes.append(edge[1])
            if edge[0] not in self.nodes:
                self.nodes.append(edge[0])
            if not self.digraph:
                self.edge_set.add(edge[::-1])  # add other direction to set.
            return True
        return False

    def add_edges(self, edges):
        """Add the edges if the graph type allows it."""
        for edge in edges:
            self.add_edge(edge)

    def make_random_edge(self):
        """Generate a random edge between any two nodes in the graph."""
        if self.loops:
            # With replacement.
            random_edge = (random.choice(self.nodes), random.choice(self.nodes))
        else:
            # Without replacement.
            random_edge = tuple(random.sample(self.nodes, 2))
        return random_edge

    def add_random_edges(self, total_edges):
        """Add random edges until the number of desired edges is reached."""
        while len(self.edges) < total_edges:
            self.add_edge(self.make_random_edge())

    def sort_edges(self):
        """If undirected, sort order that the nodes are listed in the edge."""
        if not self.digraph:
            self.edges = [((t, s) if t < s else (s, t)) for s, t in self.edges]
        self.edges.sort()

    def generate_gml(self):
        # Inspiration:
        # http://networkx.lanl.gov/_modules/networkx/readwrite/gml.html#generate_gml
        indent = ' ' * 4

        yield 'graph ['
        if self.digraph:
            yield indent + 'directed 1'

        # Write nodes
        for index, node in enumerate(self.nodes):
            yield indent + 'node ['
            yield indent * 2 + 'id {}'.format(index)
            yield indent * 2 + 'label "{}"'.format(str(node))
            yield indent + ']'

        # Write edges
        for source, target in self.edges:
            yield indent + 'edge ['
            yield indent * 2 + 'source {}'.format(self.nodes.index(source))
            yield indent * 2 + 'target {}'.format(self.nodes.index(target))
            yield indent + ']'

        yield ']'

    def write_gml(self, fname):
        with open(fname, mode='w') as f:
            for line in self.generate_gml():
                line += '\n'
                f.write(line)


def check_num_edges(nodes, num_edges, loops, multigraph, digraph):
    """Checks that the number of requested edges is acceptable."""
    num_nodes = len(nodes)
    # Check min edges
    min_edges = num_nodes - 1
    if num_edges < min_edges:
        raise ValueError('num_edges less than minimum (%i)' % min_edges)
    # Check max edges
    max_edges = num_nodes * (num_nodes - 1)
    if not digraph:
        max_edges /= 2
    if loops:
        max_edges += num_nodes
    if not multigraph and num_edges > max_edges:
            raise ValueError('num_edges greater than maximum (%i)' % max_edges)


def naive(nodes, num_edges, loops=False, multigraph=False, digraph=False):
    # Idea:
    # Each node starts off in its own component.
    # Keep track of the components, combining them when an edge merges two.
    # While there are less edges than requested:
    #     Randomly select two nodes, and create an edge between them.
    # If there is more than one component remaining, repeat the process.

    check_num_edges(nodes, num_edges, loops, multigraph, digraph)

    def update_components(components, edge):
        # Update the component list.
        comp_index = [None] * 2
        for index, comp in enumerate(components):
            for i in (0, 1):
                if edge[i] in comp:
                    comp_index[i] = index
            # Break early once we have found both sets.
            if all(x is not None for x in comp_index):
                break
        # Combine components if the nodes aren't already in the same one.
        if comp_index[0] != comp_index[1]:
            components[comp_index[0]] |= components[comp_index[1]]
            del components[comp_index[1]]

    finished = False
    while not finished:
        graph = Graph(nodes, loops=loops, multigraph=multigraph, digraph=digraph)
        # Start with each node in its own component.
        components = [set([x]) for x in nodes]
        while len(graph.edges) < num_edges:
            # Generate a random edge.
            edge = graph.make_random_edge()
            if graph.add_edge(edge):
                # Update the component list.
                update_components(components, edge)
        if len(components) == 1:
            finished = True

    return graph


def partition(nodes, num_edges, loops=False, multigraph=False, digraph=False):
    # Algorithm inspiration:
    # http://stackoverflow.com/questions/2041517/random-simple-connected-graph-generation-with-given-sparseness

    # Idea:
    # Create a random connected graph by adding edges between nodes from
    # different partitions.
    # Add random edges until the number of desired edges is reached.

    check_num_edges(nodes, num_edges, loops, multigraph, digraph)

    graph = Graph(nodes, loops=loops, multigraph=multigraph, digraph=digraph)

    # Create two partitions, S and T. Initially store all nodes in S.
    S, T = set(nodes), set()

    # Randomly select a first node, and place it in T.
    node_s = random.sample(S, 1).pop()
    S.remove(node_s)
    T.add(node_s)

    # Create a random connected graph.
    while S:
        # Select random node from S, and another in T.
        node_s, node_t = random.sample(S, 1).pop(), random.sample(T, 1).pop()
        # Create an edge between the nodes, and move the node from S to T.
        edge = (node_s, node_t)
        assert graph.add_edge(edge) == True
        S.remove(node_s)
        T.add(node_s)

    # Add random edges until the number of desired edges is reached.
    graph.add_random_edges(num_edges)

    return graph


def random_walk(nodes, num_edges, loops=False, multigraph=False, digraph=False):
    # Algorithm inspiration:
    # https://en.wikipedia.org/wiki/Uniform_spanning_tree#The_uniform_spanning_tree

    # Idea:
    # Create a uniform spanning tree (UST) using a random walk.
    # Add random edges until the number of desired edges is reached.

    check_num_edges(nodes, num_edges, loops, multigraph, digraph)

    # Create two partitions, S and T. Initially store all nodes in S.
    S, T = set(nodes), set()

    # Pick a random node, and mark it as visited and the current node.
    current_node = random.sample([*S], 1).pop()
    S.remove(current_node)
    T.add(current_node)

    graph = Graph(nodes, loops=loops, multigraph=multigraph, digraph=digraph)

    # Create a random connected graph.
    while S:
        # Randomly pick the next node from the neighbors of the current node.
        # As we are generating a connected graph, we assume a complete graph.
        neighbor_node = random.sample(nodes, 1).pop()
        # If the new node hasn't been visited, add the edge from current to new.
        if neighbor_node not in T:
            edge = (current_node, neighbor_node)
            graph.add_edge(edge)
            S.remove(neighbor_node)
            T.add(neighbor_node)
        # Set the new node as the current node.
        current_node = neighbor_node

    # Add random edges until the number of desired edges is reached.
    graph.add_random_edges(num_edges)

    return graph


def wilsons_algo(nodes, num_edges, loops=False, multigraph=False, digraph=False):
    # Algorithm inspiration:
    # https://en.wikipedia.org/wiki/Uniform_spanning_tree#The_uniform_spanning_tree

    # Idea:
    # Create a uniform spanning tree (UST) using Wilson's algorithm:
    #     Start with two random vertices.
    #     Perform a (loop-erased) random walk between the two nodes.
    #     While there are still nodes not in the tree:
    #         Pick a random node not in the tree.
    #         Perform a random walk from this node until hitting the tree.
    # Add random edges until the number of desired edges is reached.

    check_num_edges(nodes, num_edges, loops, multigraph, digraph)

    graph = Graph(nodes, loops=loops, multigraph=multigraph, digraph=digraph)

    raise NotImplementedError()

def draw_random_normal_int(low:int, high:int):

    # generate a random normal number (float)
    normal = np.random.normal(loc=0, scale=1, size=1)

    # clip to -3, 3 (where the bell with mean 0 and std 1 is very close to zero
    normal = -3 if normal < -3 else normal
    normal = 3 if normal > 3 else normal

    # scale range of 6 (-3..3) to range of low-high
    scaling_factor = (high-low) / 6
    normal_scaled = normal * scaling_factor

    # center around mean of range of low high
    normal_scaled += low + (high-low)/2

    # then round and return
    return np.round(normal_scaled)

def random_group_assgin(n_nodes=100, average_group_size=10, bias=2):
    node_list = [x for x in range(int(n_nodes))]
    group_list = []
    done_assign = False
    while not done_assign:
        if bias!=0:
            group_size = int(draw_random_normal_int(average_group_size-bias, average_group_size+bias))
        else:
            group_size = average_group_size
        #print(group_size)
        if len(node_list) >= group_size:
            assigned = random.sample(node_list, group_size)
            group_list.append(assigned)
            #print(assigned)
            for x in assigned:
                node_list.remove(x) 
        else:
            if node_list:
                group_list.append(node_list)
            break
    
    return len(group_list), group_list



def subnet_topology(n_nodes, subnet_size="small", domain_size="small", loops=True, multigraph=False, digraph=True, return_matrix=False):
    
    '''
    Let Fisrt Use "Small Network" Settings, where:
    domain_size: [5, 10]
    stream_size: [5, 10]
    backbone_size: [1, 5]
    
    Therefore, n_user_subnets should be in: [5x5x1=25, 10x10x5=500] 
    backbone nodes, domain_controller nodes are subnets but only has one host
    
    Default Example
    100 Hosts,  
    '''
    avg_subnet_size = 2
    _bias = 1
    n_user_subnets, subnets_assign = random_group_assgin(n_nodes, avg_subnet_size, bias=_bias)

    avg_domain_size = random.randint(1, 3)

    n_domains, domains_assign =  random_group_assgin(n_user_subnets, avg_domain_size, bias=2)

    N_DOMAIN_PER_STREAM = 2
    n_backbone_streams, streams_assign =  random_group_assgin(n_domains, N_DOMAIN_PER_STREAM, bias=1) 

    n_backbone_edges = n_backbone_streams + 1 
    backbone_nodes = [x+1 for x in range(int( n_backbone_streams))]
    backbone_graph = random_walk(backbone_nodes, n_backbone_edges, loops, multigraph, digraph)
   
    n_vn_backbone_nodes = backbone_graph.size
    
    # Add Intital Foothold on Randomly choosen DMZ Router
    backbone_graph.add_edge((0,random.randint(1, backbone_graph.size)))
    
    domain_graph = Graph(nodes=backbone_graph.nodes.copy(), edges=backbone_graph.edges.copy(), loops=loops, multigraph=multigraph, digraph=digraph)
    domain_graph.edge_set = backbone_graph.edge_set#.copy()
    domain_graph.sort_edges()
    source_domain_id = backbone_graph.size

    for vn_idx in range(1, n_vn_backbone_nodes+1):

        for _ in streams_assign[vn_idx-1]:
            domain_graph.add_edge((vn_idx, source_domain_id))
            source_domain_id +=1
   
    total_vn_nodes = domain_graph.size
    total_graph = Graph(nodes=domain_graph.nodes.copy(), edges=domain_graph.edges.copy(), loops=loops, multigraph=multigraph, digraph=digraph)
    total_graph.edge_set = domain_graph.edge_set#.copy()
    total_graph.sort_edges()

    for i, dm_idx in enumerate(range(backbone_graph.size, total_graph.size)):

        subnet_nodes = [x+total_vn_nodes for x in domains_assign[i]]
        subnet_graph_edges = random_walk(subnet_nodes, len(domains_assign[i]), loops, multigraph, digraph).edges
        total_graph.add_edges(subnet_graph_edges)
        total_graph.add_edge((dm_idx, random.choice(subnet_nodes)))

    total_graph.sort_edges()
    pprint(total_graph.edges)
    total_graph.n_vitrual_nodes =  domain_graph.size
    total_graph.n_subnets = len(total_graph.nodes)-total_vn_nodes
    assert total_graph.n_subnets == len(subnets_assign)
    print("Number of Backbone Streams:", backbone_graph.size - 1)
    print("Number of Domains:", domain_graph.size - backbone_graph.size)
    print("Number of Subnets:", len(total_graph.nodes)-total_vn_nodes)

    if return_matrix:
        topology_matrix = np.eye(len(total_graph.nodes), dtype=int)
        for i, j in total_graph.edges:
            topology_matrix[i][j] = 1
        return total_graph, subnets_assign, topology_matrix

    return total_graph, subnets_assign


def hosts_topology_from_subnet(subnet_graph, n_nodes, subnets_assign=None, n_subnets=None, return_vec=False, return_matrix=False):
    
    n_control_nodes = subnet_graph.size
    n_user_host = n_nodes - n_control_nodes
    if not subnets_assign:
        n_subnets, subnets_assign = random_group_assgin(n_user_host, avg_subnet_size=4, bias=1)
    else:
        assert n_subnets is None, "When Subnet Assignment is given, n_subnet is not required"

        n_subnets = subnet_graph.n_subnets
        assert (n_subnets == len(subnets_assign)), "Wrong Number of Subnets"

            
    for idx, host_idx_vec in enumerate(subnets_assign):
        subnet_idx = idx + n_subnets
        for host_idx in host_idx_vec:
            host_idx = host_idx + n_control_nodes
            subnet_graph.add_edge((subnet_idx, host_idx))

    if return_vec:
        
        subnet_vec = []
        for _ in range(subnet_graph.n_vitrual_nodes): 
            subnet_vec.append(1)
        for i, idx in enumerate(range(subnet_graph.n_vitrual_nodes, n_control_nodes)):
            subnet_vec.append(len(subnets_assign[i]))
        if not return_matrix:
            return subnet_graph, subnet_vec
        else:
            topology_matrix = np.eye(len(subnet_graph.nodes), dtype=int)
            for i, j in subnet_graph.edges:
                topology_matrix[i][j] = 1
            return subnet_graph, subnet_vec, topology_matrix

    return subnet_graph
   

def plot_topology(topology_matrix, filename="./fig.png"):

    G = nx.Graph(topology_matrix) 
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.kamada_kawai_layout(G)

    nx.draw(G, pos=pos, node_size=100, with_labels=True)
    plt.axis('equal')
    plt.savefig(filename)


def nasim_toplogy_subnet(n_nodes):
    subnet_graph, subnets_assign, topology= subnet_topology(n_nodes=n_nodes, return_matrix=True)
    _, subnet = hosts_topology_from_subnet(subnet_graph, n_nodes=n_nodes, subnets_assign=subnets_assign, n_subnets=None, return_vec=True)
    return subnet, topology


if __name__ == '__main__':
    n_nodes = 10

    subnet_graph, subnets_assign, subnet_topology_matrix = subnet_topology(n_nodes=n_nodes, return_matrix=True)
    subnet_graph, subnet_vec, host_topology_matrix = hosts_topology_from_subnet(subnet_graph, n_nodes=n_nodes, subnets_assign=subnets_assign, n_subnets=None, return_vec=True, return_matrix=True)
    plot_topology(subnet_topology_matrix)
