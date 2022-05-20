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

   #faster solution



class Graph():
    def __init__(self, nodes, edges=None, loops=False, multigraph=False,
                 digraph=False):
        self.nodes = nodes
        if edges:
            self.edges = edges
            #self.edge_set = self._compute_edge_set()
        else:
            self.edges = []
            self.edge_set = set()
        self.loops = loops
        self.multigraph = multigraph
        self.digraph = digraph
    
    @property
    def size(self):
        return len(self.nodes)


    def _compute_edge_set(self):
        raise NotImplementedError()

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
                f.write(line.encode('latin-1'))


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
    current_node = random.sample(S, 1).pop()
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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('nodes',
#                         help='filename containing node labels (one per line) '
#                              'OR integer number of nodes to generate')
#     parser.add_argument('-e', '--edges', type=int,
#                         help='number of edges (default is minimum possible)')
#     parser.add_argument('-l', '--loops', action='store_true',
#                         help='allow self-loop edges')
#     parser.add_argument('-m', '--multigraph', action='store_true',
#                         help='allow parallel edges between nodes')
#     parser.add_argument('-d', '--digraph', action='store_true',
#                         help='make edges unidirectional')
#     parser.add_argument('-w', '--wilson', action='store_const',
#                         const='wilsons_algo', dest='approach',
#                         help="use wilson's generation algorithm (best)")
#     parser.add_argument('-r', '--random-walk', action='store_const',
#                         const='random_walk', dest='approach',
#                         help='use a random-walk generation algorithm (default)')
#     parser.add_argument('-n', '--naive', action='store_const',
#                         const='naive', dest='approach',
#                         help='use a naive generation algorithm (slower)')
#     parser.add_argument('-t', '--partition', action='store_const',
#                         const='partition', dest='approach',
#                         help='use a partition-based generation algorithm (biased)')
#     parser.add_argument('--no-output', action='store_true',
#                         help='do not display any output')
#     parser.add_argument('-p', '--pretty', action='store_true',
#                         help='print large graphs with each edge on a new line')
#     parser.add_argument('-g', '--gml',  default=False,
#                         help='filename to save the graph to in GML format')
#     args = parser.parse_args()

#     # Nodes
#     try:
#         nodes = []
#         with open(args.nodes) as f:
#             for line in f:
#                 nodes.append(line.strip())
#     except IOError:
#         try:
#             nodes = [x for x in range(int(args.nodes))]
#         except ValueError:
#             raise TypeError('nodes argument must be a filename or an integer')

#     # Edges
#     if args.edges is None:
#         num_edges = len(nodes) - 1
#     else:
#         num_edges = args.edges

#     # Approach
#     if args.approach:
#         print('Setting approach:', args.approach)
#         approach = locals()[args.approach]
#     else:
#         approach = random_walk

#     # Run
#     graph = approach(nodes, num_edges, args.loops, args.multigraph,
#                      args.digraph)

#     # Display
#     if not args.no_output:
#         graph.sort_edges()
#         if args.pretty:
#             pprint(graph.edges)
#         else:
#             print(graph.edges)

#     # Save to GML
#     if args.gml:
#         graph.write_gml(args.gml)
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



def subnet_topology(n_nodes, subnet_size="small", domain_size="small", loops=True, multigraph=False, digraph=True):
    
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
    avg_subnet_size = 4
    _bias = 2
    n_user_subnets, subnets_assign = random_group_assgin(n_nodes, avg_subnet_size, bias=_bias)

    
    # print("Number of User SubNets:", n_user_subnets)
    avg_n_domain = random.randint(5, 10)

    n_domains, domains_assign =  random_group_assgin(n_user_subnets, avg_n_domain, bias=_bias)
    # print("Number of Domains:", n_domains)
    print(domains_assign)

    N_DOMAIN_PER_STREAM = 2
    n_backbone_streams, streams_assign =  random_group_assgin(n_domains, N_DOMAIN_PER_STREAM, bias=0)
    #n_backbone_nodes = np.ceil((n_domain)/N_DOMAIN_PER_STREAM)
    print(streams_assign)
    

    n_backbone_edges = n_backbone_streams + 1 

    backbone_nodes = [x+1 for x in range(int( n_backbone_streams))]

    backbone_graph = random_walk(backbone_nodes, n_backbone_edges, loops, multigraph, digraph)

    backbone_graph.sort_edges()
    
   
    n_vn_backbone_nodes = backbone_graph.size
    
    # print("Number of Backbone Streams:", backbone_graph.size)
    #print("VN for Backbone Streams:", vn_backbone_nodes)
    #print("Number of Subnet Routers:", (n_user_subnet/n_subnets_per_routers))
     # Add Intital Foothold on Randomly choosen DMZ Router
    backbone_graph.add_edge((0,random.randint(1, backbone_graph.size)))
    backbone_graph.nodes.insert(0,0)
    #pprint(backbone_graph.edge_set)

    domain_graph = Graph(nodes=backbone_graph.nodes.copy(), edges=backbone_graph.edges.copy(), loops=loops, multigraph=multigraph, digraph=digraph)
    domain_graph.edge_set = backbone_graph.edge_set#.copy()
    domain_graph.sort_edges()

    source_domain_id = backbone_graph.size
    #print(source_domain_id)
    for vn_idx in range(1, n_vn_backbone_nodes+1):
        #print(streams_assign[vn_idx])
        #print(len(n_vn_backbone_nodes))
        for _ in streams_assign[vn_idx-1]:
            #print(vn_idx)
            domain_graph.add_edge((vn_idx, source_domain_id))
            source_domain_id +=1


    

    #pprint(backbone_graph.size)
    total_vn_nodes = domain_graph.size
    # print("Number of VN:", total_vn_nodes)
    total_graph = Graph(nodes=domain_graph.nodes.copy(), edges=domain_graph.edges.copy(), loops=loops, multigraph=multigraph, digraph=digraph)
    total_graph.edge_set = domain_graph.edge_set#.copy()
    total_graph.sort_edges()
    #print(total_graph.edges)
    l = 0
    
    for dm_idx in range(backbone_graph.size, total_graph.size):
        #print(dm_idx)
        subnet_nodes = [x+total_vn_nodes for x in domains_assign[l]]
        subnet_graph_edges = random_walk(subnet_nodes, len(domains_assign[l]), loops, multigraph, digraph).edges
        
            # print(subnets_idx)
            # subnets_idx = [x+total_vn_nodes for x in subnets_idx]
            # print(subnets_idx)
        print(subnet_nodes)
        total_graph.add_edges(subnet_graph_edges)
        total_graph.add_edge((dm_idx, random.choice(subnet_nodes)))
        l +=1
    #total_graph.sort_edges()
    pprint(total_graph.edges)
    # print("Number of Backbone Streams:", backbone_graph.size-1)
    # print("Number of Domains:", domain_graph.size - backbone_graph.size)
    # print("Number of Subnets:", len(total_graph.nodes)-total_vn_nodes)

if __name__ == '__main__':
    #n_nodes = 40
    #g,g = random_group_assgin(150, 15, 5)
    #print(g)
    subnet_topology(n_nodes=100)