# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:18:47 2020

@author: Nicholas Sotiriou - github: @nsotiriou88 // nsotiriou88@gmail.com
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import copy, itertools, json, flask


# Importing the dataset
# Grab edge list data hosted on Gist
edgelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/e570c38bcc72a8d102422f2af836513b/raw/89c76b2563dbc0e88384719a35cba0dfc04cd522/edgelist_sleeping_giant.csv')
# Preview edgelist
edgelist.head(10)

# Grab node list data hosted on Gist
nodelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/f989e10af17fb4c85b11409fea47895b/raw/a3a8da0fa5b094f1ca9d82e1642b384889ae16e8/nodelist_sleeping_giant.csv')
# Preview nodelist
nodelist.head(5)

# Create empty graph
g = nx.Graph()

# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())
# Edge list example
print(elrow[0]) # node1
print(elrow[1]) # node2
print(elrow[2:].to_dict()) # edge attribute dict



# Add node attributes
for i, nlrow in nodelist.iterrows():
    g.nodes[nlrow['id']].update(nlrow[1:].to_dict())
# Node list example
print(nlrow)


# Preview first 5 edges
list(g.edges(data=True))[0:5]

# Preview first 10 nodes
list(g.nodes(data=True))[0:10]

print('# of edges: {}'.format(g.number_of_edges()))
print('# of nodes: {}'.format(g.number_of_nodes()))


# =============================================================================
#%% Starting Creating Plotting options
# =============================================================================
# Define node positions data structure (dict) for plotting
node_positions = {node[0]: (node[1]['X'], -node[1]['Y']) for node in g.nodes(data=True)}
# Preview of node_positions with a bit of hack (there is no head/slice method for dictionaries).
dict(list(node_positions.items())[0:5])

# Define data structure (list) of edge colors for plotting
edge_colors = [e[2]['attr_dict']['color'] for e in list(g.edges(data=True))]
# Preview first 10
edge_colors[0:10]

plt.figure(figsize=(8, 6))
axes = plt.subplot(1,1,1)
nx.draw(g, pos=node_positions, ax=axes, edge_color=edge_colors, node_size=10, node_color='black')
#nx.draw_spectral(g, edge_color=edge_colors, node_size=10, node_color='black')
#nx.draw_spring(g, edge_color=edge_colors, node_size=10, node_color='black')
#nx.draw_kamada_kawai(g, edge_color=edge_colors, node_size=10, node_color='black')
#nx.draw_planar(g, edge_color=edge_colors, node_size=10, node_color='black')
#nx.draw_shell(g, edge_color=edge_colors, node_size=10, node_color='black')
plt.title('Graph Representation of Sleeping Giant Trail Map', size=15)
plt.show()


# =============================================================================
#%% Calculate Graph Stats
# =============================================================================
# Calculate list of nodes with odd degree
nodes_odd_degree = [v for v, d in g.degree() if d % 2 == 1]
# Preview
nodes_odd_degree[0:5]

# Counts
print('Number of nodes of odd degree: {}'.format(len(nodes_odd_degree)))
print('Number of total nodes: {}'.format(len(g.nodes())))


# Compute all pairs of odd nodes. in a list of tuples
odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))

# Preview pairs of odd degree nodes
odd_node_pairs[0:10]

print('Number of pairs: {}'.format(len(odd_node_pairs)))



def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.
    Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances


# Compute shortest paths.  Return a dictionary with node pairs keys and a single value equal to shortest path distance.
odd_node_pairs_shortest_paths = get_shortest_paths_distances(g, odd_node_pairs, 'distance')
# Preview with a bit of hack (there is no head/slice method for dictionaries).
dict(list(odd_node_pairs_shortest_paths.items())[0:10])

'''
import json

import flask
import networkx as nx
from networkx.readwrite import json_graph

G = nx.barbell_graph(6, 3)
# this d3 example uses the name attribute for the mouse-hover value,
# so add a name to each node
for n in G:
    G.nodes[n]['name'] = n
# write json formatted data
d = json_graph.node_link_data(G)  # node-link format to serialize
# write json
json.dump(d, open('force/force.json', 'w'))
print('Wrote node-link JSON data to force/force.json')

# Serve the file over http to allow for cross origin requests
app = flask.Flask(__name__, static_folder="force/")

@app.route('/')
def static_proxy():
    return app.send_static_file('force.html')

print('\nGo to http://localhost:8000 to see the example\n')
app.run(port=8686)
'''

