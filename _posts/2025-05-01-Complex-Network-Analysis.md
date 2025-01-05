---
layout: post
title: How to build a network and detect communities using gephi (Part 1)
tags: ["NetworkX", "Gephi", "Wikipedia Pages", "Community Detection", "Snowball Sampling"]
mathjax: true
---

<!-- ## Network Analysis: How to build a network and detect communities using gephi -->
In this post, we will discuss the concept of community detection in network analysis. Community detection is a fundamental concept in network analysis that involves identifying groups of nodes that are more densely connected to each other than to the rest of the network. These groups are known as communities or clusters, and they represent substructures within the network that exhibit a higher degree of internal connectivity. We will explore a case study of community detection in a network of Wikipedia pages to illustrate the concept in practice.

### Case study: Network of Wikipedia Pages

To demonstrate community detection in practice, we will consider a network of Wikipedia pages related to data science topics. Each node in the network represents a Wikipedia page, and edges between nodes indicate hyperlinks between the corresponding pages. The goal is to identify communities of pages that are thematically related based on their hyperlink structure.

##### Part I. Creation of the Network using Snowball Sampling

We will start by creating a network of Wikipedia pages using snowball sampling. 

Snowball sampling is a technique that involves iteratively expanding a network by adding nodes and edges based on a set of initial seed nodes. In this case, we will select a single seed node corresponding to the word *"data science"*  and expand the network by adding pages that are linked to the seed nodes.

**A. Importing Libraries and Setting Seed and Stop Words**
```python
from operator import itemgetter 
import networkx as nx
import wikipedia
SEED = "Data Science".title()

STOPS = ("International Standard Serial Number", "International Standard Book Number",
"National Diet Library",
"International Standard Name Identifier", "International Standard Book Number (Identifier)", "Pubmed Identifier", "Pubmed Central",
"Digital Object Identifier", "Arxiv",
"Proc Natl Acad Sci Usa", "Bibcode",
"Library Of Congress Control Number", "Jstor")
```

**B. Concept of Layers and Expansion of the Network**

- Starting with seed(Layer 0), we will expand the network by adding nodes and edges iteratively based on the hyperlinks in the Wikipedia pages.

- Therefore, Layer 1 will contain say $N_1$ nodes, Layer 2 will contain $N_1 \times N_2$ nodes, and so on.

- As you can see, for Layer K, the number of nodes will be $N_1 \times N_2 \times \ldots \times N_K$.

- Total number of nodes in the network will be $N(K) = N_0 + N_1 + N_1 \times N_2 + \ldots + N_1 \times N_2 \times \ldots \times N_K$. 
  - Where K is the number of layers.

- The network will grow exponentially with the number of layers. Hence for our case study, we will limit the number of layers to 2.

**C. Algorithm for Snowball Sampling**
```python
todo_lst = [(0, SEED)] # The SEED is in the layer 0 
todo_set = set(SEED) # The SEED itself
done_set = set() # Nothing is done yet

while layer < 2:
# Block 1: Remove current page from todo_lst and mark as processed
 # Check if todo_lst is empty
    if not todo_lst:
        break
    del todo_lst[0]
    done_set.add(page)
    print(layer, page)  # Show progress
    
    # Block 2: Attempt to download the page
    try:
        wiki = wikipedia.page(page)
    except:
        layer, page = todo_lst[0]  # Get next page if download fails
        print("Could not load", page)
        continue
    
    # Block 3: Evaluate and process links
    for link in wiki.links:
        link = link.title()
        if link not in STOPS and not link.startswith("List Of"):
            if link not in todo_set and link not in done_set:
                todo_lst.append((layer + 1, link))
                todo_set.add(link)
            F.add_edge(page, link)
    
    # Block 4: Get the next page from todo_lst
    layer, page = todo_lst[0]

print("{} nodes, {} edges".format(len(F), nx.number_of_edges(F)))
```
Explanation:

1.**Initialization of the todo_lst, todo_set, and done_set**
Why do we need a todo_lst and todo_set?
   1. todo_lst: It is a list of tuples where each tuple contains the layer number and the page name. The todo_lst is used to keep track of the pages that need to be processed in the current layer.
   2. todo_set: It is a set that contains the names of the pages that need to be processed. The todo_set is used to efficiently check if a page is already in the todo_lst or done_set.
2. The while loop iterates until the specified number of layers is reached or the todo_lst is empty.
3. **Block 1:** Remove the current page from the todo_lst and mark it as processed by adding it to the done_set.
4. **Block 2:** Attempt to download the page using the Wikipedia API. If the download fails, move to the next page in the todo_lst.
5. **Block 3:** Evaluate and process the links on the downloaded page. For each link, check if it is a stop word or starts with "List Of". If not, add it to the todo_lst and todo_set and create an edge between the current page and the linked page in the network.

**D. Result of Snowball Sampling**
```
print("{} nodes, {} edges".format(len(F), nx.number_of_edges(F)))
```
Output:
```
21318 nodes, 37891 edges
```
As we can see, the snowball sampling process has resulted in a network with 21,318 nodes and 37,891 edges. Even with a limited number of layers, the network has grown significantly in size.


##### Part II. Graph Truncation and Community Detection using Gephi

In this part, we will truncate the graph to focus on a subset of nodes and edges for community detection using Gephi.

**A. Removing Self-Loops and Merging Duplicate Nodes**
```python
## Removing self loops
loops = list(nx.selfloop_edges(F, data=True, keys=True))
F.remove_edges_from(loops) 

## Merging duplicate nodes in the graph 
"""Usage: nx.contracted_nodes(graph_object, tuple_of_dup_nodes, self_loops=False)
"""
duplicates = [(node, node + "s") for node in F if node + "s" in F] 
for dup in duplicates:
    F = nx.contracted_nodes(F, *dup, self_loops=False) 
duplicates = [(x, y) for x, y in [(node, node.replace("-", " ")) for node in F] if x != y and y in F] 
for dup in duplicates:
    F = nx.contracted_nodes(F, *dup, self_loops=False) 
    
## nx.contracted_nodes creates a new node attribute called "contraction" whose value is a dictionary
## GrapML does not support dictionary attributes - hence we set the "contraction" to 0
nx.set_node_attributes(F, "contraction", 0)
```

**B. Truncation of Graph**
```python
# Create subgraph first
core = [node for node, deg in F.degree() if deg >= 2]
G = nx.subgraph(F, core)

# Remove all dictionary attributes from nodes
for node in G.nodes():
    node_attrs = dict(G.nodes[node])
    for key, value in node_attrs.items():
        if isinstance(value, dict):
            del G.nodes[node][key]

# Remove all dictionary attributes from edges
for u, v, data in G.edges(data=True):
    edge_attrs = dict(data)
    for key, value in edge_attrs.items():
        if isinstance(value, dict):
            del G.edges[u, v][key]

print("{} nodes, {} edges".format(len(G), nx.number_of_edges(G)))

# Save using the basic GraphML writer
nx.write_graphml(G, "cnads.graphml")
```

**C.Visualization and Community Detection using Gephi**

Now that we saved our graph as a GraphML file, we can import it into Gephi for visualization and community detection.

1. Open Gephi and create a new project.
2. Go to File > Open and select the "cnads.graphml" file.
3. Use the layout algorithms in Gephi to visualize the network. For example, you can use the ForceAtlas2 layout for a good initial layout. 
4. Use the modularity statistic in Gephi to detect communities in the network. Go to the Statistics tab, select Modularity, and click Run.
5. Apply the modularity partition to the network by going to the Partition tab, selecting Modularity, and clicking Apply.
6. Use the color palette in Gephi to visualize the communities in the network.
7. Explore the network to identify thematic clusters and analyze the structure of the communities.

In our case study of the Wikipedia pages related to data science topics, community detection can help identify groups of pages that are thematically related - Here is a snapshot of the network visualization in Gephi:

![Wikipedia Pages Network starting with seed "Data Science"][image-id]

[image-id]: ../files/NetworkAnalysis/cna_ds_wiki_with_eccentricity_nodes.png

**Observations:**

1. The network visualization shows **13 distinct communities** for different pages associated with the word "Data Science".

2. The **communities represent thematic clusters of pages** that are more densely connected to each other than to the rest of the network.
   1. For example, the community in the top left corner represents pages related to "Machine Learning" and "Artificial Intelligence".
   2. The Dark Green community in the bottom right corner represents pages related to "Data Analysis" and "Data Visualization".

3. The bigger nodes represent the pages with **higher eccentricity values**, indicating their **ability to connect cross communities**.