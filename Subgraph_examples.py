import Subgraphs.lib as Util
import networkx as nx
import matplotlib.pyplot as pyp
import molecular_graphs.subgraph_isomorphism as sub_iso

if __name__ == '__main__':
    G = nx.path_graph(4)
    G.add_edge(0, 3)
    nx.draw_networkx(G, with_labels=True)
    nx.set_node_attributes(G, ['C'], 'label')
    pyp.show()
    H = nx.ladder_graph(5)
    nx.draw_networkx(H, with_labels=True)
    nx.set_node_attributes(H, ['C'], 'label')
    pyp.show()
    GM = sub_iso.get_isomorphisms(H, G)
    print(GM)
