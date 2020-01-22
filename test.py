import warfarin.parse as MaxsUtil
import Subgraphs.lib as sublib
import networkx as nx
import matplotlib.pyplot as plt


def gen_test_graphs():
    G = sublib.graph_generation(r'Warfarin_PDB_Files/2225_warfarin.pdb')
    H = sublib.fragment(G,4,2)
    return H, G


if __name__ == '__main__':
    H, G = gen_test_graphs()
