__author__ = 'Callum Macfarlane'
__email__ = 'c.macfarlane@uqconnect.edu.au'

import warfarin.parse as parse
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Subgraphs.lib_old as MS
from math import ceil

plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg'


def debug_view_match(match_buffer_map_output, mol_graph_1, mol_graph_2):
    f, axes = plt.subplots(1, 2, figsize=(10 * (2), 10))
    # test case with W1, W2, '2225_warfarin_node_id6_lv5'
    mol_1_layout = nx.spring_layout(mol_graph_1)
    mol_1_labels = nx.get_node_attributes(mol_graph_1, 'label')
    mol_1_mapped_nodes = match_buffer_map_output['2225_warfarin_node_id6_lv5'][mol_graph_1.name][0]
    mol_1_unmapped_nodes_color = {i: "#c0c1c2" for i in mol_graph_1.nodes}
    mol_1_mapped_nodes_color = {i: 'r' for i in mol_1_mapped_nodes}
    nx.draw_networkx_labels(mol_graph_1, mol_1_layout, mol_1_labels, font_size=16, ax=axes[0])
    nx.draw_networkx_edges(mol_graph_1, mol_1_layout, ax=axes[0])
    nx.draw_networkx_nodes(mol_graph_1, mol_1_layout, ax=axes[0],
                           node_color=list({**mol_1_unmapped_nodes_color, **mol_1_mapped_nodes_color}.values()),
                           nodelist=list({**mol_1_unmapped_nodes_color, **mol_1_mapped_nodes_color}.keys()))

    mol_2_layout = nx.spring_layout(mol_graph_2)
    mol_2_labels = nx.get_node_attributes(mol_graph_2, 'label')
    mol_2_mapped_nodes = match_buffer_map_output['2225_warfarin_node_id6_lv5'][mol_graph_2.name][0]
    mol_2_unmapped_nodes_color = {i: "#c0c1c2" for i in mol_graph_2.nodes}
    mol_2_mapped_nodes_color = {i: 'r' for i in mol_2_mapped_nodes}
    nx.draw_networkx_labels(mol_graph_2, mol_2_layout, mol_2_labels, font_size=16, ax=axes[1])
    nx.draw_networkx_edges(mol_graph_2, mol_2_layout, ax=axes[0])
    nx.draw_networkx_nodes(mol_graph_2, mol_2_layout, ax=axes[0],
                           node_color=list({**mol_2_unmapped_nodes_color, **mol_2_mapped_nodes_color}.values()),
                           nodelist=list({**mol_2_unmapped_nodes_color, **mol_2_mapped_nodes_color}.keys()))

    plt.show()


def fix_graph_attr(G, label_src=None):
    """
    Changed from Maxwell's code
    :param G: <networkx.classes.graph.Graph>
    :param label_src:
    :return:
    """
    H = G.copy()
    nx.set_node_attributes(H, [], "edge_count")
    for H_node in H.nodes:
        H.nodes[H_node]["edge_count"] = len(H.edges(H_node))  # number of neighbours
    if label_src is not None:
        if type(label_src) is nx.Graph:
            # copy labels from other graph
            for H_node in H.nodes:
                H.nodes[H_node]["label"] = label_src.nodes[H_node]["label"]
    return H


def animate_mappings(mappings, target_graph, title='map_animation'):
    """
    Display an animated plot of the mappings between two graphs, only requires the larger and the mappings from the smaller
    Currently only works if one is a subgraph of the other
    :param mappings: <list <dict <int>:<int> >> e.g. [{0: 0, 1: 1, 2: 2}, {0: 0, 2: 1, 1: 2}] generated from
    get_isomorphisms
    :param target_graph: <networkx.classes.graph.Graph> "Larger" of the two graphs
    :return:
    """

    # TODO: properly assign graph plot to specific axis to allow for multiplot graph visualisations
    def update(it, pos, labels):
        ax.clear()
        target_graph_mapped_nodes = [key for key in mappings[it]]
        unmapped_nodes_color = {i: "#c0c1c2" for i in target_graph.nodes}
        mapped_nodes_color = {i: 'r' for i in target_graph_mapped_nodes}
        print(target_graph_mapped_nodes)
        nx.draw_networkx_labels(target_graph, pos, labels, font_size=16)
        nx.draw_networkx_edges(target_graph, pos)
        nx.draw_networkx_nodes(target_graph, pos,
                               node_color=list({**unmapped_nodes_color, **mapped_nodes_color}.values()),
                               nodelist=list({**unmapped_nodes_color, **mapped_nodes_color}.keys()))

    # mappings = mappings.copy() dont beleive this is needed as works without deepcopy
    fig, ax = plt.subplots(figsize=(10, 10))
    target_graph_position = nx.spring_layout(target_graph)
    labels = nx.get_node_attributes(target_graph, 'label')
    ani = animation.FuncAnimation(fig, update, fargs=(target_graph_position, labels,), frames=len(mappings),
                                  interval=750, repeat=True)
    ani.save(title + '.gif')


def animate_mappings_multi(mappings, graphs, reference_graph, title):
    """
    Does not compute any sort of mappings from fragment_similarity_..., must be already calculated
    Currently should only displays a single fragment of interest, must be pre-formatted to only have 1 fragment
    :param mappings:
    :param graphs:
    :param title:
    :return:
    """

    # I'm pretty certain there is a way to update just the node colours but currently the entire graphs are updated each round
    # also seeing as this is the third time this code is used might put it in a separate function
    def update(it, pos, labels, graph_axis):
        for graph in molecule_names:
            if it < len(graph_mappings[graph]):
                axes[graph_axis[graph]].clear()
                target_graph_mapped_nodes = [key for key in graph_mappings[graph][it]]
                unmapped_nodes_color = {i: "#c0c1c2" for i in graph_dict[graph].nodes}
                mapped_nodes_color = {i: 'r' for i in target_graph_mapped_nodes}
                # print(target_graph_mapped_nodes)
                nx.draw_networkx_labels(graph_dict[graph], pos[graph], labels[graph], font_size=16,
                                        ax=axes[graph_axis[graph]])
                nx.draw_networkx_edges(graph_dict[graph], pos[graph], ax=axes[graph_axis[graph]])
                nx.draw_networkx_nodes(graph_dict[graph], pos[graph],
                                       ax=axes[graph_axis[graph]],
                                       node_color=list({**unmapped_nodes_color, **mapped_nodes_color}.values()),
                                       nodelist=list({**unmapped_nodes_color, **mapped_nodes_color}.keys()))
                axes[graph_axis[graph]].set_title(graph)

    # First determine how many Molecules need to be displayed
    graph_dict = {x.name: x for x in graphs}
    molecule_names = set([map_dict['target'] for map_dict in mappings])
    molecule_number = len(molecule_names)
    # edge_length = ceil(molecule_number**0.5)
    f, axes = plt.subplots(1, molecule_number + 1, figsize=(
    10 * (molecule_number + 1), 10))  # Doesn't unpack axes (f, ax1, ax2 = ) as the axes length is unknown
    axes[0].set_title(reference_graph.name)
    pos_ref = nx.spring_layout(reference_graph)
    reference_mapped_nodes = [x for x in mappings[0]['mapping'][0].values()]
    unmapped_nodes_color = {i: "#c0c1c2" for i in reference_graph.nodes}
    mapped_nodes_color = {i: 'r' for i in reference_mapped_nodes}
    labels_ref = nx.get_node_attributes(reference_graph, 'label')
    # print(target_graph_mapped_nodes)
    nx.draw_networkx_labels(reference_graph, pos_ref, labels_ref, font_size=16, ax=axes[0])
    nx.draw_networkx_edges(reference_graph, pos_ref, ax=axes[0])
    nx.draw_networkx_nodes(reference_graph, pos_ref, ax=axes[0],
                           node_color=list({**unmapped_nodes_color, **mapped_nodes_color}.values()),
                           nodelist=list({**unmapped_nodes_color, **mapped_nodes_color}.keys()))

    graph_positions = {x.name: nx.spring_layout(x) for x in graphs}
    graph_labels = {x.name: nx.get_node_attributes(x, 'label') for x in graphs}
    graph_mappings = {x['target']: x['mapping'] for x in mappings}
    graph_axis = {x: i for x, i in zip(molecule_names, range(1, molecule_number + 1))}
    max_mappings = max(len(graph_mappings[x]) for x in molecule_names)

    ani = animation.FuncAnimation(f, update, fargs=(graph_positions, graph_labels, graph_axis,), frames=max_mappings,
                                  interval=1000, repeat=True)
    ani.save(title + '.gif')


def buffer_match_mappings(graph, title='buffer matches'):
    labels = nx.get_node_attributes(graph, 'label')
    matches_colour = {i: ("#00f208" if (graph.nodes[i]['buffer_match'] == True) else "#c0c1c2") for i in graph.nodes}
    nx.draw_spring(graph, node_color=list(matches_colour.values()), labels=labels)
    plt.show()


# def match_buffer_map_view(match_buffer_map_output, graphs):
#     return None
#     def update(it, graph_dict_util, max_mappings):
#         for entry in match_buffer_map_output:
#             if it < max_mappings:
#                 axes[graph_dict_util['reference_name']['axis']].clear()
#                 axes[graph_dict_util['target_name']['axis']].clear()
#                 target_graph_mapped_nodes = [key for key in match_buffer_map_output[graph][it]]
#                 unmapped_nodes_color = {i: "#c0c1c2" for i in graph_dict[graph].nodes}
#                 mapped_nodes_color = {i: 'r' for i in target_graph_mapped_nodes}
#                 # print(target_graph_mapped_nodes)
#                 nx.draw_networkx_labels(graph_dict[graph], pos[graph], labels[graph], font_size=16,
#                                         ax=axes[graph_axis[graph]])
#                 nx.draw_networkx_edges(graph_dict[graph], pos[graph], ax=axes[graph_axis[graph]])
#                 nx.draw_networkx_nodes(graph_dict[graph], pos[graph],
#                                        ax=axes[graph_axis[graph]],
#                                        node_color=list({**unmapped_nodes_color, **mapped_nodes_color}.values()),
#                                        nodelist=list({**unmapped_nodes_color, **mapped_nodes_color}.keys()))
#                 axes[graph_axis[graph]].set_title(graph)
#
#
#     graph_dict_util = {x.name:{'graph':x, 'layout': nx.spring_layout(x)}  for x in graphs}
#     fragments_number = len(match_buffer_map_output)
#     # max_mappings = # TODO: IMPORTANT for now just cap the iterator
#     max_mappings = 4
#     f, axes = plt.subplots(1, fragments_number + 1, figsize=(10 * (fragments_number + 1), 10))
#
#     for key, i in zip(graph_dict_util,range(len(graph_dict_util)):
#         graph_dict_util[key]['axis'] = axes[i]
#
#     ani = animation.FuncAnimation(f, update, fargs=(graph_dict_util, max_mappings,), frames=max_mappings,
#                                   interval=1000, repeat=True)
#
#
#
#
# if __name__ == "__main__":
#     G = MS.pdb_to_graph('Subgraphs/Warfarin_PDB_Files/2202_warfarin.pdb')
#     g = fix_graph_attr(G)
#     H = MS.pdb_to_graph('Subgraphs/Warfarin_PDB_Files/2222_warfarin.pdb')
#     H = fix_graph_attr(H)
#
#     maps, subs = parse.iso_compare(G, H, 5)
#
#     print(maps)


