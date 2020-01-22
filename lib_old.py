__author__ = 'Callum Macfarlane'


# If highest depth match is found dont revisit.

# TODO: Break symmetry based on bond Types
# TODO: Add functionality to node match from other data types. - Might have to change PDB Parse tool
# TODO: Maye also add qualities to edges
# TODO: Comment node matching methods

import time
import os
import warnings
import matplotlib.pyplot as plt  # Used for plotting graphs
import networkx as nx  # Used for generating graph objects that can be plotted
from itertools import combinations
from molecular_graphs.lib import pdb_to_graph
from networkx.algorithms import isomorphism as iso
from copy import deepcopy
import pandas as pd
import Subgraphs.Match_Viewer as mv
from math import exp


class NodeMatcher:
    @staticmethod
    def label(n1, n2):
        return n1["label"] == n2["label"]  # label is atomic element

    @staticmethod
    def edge(n1, n2):  # possibly not needed? There is an edge match function in the isomorphism checker
        warnings.warn('NYE', NotImplementedError)
        return n1.degree() == n2.degree()  # edge degree/number of connected edges

    @staticmethod
    def label_and_edge(n1, n2):
        return n1["label"] == n2["label"] and n1.degree() == n1.degree()  # combination of above two


def get_isomorphisms(target_graph, ref_graph, node_match=NodeMatcher.label, edge_match=None, mute=True):
    """
    Only used to check for a subgraph against a graph
    modified function from molecular_graphs.subgraph_isomorphism
    :param edge_match: <function> function to check the added cases of isomorphism requirements
    :param ref_graph: <networkx.classes.graph.Graph>, "smaller" of the two graphs
    :param target_graph: <networkx.classes.graph.Graph> "Larger" of the two graphs
    :param node_match: <function> function to check the added cases of isomorphism requirements
    :return: <list <dict>> i.e [{1:2},{4:3}] target node id's are keys, reference node id's are values
    """

    graph_match = iso.GraphMatcher(target_graph, ref_graph, node_match=node_match, edge_match=edge_match)
    # creates an instance of the GraphMatcher result (see documentation for GraphMatcher Object in networkx)
    if graph_match.is_isomorphic():
        mapping_generator = graph_match.isomorphisms_iter()  # If isomorphic, return the isomorphic matches
    elif graph_match.subgraph_is_isomorphic():  # if there are subgraphs that are isomorphic, return those mappings
        mapping_generator = graph_match.subgraph_isomorphisms_iter()
    else:
        if mute == False:
            print("Mapping could not be found from %s to %s" % (
                target_graph.name, ref_graph.name))  # say what mapping could not be found
        return []  # return empty list instead of raising exception
        # TODO: update mappings functions to not account for the none result (function used to return None here
    mappings = list(mapping_generator)  # turn the generator object into a list of the mappings

    # basic consistency check to ensure element types all match
    assert all([target_graph.nodes[i]["label"] == ref_graph.nodes[j]["label"] for mapping in mappings for i, j in
                mapping.items()]), \
        "Element type mismatch, indicates something went very wrong: {}".format(
            [
                (target_graph.nodes[i]["label"], ref_graph.nodes[j]["label"]) for mapping in mappings
                for i, j in mapping.items() if target_graph.nodes[i]["label"] != ref_graph.nodes[j]["label"]
            ]
        )
    return mappings


def fragment_old(G, source_node_id, steps):
    """

    :param G: <networkx.classes.graph.Graph> Graph object to pull fragments form
    :param source_node_id:: <Int> Node Value/ID to start the fragment from
    :param steps: maximum number of steps to take from the source node
    :return: <networkx.classes.graph.Graph> Graph object of the fragment
    """

    # warnings.warn('Use fragment instead', DeprecationWarning)

    # returns subraph of G, n edges away from H (a node)
    # i.e. calling H=1, n=1 will find a subgraph using all points directly connected (n=1) to node 1
    warnings.warn('Use new fragment method', DeprecationWarning)

    def subgraph_find(G, source_node, n):
        if n != 0:
            for H_node in list(source_node.nodes):
                for G_node in [x for x in G.nodes if G.has_edge(H_node, x)]:
                    source_node.add_edge(H_node, G_node)  # will also add G_node to H
            for (H_node_a, H_node_b) in combinations(list(source_node.nodes), 2):
                if G.has_edge(H_node_a, H_node_b):
                    source_node.add_edge(H_node_a, H_node_b)  # fixes edges between two newly added nodes
            return subgraph_find(G, source_node, n - 1)
        else:
            return source_node

    H = nx.Graph({source_node_id: {}})  # Initialises the graph with one node labelled as the node_id
    nx.set_node_attributes(H, '', 'label')

    return subgraph_find(G, H, steps)


def fragment(G, source_node_id, steps):
    """
    16000x faster speedup over fragment_old
    Finds a subgraph starting at source_node_id that extends the number of steps out
    :param G: <networkx.classes.graph.Graph>, must have been imported from a pdb and have the label attribute
    :param source_node_id: <int> Starting node
    :param steps: <int>, number of steps
    :return: <networkx.classes.graph.Graph>
    """
    # TODO: Maybe, ensure that the last nodes visited can still be connected even though it is outside of the range of the steps

    assert source_node_id in G.nodes  # checks the node id is valid

    def _fragment(G, H, source_node, n):
        """
        Returns NoneType as the result, H is mutable and changes are saved as they occur
        :return: NoneType
        """
        if n != 0:
            for i in G.neighbors(source_node):
                if not H.has_edge(source_node, i):  # stops the tree algorithm going back on itself by checking if
                    # it has already made a path
                    H.add_edge(source_node, i)  # this will also add the node to the graph H
                    H.nodes[i]['label'] = G.nodes[i]['label']  # copies over the element type
                    _fragment(G, H, i, n - 1)

    H = nx.Graph({source_node_id: {}})  # Creates Graph of one node with given node id
    nx.set_node_attributes(H, '', 'label')
    H.nodes[source_node_id]['label'] = G.nodes[source_node_id]['label']  # sets the correct element type
    _fragment(G, H, source_node_id, steps)  # runs the recursive function
    H.name = G.name + '_frag_id' + str(source_node_id) + '_lv' + str(
        steps)  # sets the name to indicate how it was created
    return H


def graph_generation(pdb_path):
    """
    Function to generate networkx Graph objects from pdb files
    :param pdb_path: <str> path to pdb file
    :return: <networkx.classes.graph.Graph> Graph object of inputted molecule
    """
    with open(pdb_path) as fh:
        pdb_str = fh.read()
    mol_graph = pdb_to_graph(pdb_str)
    name = pdb_path.split('/')[-1].split('\\')[-1]
    mol_graph.name = '.'.join(name.split('.')[0:-1])
    return mol_graph


def fix_graph_attr(G, label_src=None):
    """
    not neeeded,  use Graph.degree instead

    From function warafin.parse.fix_graph_attr \n
    :param G: <networkx.classes.graph.Graph> Input graph to add number of neighbors as an attribute
    :return: <NoneType> Applies changes to inputed graph object.

    """
    warnings.warn('Adding the edge attribute is unnecessary, please use code built on node degrees', DeprecationWarning)
    H = G.copy()
    for H_node in H.node:
        H.node[H_node]["edge_count"] = len(H.edges(H_node))  # number of neighbours
    if label_src is not None:
        if type(label_src) is nx.Graph:
            # copy labels from other graph
            for H_node in H.nodes:
                H.node[H_node]["label"] = label_src.node[H_node]["label"]
    return H


def display_graph(graph, save_path=None):
    """
    Displays the graph and optionally saves the file
    saves a few line of code
    :param save_path: <str> or <NoneType>, If None does not save the file
    :param graph: <networkx.classes.graph.Graph>
    :return: NoneType: produces Graph and optionally saves the file
    """
    nx.draw_networkx(graph, with_labels=True)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def fragment_similarity_Koch(mol_graph_list, level, save_path=None):
    """
    Attempts to implement Koch's algorithm as was used for OFramP, not really developed
    :param mol_graph_list: <list<networkx.classes.graph.Graph>> list of graphs imported from PDB's
    :param level: <int> size of substructures to compare
    :param save_path: <str> path with filename, or just filename with no file extension
    :return: <dict> of mappings and writes file with mappings
    """

    def visit(K):
        pass

    if save_path is None:
        save_path = "Molecular_Comparison_" + str(level) + ".messy.dat"
    else:
        save_path += ".messy.dat"
    if os.path.isfile(save_path):
        save_path = save_path.split(".messy.dat") + "_"
    with open(save_path, 'wr') as save_file:
        old_fragments = []
        for mol_graph_1, mol_graph_2 in combinations(mol_graph_list, 2):
            tree = nx.minimum_spanning_tree(mol_graph_2)
            current_frag = fragment(mol_graph_1, level)
            for node in mol_graph_1.nodes:
                pass


def fragment_similarity_iter(mol_graph_list, level, save_path=None, write=False):
    """
    Iterative method to determine similar structures by using largest fragments
    For every molecule, generates all fragments at the desired level and compares them against a list of:
     - The old fragments ( to see if it should skip)
     - The other molecules, to see if there is an isomorphism
    :param mol_graph_list: <list<networkx.classes.graph.Graph>> list of graphs imported from PDB's
    :param level: <int> size of substructures to compare
    :param save_path: <str> path with filename, or just filename with no file extension
    :return: <dict> of mappings and writes file with mappings
    """
    # TODO: an issue with the old fragments is they need to be categorised by source molecule, maybe? Or maybe not

    old_fragments = {}  # Initialises empty lists for the fragments as they are created
    maps = []  # initialises the empty dictionary to store the mapping dictionaries

    for mol_graph_ref in mol_graph_list:
        for node_id in mol_graph_ref.nodes:
            current_frag = fragment(mol_graph_ref, node_id, level)
            if not any([nx.is_isomorphic(x, current_frag) for x in list(old_fragments.values())]):
                for mol_graph_target in mol_graph_list:
                    if mol_graph_ref != mol_graph_target:
                        maps.append({'reference': mol_graph_ref.name,
                                     'target': mol_graph_target.name,
                                     'fragment': current_frag.name,
                                     'mapping': get_isomorphisms(mol_graph_target, current_frag)}
                                    )  # currently does not remove symmetrical maps, adds the maps to the mapping list
                old_fragments.update({current_frag.name: current_frag})  # adds the fragment if it is not isomorphic with any previous

    if write:
        # This section deals with determining the file name, adds (<int>) similar to saving files on windows
        if save_path is None:  # Creates a file name from the Graph names
            save_path = ''.join(x.name[0:5] for x in mol_graph_list) + "_lv_" + str(level)
        else:
            save_path += ".txt"
        i = 1  # Adding numbers to end of files, checks in while loop
        while os.path.exists(save_path + " (%s).txt" % i):
            i += 1
        save_path = save_path + " (%s).txt" % i
        print(save_path)
        with open(save_path, 'w') as save_file:  # creates the file as save_file
            save_file.writelines("%s\n" % map for map in maps)  # Writes the mapping list to a file
            save_file.close()  # Closes the file

    return old_fragments, maps


def remove_symmetries(mappings, save_path=None, write=False):
    """
    Passes by value. \n
    Checks for 'symmetric' mapping by checking if a mapping has the same set of target and reference maps, irrespective
    of how they are paired, i.e these dictionaries are symmetric:
    {1:2, 2:3, 3:1}, {1:3, 2:1, 3:2}.
    \n \n \n
    This makes assumptions about the way mappings work, it does not check if number occur twice for e.g. as they shouldn't.
    If it takes a mapping where this is the case it will not account for it.
    \n \n
    Assumptions:
     - nodes can't have two mappings in one mapping dict
     - graphs can't be structured such that a symmetric mapping can actually exit (this is true for chemicals, see lab book)
     - # TODO demonstrate all previous points in lab book
    :param mappings: mapping output generated by fragment_similarity_iter, currently can't read a saved list generated
    by same function # TODO add functionality to load from file
    :return <dict> of mappings and writes file with mappings:
    """
    mappings_copy = deepcopy(mappings)  # need to use deepcopy for the substructures, otherwise the passed mapping object also changes
    maps_clean = [] # empty list which the new de-symmetried top-level dictionaries are put into
    for i in mappings_copy: # list is  iterated through, see fragment_similarity_iter output for what the structure looks like
        current_maps = []
        maps_clean.append(i)  # adds the line being analysed, might be faster not to add maps now (part of i)
        # but requires more effort
        if i['mapping'] is not None:  # separate if statements needed as len() condition was being evaluated on None and crashing
            if not len(i['mapping']) == 1: # checks if there is not only 1 mapping
                # TODO Add visualisation in lab book for next step
                # TODO manually check a file at some point

                # Probably a better way to structure the following code
                for j, map_1 in enumerate(i['mapping']):
                    # j is the index, map_1 is the current map dictionary. iterating over the original maps list
                    inst_keyset = set(map_1.keys())
                    # stores the unique set (in this case just orders it) of the key which correspond to
                    # the nodes being mapped too (from the fragment)

                    to_check_cond = not inst_keyset in [set(x.keys()) for x in
                                                        [q for p, q in enumerate(i['mapping']) if p != j]]
                    # stores the condition, seeing if the key set is in a list of the key sets from the old mappings
                    # this list [q for p, q in enumerate(i['mapping']) if p != j] is the same list of old mappings
                    # excluding map_1, done by saying when p(index of new list) is not the same as j, index of larger for loop

                    past_check_cond = not inst_keyset in [set(x.keys()) for x in current_maps]
                    # stores the condition, seeing if the keyset is in a list of the keysets already added to the new maps
                    if to_check_cond or past_check_cond: # combines the previous conditions
                        current_maps.append(map_1) # adds the current map if it passes the tests
                maps_clean[-1]['mapping'] = current_maps # as the template is always added at the start of this code block
                # (maps_clean.append(i)), takes the most recently added and changes the mapping in the top level dictionary
                # ({target:, reference:, mapping:,}

    if write:
        # This section deals with determining the file name, adds (<int>) similar to saving files on windows
        if save_path is None:
            save_path = 'Cleaned mapping file'
        else:
            save_path += ".txt"
        i = 1  # Adding numbers to end of files, checks in while loop
        while os.path.exists(save_path + " (%s).txt" % i):
            i += 1
        save_path = save_path + " (%s).txt" % i
        print(save_path)
        with open(save_path, 'w') as save_file:  # creates the file as save_file
            save_file.writelines("%s\n" % line for line in maps_clean)  # Writes the mapping list to a file
            save_file.close()  # Closes the file

    return maps_clean


def recursive_unmatch(mol_graph_1, mol_graph_2, mappings, node, steps_max, step):
    if step < steps_max:
        for neighbour in mol_graph_2.neighbors(node):
            mol_graph_2.nodes[neighbour]['buffer_match'] = False

            # Following code searches mapping file for matches back to other molecule

            for mapping_set in mappings:
                for mapping in mapping_set['mapping']:
                    if neighbour in mapping.keys():
                        mol_graph_1.nodes[mapping[neighbour]]['buffer_match'] = False
                        print('unmatched node', mapping[neighbour], mol_graph_1.nodes[mapping[neighbour]]['label'])

            recursive_unmatch(mol_graph_1, mol_graph_2, mappings, neighbour, steps_max, step+1)


def match_buffer_multi(mol_graph_list):

    old_fragments = {}  # Initialises empty lists for the fragments as they are created

    for mol_graph_ref in mol_graph_list:
        for depth in list(range(1, len(mol_graph_ref)+1))[::-1]:
            for node_id in mol_graph_ref.nodes:
                current_frag = fragment(mol_graph_ref, node_id, depth)
                if not any([nx.is_isomorphic(x, current_frag) for x in list(old_fragments.values())]):
                    old_fragments.update({current_frag.name: current_frag})

def match_buffer_pair_v3(mol_graph_1, mol_graph_2, buffer=3, min_depth=3, min_size = 0):
    """

    :param mol_graph_1: <netowrkx Graph>
    :param mol_graph_2: <netowrkx Graph>
    :param mappings: <list> from remove_symmetries function
    :param buffer: <int> for buffer size
    :param min_depth: <int> minimum depth search for fragments
    :return:
    """
    if len(mol_graph_2) < len(mol_graph_1): # ensures mol_graph_1 is the larger molecule
        mol_graph_1, mol_graph_2 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    else:
        mol_graph_2, mol_graph_1 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    # Deepcopies needed as these graphs are modified Otherwise code modifies the passed graphs

    nx.set_node_attributes(mol_graph_1, False,'buffer_match')  # creates a binary attribute on the nodes to confirm matches
    nx.set_node_attributes(mol_graph_2, False, 'buffer_match')
    frags = [] # TODO: Change this to an updatable dictionary
    mappings = []
    for depth in list(range(min_depth, len(mol_graph_1) + 1))[::-1]:  # Short albeit moderately unreadable code, creates
    # a reversed list to iterate through, starting at the maximum size of the molecule

        inst_frags, inst_mappings = fragment_similarity_iter([mol_graph_1, mol_graph_2], depth)
        inst_frags = {x:y for x, y in inst_frags.items() if len(y) >= min_size }
        frags.extend(inst_frags)
        mappings.extend(inst_mappings)
    mappings = remove_symmetries(mappings)

    for mapping_set in mappings:
        for mapping in mapping_set['mapping']:
            if all([mol_graph_1.nodes[mapping[key]][
                    'buffer_match'] == True for key in mapping]) and all([mol_graph_2.nodes[key][
                    'buffer_match'] == True for key in mapping]):
                del mapping
            else:
                for key in mapping:
                    mol_graph_1.nodes[mapping[key]][
                        'buffer_match'] = True  # mol_graph_1 is the reference graph
                    # (source of the fragment) ans thus uses dictionary values as the node_id
                    mol_graph_2.nodes[key][
                        'buffer_match'] = True  # mol_graph_2 is the target node and uses keys for node_id
                    # see documentation for get_isomorphisms as 'mapping' values are obtained from there)

    # Buffering code Uses recursive code from non-matched nodes
    # uses a iterative method over the nodes to find them
    # Doesn't check if the nodes it comes across are unmatched or not as the only use is to unmatch them
    # does for one molecule to start
    # creates a list and then starts unmatching nodes to stop the code recursively unmatching



    edge_nodes = []
    for node in mol_graph_2.nodes:
        if any(mol_graph_2.nodes[neighbour]['buffer_match']==True for neighbour in mol_graph_2.neighbors(node))\
                and\
                mol_graph_2.nodes[node]['buffer_match']==False:
            edge_nodes.append(node)
    for node in edge_nodes:
        recursive_unmatch(mol_graph_1, mol_graph_2, node, mappings, buffer, 0)


    return mol_graph_1, mol_graph_2

def match_buffer_pair_levels(mol_graph_1, mol_graph_2, buffer=3, min_depth=3, min_size = 10):
    """
    # TODO: It seems like this is going to need very large amounts of data filtering and storage
    # TODO: this method is less accurrate then v3
    :param mol_graph_1:
    :param mol_graph_2:
    :param buffer:
    :param min_depth:
    :param min_size:
    :return:
    """

    if len(mol_graph_2) < len(mol_graph_1): # ensures mol_graph_1 is the larger molecule
        mol_graph_1, mol_graph_2 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    else:
        mol_graph_2, mol_graph_1 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    # Deepcopies needed as these graphs are modified Otherwise code modifies the passed graphs

    nx.set_node_attributes(mol_graph_1, False, 'buffer_match')  # creates a binary attribute on the nodes to confirm matches
    nx.set_node_attributes(mol_graph_2, False, 'buffer_match')

    frags = {}
    mappings = []

    for depth in list(range(min_depth, len(mol_graph_1) + 1))[::-1]:  # Short albeit moderately unreadable code, creates
    # a reversed list to iterate through, starting at the maximum size of the molecule

        inst_frags, inst_mappings = fragment_similarity_iter([mol_graph_1, mol_graph_2], depth)
        inst_frags = {x:y for x, y in inst_frags.items() if len(y) >= min_size }
        frags.update(inst_frags)
        mappings.extend(inst_mappings)
    mappings = remove_symmetries(mappings)

    mappings_raw = [] # this currently only works for pairs as they need the same target and source
    for mapping_set in mappings:
        mappings_raw.extend(mapping_set['mapping'])

    max_length = max([len(x) for x in mappings_raw])

    weighted_matches = pd.DataFrame(columns=['mol_1_node', 'mol_2_node', 'freq']) # this method doesn't prioritise size
    for mapping_set in mappings:
        for mapping in mapping_set['mapping']:
            for mol_2_node, mol_1_node in mapping.items():
                bool_test = (weighted_matches['mol_1_node'] == mol_1_node) & (weighted_matches['mol_2_node'] == mol_2_node)
                if sum(bool_test)==1:
                    weighted_matches.at[bool_test, 'freq'] = weighted_matches.loc[bool_test, 'freq'] +  exp(-5*len(mapping)/max_length)
                elif  sum(bool_test)==0:
                    weighted_matches = weighted_matches.append({'mol_1_node':mol_1_node, 'mol_2_node':mol_2_node, 'freq':exp(-5*len(mapping)/max_length)}, ignore_index=True)
                    # Pandas does not pass by reference
                else:
                    print('something is seriously wrong, 2+ occurrences of node pairs are in dataframe')
    weighted_matches['freq'] = pd.to_numeric(weighted_matches['freq'])
    # return weighted_matches

    mapping_final = {}
    for node in mol_graph_1.nodes:
        node_pairs = weighted_matches.loc[weighted_matches['mol_1_node'] == node]
        if len(node_pairs)>0:
            max_id = node_pairs['freq'].idxmax()
            final_col = weighted_matches.loc[max_id]
            mapping_final.update({final_col['mol_2_node']:final_col['mol_1_node']})



    for key in mapping_final:
        mol_graph_1.nodes[mapping_final[key]][
            'buffer_match'] = True  # mol_graph_1 is the reference graph
        # (source of the fragment) ans thus uses dictionary values as the node_id
        mol_graph_2.nodes[key][
            'buffer_match'] = True  # mol_graph_2 is the target node and uses keys for node_id
        # see documentation for get_isomorphisms as 'mapping' values are obtained from there)

    mv.buffer_match_mappings(mol_graph_1, 'test')
    mv.buffer_match_mappings(mol_graph_2, 'test')

    edge_nodes = []
    for node in mol_graph_2.nodes:
        if any(mol_graph_2.nodes[neighbour]['buffer_match'] == True for neighbour in mol_graph_2.neighbors(node)) \
                and \
                mol_graph_2.nodes[node]['buffer_match'] == False:
            edge_nodes.append(node)
    for node in edge_nodes:
        recursive_unmatch(mol_graph_1, mol_graph_2, mappings, node, buffer, 0)

    return mol_graph_1, mol_graph_2, weighted_matches

def match_buffer_pair_snap(mol_graph_1, mol_graph_2, buffer=3, min_depth=3, min_size = 10):
    """
    # TODO: add lengths somewhere earlier in process to fragments? This is easily obtainable data though
    :param mol_graph_1:
    :param mol_graph_2:
    :param buffer:
    :param min_depth:
    :param min_size:
    :return:
    """
    if len(mol_graph_2) < len(mol_graph_1): # ensures mol_graph_1 is the larger molecule
        mol_graph_1, mol_graph_2 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    else:
        mol_graph_2, mol_graph_1 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    # Deepcopies needed as these graphs are modified Otherwise code modifies the passed graphs

    nx.set_node_attributes(mol_graph_1, False, 'buffer_match')  # creates a binary attribute on the nodes to confirm matches
    nx.set_node_attributes(mol_graph_2, False, 'buffer_match')

    frags = {}
    mappings = []

    for depth in list(range(min_depth, len(mol_graph_1) + 1))[::-1]:  # Short albeit moderately unreadable code, creates
        # a reversed list to iterate through, starting at the maximum size of the molecule

        inst_frags, inst_mappings = fragment_similarity_iter([mol_graph_1, mol_graph_2], depth)
        inst_frags = {x: y for x, y in inst_frags.items() if len(y) >= min_size}
        frags.update(inst_frags)
        mappings.extend(inst_mappings)
    mappings = remove_symmetries(mappings)

    frags_lengths = {}
    for fragment_name in frags.keys():
        frags_lengths.update({fragment_name:len(frags[fragment_name])})
    
    frags = sorted(frags, key = lambda fragment_name: frags_lengths[fragment_name], reverse=True)
    # this is now a list of tuples
    # frags are now ordered:

    mapping_final = {}

    for fragment_name in frags:
        inst_mapping_set = next(mapping_set for mapping_set in mappings if mapping_set['fragment']==fragment_name)
        #Generator expression used here to search the dictionary, from my development implies I need to rework data structures
        inst_frag_mappings = inst_mapping_set['mapping']
        if len(inst_frag_mappings) == 1:
            print("Only one mapping found between fragment %s and molecule %s, considering this successful" % (fragment_name, mol_graph_1.name))



            mapping_final.update({inst_frag_mappings[0]})
        elif len(inst_frag_mappings)>1: # TODO: ensure keys and values are references the graphs in the right order i.e. fragments and keys should be the same molecule
            pairs_del = []
            for frag_mapping_ref in inst_frag_mappings:
                # Check for discrepancies or if it has rotated about a point (often happens with hydrogen, strip these off the mapping)
                # This doesn't not neccesarily mean charges will not match properly but the
                # buffering system in its current state does not deal with this well

                # This method still used the first one as a basis also very inefficient as it loops twice
                for node_pair in frag_mapping_ref.items():
                    for frag_mapping_compare in inst_frag_mappings:
                        if not node_pair in frag_mapping_compare.items(): # If this node pair is not consistent across every mapping
                            pairs_del.append(node_pair) #mark it for deletion
            for node_pair in pairs_del:
                for frag_mapping in inst_frag_mappings:
                    del frag_mapping[node_pair[0]]
            print('Deleted %d nodes from %d mappings for %s'
                  % (len(pairs_del), len(inst_frag_mappings), fragment_name)
                  )
            mapping_final.update(inst_frag_mappings[0]) # They should all be the same now
        else:
            print('%s had no mappings' % fragment_name)


    for key in mapping_final:
        mol_graph_1.nodes[mapping_final[key]][
            'buffer_match'] = True  # mol_graph_1 is the reference graph
        # (source of the fragment) ans thus uses dictionary values as the node_id
        mol_graph_2.nodes[key][
            'buffer_match'] = True  # mol_graph_2 is the target node and uses keys for node_id
        # see documentation for get_isomorphisms as 'mapping' values are obtained from there)

    mv.buffer_match_mappings(mol_graph_1, 'test')
    mv.buffer_match_mappings(mol_graph_2, 'test')

    edge_nodes = []
    for node in mol_graph_2.nodes:
        if any(mol_graph_2.nodes[neighbour]['buffer_match'] == True for neighbour in mol_graph_2.neighbors(node)) \
                and \
                mol_graph_2.nodes[node]['buffer_match'] == False:
            edge_nodes.append(node)
    for node in edge_nodes:
        recursive_unmatch(mol_graph_1, mol_graph_2, mappings, node, buffer, 0)

    return mol_graph_1, mol_graph_2


def match_buffer_pair_buffer1(mol_graph_1, mol_graph_2, buffer=3, min_depth=3, min_size = 10):
    if len(mol_graph_2) < len(mol_graph_1): # ensures mol_graph_1 is the larger molecule
        mol_graph_1, mol_graph_2 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    else:
        mol_graph_2, mol_graph_1 = deepcopy(mol_graph_1), deepcopy(mol_graph_2)
    # Deepcopies needed as these graphs are modified Otherwise code modifies the passed graphs

    nx.set_node_attributes(mol_graph_1, False, 'buffer_match')  # creates a binary attribute on the nodes to confirm matches
    nx.set_node_attributes(mol_graph_2, False, 'buffer_match')
    frags = [] # will probably be another list of dictionaries
    # Might have to write a second lib file
    # currently code cumbersomely rewrites the data formats on the fly
    mappings = []

    for depth in list(range(min_depth, len(mol_graph_1) + 1))[::-1]:  # Short albeit moderately unreadable code, creates
        # a reversed list to iterate through, starting at the maximum size of the molecule

        inst_frags, inst_mappings = fragment_similarity_iter([mol_graph_1, mol_graph_2], depth)
        inst_frags = {x: y for x, y in inst_frags.items() if len(y) >= min_size}
        frags.update(inst_frags)
        mappings.extend(inst_mappings)
    mappings = remove_symmetries(mappings)

def vector_construction(mappings, save_path=None, write=False):
    """
    Generates a wighted mapping object that can be visualised with yet to be written code in
    Match_Viewer.py
    :param mappings: Typically output form remove_symmetry, but can be any similarly structures mapping object
    Does require a fragment option or something similar, for now 'fragment' is hardcoded
    :return: TODO yet to work out how to sort this data
    # TODO: ensure the outputted vector lengths are all the same and match up correctly
    """
    # If incoming data was structured better this could likely be rewritten more elegantly
    fragment_names = list(set([map_dict['fragment'] for map_dict in mappings]))
    molecule_names = list(set([map_dict['reference'] for map_dict in mappings] + [map_dict['target'] for map_dict in mappings]))
    cycle_maps = pd.DataFrame(columns=molecule_names)
    match_counts = {name:[] for name in molecule_names}
    for name in fragment_names:
        current_map_dicts = [x for x in mappings if x['fragment']==name]
        for entry in current_map_dicts:

            if entry['mapping'] is None:
                match_counts[entry['target']].append(0)
            else:
                match_counts[entry['target']].append(len(entry['mapping']))
    return fragment_names, match_counts







if __name__ == '__main__':
    G1 = graph_generation(r'Warfarin_PDB_Files/2225_warfarin.pdb')
    G2 = graph_generation(r'Warfarin_PDB_Files/2222_warfarin.pdb')
    G3 = graph_generation(r'Warfarin_PDB_Files/2202_warfarin.pdb')
    G4 = graph_generation(r'Warfarin_PDB_Files/2227_warfarin.pdb')
    G5 = graph_generation(r'Warfarin_PDB_Files/4386_warfarin.pdb')
    G6 = graph_generation(r'Warfarin_PDB_Files/4388_warfarin.pdb')
    G7 = graph_generation(r'Warfarin_PDB_Files/4450_warfarin.pdb')
    G8 = graph_generation(r'Warfarin_PDB_Files/4451_warfarin.pdb')
    G9 = graph_generation(r'Warfarin_PDB_Files/5523_warfarin.pdb')
    G10 = graph_generation(r'Warfarin_PDB_Files/5568_warfarin.pdb')
    G11 = graph_generation(r'Warfarin_PDB_Files/5582_warfarin.pdb')
    G12 = graph_generation(r'Warfarin_PDB_Files/5583_warfarin.pdb')

    test1 = graph_generation(r'Other Files\371399.pdb')


    buffer_test1 = graph_generation(r'Other Files\toluene.pdb')
    buffer_test2 = graph_generation(r'Other Files\2-Chlorotoluene.pdb')


    F1, F2, matches = match_buffer_pair_snap(G1,G2, min_size=15, buffer=1)

    mv.buffer_match_mappings(F2,'test')
    mv.buffer_match_mappings(F1, 'test')
    #
    # test2 = graph_generation(r'Other Files\372906.pdb')

    # frags, maps = fragment_similarity_iter([G1, G3, G4], 3)
    # clean_maps = remove_symmetries(maps, write=False)
    #
    # frags_tested, count_vector = vector_construction(clean_maps)

    # maps_test = [x for x in maps if x['mapping'] is not None]

    # maps_length = [len(x['mapping']) for x in maps_test]

    # maps_sym_length = [len(x['mapping']) for x in clean_maps]

    # frag1_maps = [x for x in clean_maps if x['fragment'] == '2225_warfarin_frag_id1_lv3']

    # mv.animate_mappings_multi(frag1_maps, [G3, G3, G4, G5, G6, G7], G1, 'nn')

    # mv.animate_mappings(maps[0]['mapping'], G2, 'newtest')
