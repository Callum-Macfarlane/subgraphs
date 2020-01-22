__author__ = 'Callum Macfarlane'
__email__ = 'c.macfarlane@uqconnect.edu.au'

''' This version in done primarily using Pandas which by comparison to just dictionaries and arrays has terrible performance.
Readability and development of this program is substantially easier though
'''

# TODO: Break symmetry based on bond Types
# - Might be solved by buffering out fragment matches
# TODO: Decide between list of dicts or pandas DataFrame
# TODO: fragment iter similarity will always match a fragment back on itself when checking a fragment against its source
#  molecule for similarities


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
from math import exp


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
    nx.draw_networkx_edges(mol_graph_2, mol_2_layout, ax=axes[1])
    nx.draw_networkx_nodes(mol_graph_2, mol_2_layout, ax=axes[1],
                           node_color=list({**mol_2_unmapped_nodes_color, **mol_2_mapped_nodes_color}.values()),
                           nodelist=list({**mol_2_unmapped_nodes_color, **mol_2_mapped_nodes_color}.keys()))

    plt.show()


class NodeMatcher:
    """
    Class for storing node matching check functions used in get_isomorphisms
    """

    @staticmethod
    def label(n1, n2):
        return n1["label"] == n2["label"]


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


def get_isomorphisms(target_graph, ref_graph, node_match=NodeMatcher.label, edge_match=None, mute=True):
    """
    Only used to check for a subgraph (reference graph) against a graph
    modified function from molecular_graphs.subgraph_isomorphism
    :param edge_match: <function> function to check the added cases of isomorphism requirements
    :param ref_graph: <networkx.classes.graph.Graph>, "smaller" of the two graphs (subgraph)
    :param target_graph: <networkx.classes.graph.Graph> "Larger" of the two graphs (target graph)
    :param node_match: <function> function to check the added cases of isomorphism requirements
    :param mute: <Boolean> silence output of this function
    :return: <generator> target node id's are keys, reference node id's are values
    """

    graph_match = iso.GraphMatcher(target_graph, ref_graph, node_match=node_match, edge_match=edge_match)
    # creates an instance of the GraphMatcher result (see documentation for GraphMatcher Object in networkx)
    if graph_match.is_isomorphic():
        mapping_generator = graph_match.isomorphisms_iter()  # If isomorphic, return the isomorphic matches
        if not mute:
            print("A mapping from reference %s to target %s was Isomorphic"
                  % (ref_graph.name, target_graph.name))  # Print if isomorphic

    elif graph_match.subgraph_is_isomorphic():  # if there are subgraphs that are isomorphic, return those mappings
        mapping_generator = graph_match.subgraph_isomorphisms_iter()
        if not mute:
            print("Mappings from reference %s to target %s were subgraph Isomorphic"
                  % (ref_graph.name, target_graph.name))  # Print if subgraph isomorphic

    else:
        if not mute:
            print("Mappings from reference %s to target %s could not be found" % (
                ref_graph.name, target_graph.name))  # print what mapping could not be found
        return []  # return empty list instead of raising exception
        # TODO: update mappings functions to not account for the none result (function used to return None here
        # - Being done in this file
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

    # invert mappings so smaller node id's are used as keys
    mappings = [{item: key for key, item in mapping_dict.items()} for mapping_dict in mappings]
    return mappings  # As a generator is faster but single use effectively list seems more appropriate for now


def fragment(G, source_node_id, steps):
    """
    16000x faster speedup over fragment_old
    Finds a subgraph starting at source_node_id that extends the number of steps out
    :param G: <networkx.classes.graph.Graph>, must have been imported from a pdb and have the label attribute
    :param source_node_id: <int> Starting node
    :param steps: <int>, number of steps to take
    :return: <networkx.classes.graph.Graph> Returns the fragment as a graph
    """
    # Done: Maybe, ensure that the last nodes visited can still be connected even though it is outside of the range of the steps
    # - Fixed when buffering out fragments. This is not done here, see matching functions

    assert source_node_id in G.nodes  # checks the node id is valid

    def recursive_fragment(G, H, source_node, n):
        """
        :param n: current step, counted down from param steps of parent function
        Returns NoneType as the result, H is mutable and changes are saved as they occur
        :return: NoneType
        """
        if n != 0:
            for i in G.neighbors(source_node):
                if not H.has_edge(source_node, i):  # stops the tree algorithm going back on itself by checking if
                    # it has already made a path
                    H.add_edge(source_node, i)  # this will also add the node to the graph H
                    H.nodes[i]['label'] = G.nodes[i]['label']  # copies over the element type
                    recursive_fragment(G, H, i, n - 1)  # rerun with one less step size

    H = nx.Graph({source_node_id: {}})  # Creates Graph of one node with given node id
    nx.set_node_attributes(H, '', 'label')
    H.nodes[source_node_id]['label'] = G.nodes[source_node_id][
        'label']  # sets the correct element type of the initial node
    recursive_fragment(G, H, source_node_id, steps)  # runs the recursive function
    H.name = G.name + '_node_id' + str(source_node_id) + '_lv' + str(
        steps)  # sets the name to indicate how it was created, '_frag_id' changed to '_node_id'
    return H  # returns the fragment graph


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


def fragment_similarity_iter(mol_graph_list, level, save_path=None, write=False, mute=True):
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
    # Done: an issue with the old fragments is they need to be categorised by source molecule, maybe? Or maybe not
    # Began using pandas DataFrames, may sacrifice speed for development ease

    # TODO: Intramolecular mappings
    # TODO: add check for identical fragments (node id and all)) all line should be good

    fragments = pd.DataFrame(columns=['name', 'reference_name', 'target_name', 'mapping', 'graph', 'source_node',
                                      'level'])  # Initialises empty Dataframe for the fragments as they are created
    for mol_graph_ref in mol_graph_list:  # TODO: Consider moving this loop outside this function
        for node_id in mol_graph_ref.nodes:
            current_frag = fragment(mol_graph_ref, node_id, level)
            # if not any([nx.is_isomorphic(x, current_frag) for x in list(fragments['graph'])]):
            # TODO: See if this previous line needs to be implemented
            # TODO: These lines deal with intermolecular mappings maybe, so far just added loop to compare against itself
            # TODO: Consider only comparing fragments to other fragments? But Currently not confident the fragment code
            #  is robust enough to produce all necessary ones
            for mol_graph_target in mol_graph_list:
                fragments = fragments.append({'reference_name': mol_graph_ref.name,
                                              'target_name': mol_graph_target.name,
                                              'name': current_frag.name,
                                              'mapping': get_isomorphisms(mol_graph_target, current_frag, mute=mute),
                                              'graph': current_frag,
                                              'source_node': node_id,
                                              'level': level},
                                             ignore_index=True
                                             )  # currently does not remove symmetrical maps, adds the maps to the mapping list
                # Not sure if all this data is needed at this point, something to look at when optimising
    fragments[['source_node', 'level']] = fragments[['source_node', 'level']].astype(int)
    if write:
        # This section deals with determining the file name, adds (<int>) similar to saving files on windows
        if save_path is None:  # Creates a file name from the Graph names
            save_path = ''.join(x.name[0:5] for x in mol_graph_list) + "_lv_" + str(level)
        else:
            save_path += ".xlsx"
        i = 1  # Adding numbers to end of files, checks in a while loop
        while os.path.exists(save_path + " (%s).xlsx" % i):
            i += 1
        save_path = save_path + " (%s).xlsx" % i
        print(save_path)
        fragments.to_excel(save_path)
        print("Note Graph objects are not saved")

    return fragments


def remove_copies(fragment_DataFrame, save_path=None, write=False):
    """
    Passes by value. \n
    Checks for 'symmetric' mapping by checking if a mapping has the same set of target and reference maps, irrespective
    of how they are paired, i.e these dictionaries are copies:
    {1:2, 2:3, 3:1}, {1:3, 2:1, 3:2}.
    \n \n \n
    This makes assumptions about the way mappings work, it does not check if number occurs twice for e.g. as they shouldn't.
    If it takes a mapping where this is the case it will not account for it.
    \n \n
    Assumptions:
     - nodes can't have two mappings in one mapping dict
     - graphs can't be structured such that a copied mapping can actually exist (this is true for chemicals, see lab book)
     - # TODO demonstrate all previous points in lab book
    :param mappings: mapping output generated by fragment_similarity_iter, currently can't read a saved list generated
    by same function # TODO add functionality to load from file
    :return <dict> of mappings and writes file with mappings:
    """

    def remove_copies_lambda(mappings_list):
        if len(mappings_list) > 1:  # checks there are 2 or more mappings
            maps_cleaned = []
            # TODO Add visualisation in lab book for next step
            # TODO manually check a file at some point

            # Probably a better way to structure the following code
            for j, map_1 in enumerate(mappings_list):
                # j is the index (which mapping in the set of them(, map_1 is the current map dictionary. iterating over the original maps list
                inst_keyset = set(map_1.keys())
                # stores the unique set (in this case just orders it, as they should never double up)
                # of the key which correspond to the nodes being mapped too (from the fragment)

                to_check_cond = not inst_keyset in [set(x.keys()) for x in
                                                    [q for p, q in enumerate(mappings_list) if p != j]]
                # stores the condition, seeing if the key set is in a list of the key sets from the old mappings
                # this list [q for p, q in enumerate(i['mapping']) if p != j] is the same list of old mappings
                # excluding map_1, done by saying when p(index of new list) is not the same as j, index of larger for loop

                past_check_cond = not inst_keyset in [set(x.keys()) for x in maps_cleaned]
                # stores the condition, seeing if the keyset is in a list of the keysets already added to the new maps
                if to_check_cond or past_check_cond:  # combines the previous conditions
                    maps_cleaned.append(map_1)  # adds the current map if it passes the tests
            return maps_cleaned
        else:
            return mappings_list
            # (maps_clean.append(i)), takes the most recently added and changes the mapping in the top level dictionary
            # ({target:, reference:, mapping:,}

    # Since different entries have the same node these need to be sorted by their source molecule and target molecule
    # It does this per entry as different fragments though identical may have different buffer regions
    # example code:
    # test_graph = nx.path_graph(5)
    # fragment1 = fragment(test_graph, 3, 2)
    # fragment2 = fragment(test_graph, 3, 6)
    # Though both fragments are the same graph fragment one would buffer the edges of the line but fragment 2 would not
    # TODO: double check this example

    fragment_DataFrame['mapping'] = fragment_DataFrame['mapping'].apply(remove_copies_lambda)

    if write:
        # This section deals with determining the file name, adds (<int>) similar to saving files on windows
        if save_path is None:  # Creates a file name from the Graph names
            save_path = 'Cleaned mapping file'
        else:
            save_path += ".xlsx"
        i = 1  # Adding numbers to end of files, checks in a while loop
        while os.path.exists(save_path + " (%s).xlsx" % i):
            i += 1
        save_path = save_path + " (%s).xlsx" % i
        print(save_path)
        fragment_DataFrame.to_excel(save_path)
        print("Note Graph objects are not saved")

    return fragment_DataFrame


def match_buffer(mol_graph_1, mol_graph_2, fragment_buffer=1, min_depth=3, min_size=10, mute=True, max_depth=9999):
    # def recursive_unmatch(entry, step):
    #     match_level = entry['level'] - fragment_buffer

    fragment_DataFrame = pd.DataFrame(
        columns=['name', 'reference_name', 'target_name', 'mapping', 'graph', 'source_node', 'level'])

    if len(mol_graph_1) < max_depth or len(mol_graph_2) < max_depth:
        max_depth = max([len(mol_graph_1), len(mol_graph_2)])

    for depth in list(range(min_depth, max_depth + 1))[::-1]:  # Short albeit moderately unreadable code, creates
        # a reversed list to iterate through, starting at the maximum size of the molecule
        inst_DataFrame = remove_copies(fragment_similarity_iter([mol_graph_1, mol_graph_2], depth, mute=mute))
        fragment_DataFrame = fragment_DataFrame.append(inst_DataFrame, ignore_index=True)

    fragment_series = fragment_DataFrame['name'].unique()  # Not relevant for 2 molecules but is for more

    # grouped_nodes_template = pd.DataFrame({'name':fragment_series, mol_graph_1.name: [[]]*len(fragment_series), mol_graph_2.name: [[]]*len(fragment_series)})
    #  [[]]*len(fragment_series) created every entry as an empty list by default, grouped_nodes[mol_graph_1.name] = []
    #  does not work as it tries to open the list

    # TODO: Need to change the way this data is stored, the source fragment becomes irrelevant after this process

    # todo: Change so nodes are group not the actual fragments

    # Instead of groups use the fragment names
    match_data = {
        fragment_name: {mol_graph_1.name: [], mol_graph_2.name: [], 'graph': None, 'source_node': None, 'level': None}
        for fragment_name in fragment_series}

    def grouping_lambda(entry, match_data):
        ###### TARGET GRAPH == DICT Values
        ###### REFERENCE GRAPH == DICT KEYS  SWITCHED ON 13/1/20
        # These entries will have the same reference graphs as it is based on the fragment
        mappings = entry['mapping']
        if len(mappings) > 0:

            name = entry['name']
            graph = entry['graph']
            source_node = entry['source_node']
            level = entry['level']
            ref_name = entry['reference_name']

            # Only the target will be changing
            target_name = entry['target_name']

            # TODO: make sure nodes aren't doubling up in lists, doesn't affect outcome though but creates cumbersome data
            #  This can be fixed by deleting identical fragments

            # TODO: check if these need to be ordered? The keys that is (and thus the values) yes they do (13/1/20)

            print("Matching entry %s" % name)

            sorted_keys = sorted(mappings[0].keys())
            if match_data[name][ref_name] != 1:
                match_data[name][ref_name].append(sorted_keys)  # TODO: Dodgy workaround needs to be fixed

            for mapping in mappings:  # TODO: create dataFrame for each node
                sorted_values = [mapping[x] for x in sorted_keys]
                match_data[name][target_name].append(sorted_values)

            match_data[name]['graph'] = graph
            match_data[name]['source_node'] = source_node
            match_data[name]['level'] = level
            match_data[name]['reference_name'] = ref_name
        return entry

    for fragment_name in fragment_series:  # Iterates matching over the different fragments that are generated
        fragment_DataFrame.loc[fragment_DataFrame['name'] == fragment_name].apply(grouping_lambda, args=(match_data,),
                                                                                  axis=1)

    # clean the match data # TODO wrewrite so this isnt needed

    for entry in match_data:
        for molecule in [mol_graph_1.name, mol_graph_2.name]:
            set_of_mappings = set(tuple(x) for x in match_data[entry][molecule])
            match_data[entry][molecule] = [list(mapping) for mapping in set_of_mappings]

    mol_graph_1_buffered = deepcopy(mol_graph_1)
    mol_graph_2_buffered = deepcopy(mol_graph_2)

    for fragment_name in fragment_series:  # becomes redundant at this point both fragment series and match_data
        #     # have unique list of names, easier to visualise though and helps with error catching
        #     # Remove the buffered area
        #     # As I'm writing this I'm realising that the match_data object doesn't have enough info to be used so I have added the graph object
        #
        #      TODO: need way to check which molecule is the source molecule
        graph = match_data[fragment_name]['graph']
        source_node = match_data[fragment_name]['source_node']
        buffered_level = match_data[fragment_name]['level'] - fragment_buffer
        reference_name = match_data[fragment_name]['reference_name']
        buffered_fragment = fragment(graph, source_node, buffered_level)
        # create new mapping with only new nodes
        nodes_for_deletion = [node for node in graph.nodes if node not in buffered_fragment.nodes]
        for node in nodes_for_deletion:
            index = match_data[fragment_name][reference_name][0].index(node)
            for molecule in [mol_graph_1.name, mol_graph_2.name]:
                for mapping in match_data[fragment_name][molecule]:
                    del mapping[index]

    return match_data


def match_buffer_multi(graph_list, fragment_buffer=1, min_depth=3, mute=True, max_depth=10):
    # def recursive_unmatch(entry, step):
    #     match_level = entry['level'] - fragment_buffer

    fragment_DataFrame = pd.DataFrame(
        columns=['name', 'reference_name', 'target_name', 'mapping', 'graph', 'source_node', 'level'])

    for depth in list(range(min_depth, max_depth + 1))[::-1]:  # Short albeit moderately unreadable code, creates
        # a reversed list to iterate through, starting at the maximum size of the molecule
        inst_DataFrame = remove_copies(fragment_similarity_iter(graph_list, depth, mute=mute))
        fragment_DataFrame = fragment_DataFrame.append(inst_DataFrame, ignore_index=True)

    fragment_series = fragment_DataFrame['name'].unique()  # Not relevant for 2 molecules but is for more

    # grouped_nodes_template = pd.DataFrame({'name':fragment_series, mol_graph_1.name: [[]]*len(fragment_series), mol_graph_2.name: [[]]*len(fragment_series)})
    #  [[]]*len(fragment_series) created every entry as an empty list by default, grouped_nodes[mol_graph_1.name] = []
    #  does not work as it tries to open the list

    # TODO: Need to change the way this data is stored, the source fragment becomes irrelevant after this process

    # todo: Change so nodes are group not the actual fragments

    # Instead of groups use the fragment names
    match_data = {fragment_name: {**{molecule.name: [] for molecule in graph_list},
                                  **{'graph': None, 'source_node': None, 'level': None}} for fragment_name in
                  fragment_series}

    def grouping_lambda(entry, match_data):
        ###### TARGET GRAPH == DICT Values
        ###### REFERENCE GRAPH == DICT KEYS  SWITCHED ON 13/1/20
        # These entries will have the same reference graphs as it is based on the fragment
        mappings = entry['mapping']
        if len(mappings) > 0:

            name = entry['name']
            graph = entry['graph']
            source_node = entry['source_node']
            level = entry['level']
            ref_name = entry['reference_name']

            # Only the target will be changing
            target_name = entry['target_name']

            # TODO: make sure nodes aren't doubling up in lists, doesn't affect outcome though but creates cumbersome data
            #  This can be fixed by deleting identical fragments

            # TODO: check if these need to be ordered? The keys that is (and thus the values) yes they do (13/1/20)

            print("Matching entry %s" % name)

            sorted_keys = sorted(mappings[0].keys())
            if match_data[name][ref_name] != 1:
                match_data[name][ref_name].append(sorted_keys)  # TODO: Dodgy workaround needs to be fixed

            for mapping in mappings:  # TODO: create dataFrame for each node
                sorted_values = [mapping[x] for x in sorted_keys]
                match_data[name][target_name].append(sorted_values)

            match_data[name]['graph'] = graph
            match_data[name]['source_node'] = source_node
            match_data[name]['level'] = level
            match_data[name]['reference_name'] = ref_name
        return entry

    for fragment_name in fragment_series:  # Iterates matching over the different fragments that are generated
        fragment_DataFrame.loc[fragment_DataFrame['name'] == fragment_name].apply(grouping_lambda, args=(match_data,),
                                                                                  axis=1)

    # clean the match data # TODO wrewrite so this isnt needed
    for entry in match_data:
        for molecule in graph_list:
            set_of_mappings = set(tuple(x) for x in match_data[entry][molecule.name])
            match_data[entry][molecule] = [list(mapping) for mapping in set_of_mappings]

    mol_dict_template = {molecule.name: [] for molecule in graph_list}
    node_groups = {}

    for fragment_name in fragment_series:  # becomes redundant at this point both fragment series and match_data
        #     # have unique list of names, easier to visualise though and helps with error catching
        #     # Remove the buffered area
        #     # As I'm writing this I'm realising that the match_data object doesn't have enough info to be used so I have added the graph object
        #

        #      TODO: need way to check which molecule is the source molecule
        graph = match_data[fragment_name]['graph']
        source_node = match_data[fragment_name]['source_node']
        buffered_level = match_data[fragment_name]['level'] - fragment_buffer
        reference_name = match_data[fragment_name]['reference_name']
        buffered_fragment = fragment(graph, source_node, buffered_level)
        # create new mapping with only new nodes
        nodes_for_deletion = [node for node in graph.nodes if node not in buffered_fragment.nodes]
        for node in nodes_for_deletion:
            index = match_data[fragment_name][reference_name][0].index(node)
            for molecule in graph_list:
                for mapping in match_data[fragment_name][molecule.name]:
                    del mapping[index]

        # Node grouping code
    group_id = 0
    for key in match_data.keys():
        reference_name = match_data[key]['reference_name']
        ref_nodes = match_data[key][reference_name][0]
        for index, node in enumerate(ref_nodes):  # reference only ever has one list
            node_groups[group_id] = deepcopy(mol_dict_template) # This deepcopy is really, really important (.copy() doesnt work)
            node_groups[group_id][reference_name].append(node)
            for molecule in graph_list:
                if molecule.name != reference_name:
                    for mapping in match_data[key][molecule.name]:
                        node_groups[group_id][molecule.name].append(mapping[index])
            group_id += 1

    # Clean out entries with no information
    node_groups = {index:data for index, data in node_groups.items()  if sum(len(node_list) for node_list in data.values())!=1}

    # for group_id in node_groups.keys():
    #     if sum(len(node_list) for node_list in list(node_groups[group_id].values()))==1: # checks for just the one node:


    # TODO: Remove duplicate node groups and node 'groups' which is just a single group of nodes on one molecule

    return match_data, node_groups


if __name__ == "__main__":
    pass
    W1 = graph_generation(r'Warfarin_PDB_Files/2225_warfarin.pdb')
    W2 = graph_generation(r'Warfarin_PDB_Files/2222_warfarin.pdb')
    W3 = graph_generation(r'Warfarin_PDB_Files/2202_warfarin.pdb')
    W4 = graph_generation(r'Warfarin_PDB_Files/2227_warfarin.pdb')

    # test_frag = fragment(W1, 2, 3)
    #
    # switch_test = list(get_isomorphisms(W1, test_frag))
    #
    # # thingo = get_isomorphisms(W1, W1, mute=0)
    #
    # thingo2 = fragment_similarity_iter([W1,W2], 5, write=False)
    #
    # remove_copies(thingo2, write=False)
    #
    match_data, node_groups = match_buffer_multi([W1, W2], fragment_buffer=1, min_depth=5, mute=False,
                                                 max_depth=10)

    # debug_view_match(test, W1, W2)
    test_view = pd.DataFrame.from_dict(match_data, orient='index')
    node_groups_view = pd.DataFrame.from_dict(node_groups, orient='index')
