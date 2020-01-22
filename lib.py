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


import os
import matplotlib.pyplot as plt  # Used for plotting graphs
import networkx as nx  # Used for generating graph objects that can be plotted
from molecular_graphs.lib import pdb_to_graph
from networkx.algorithms import isomorphism as iso
from copy import deepcopy  # Required for copying objects that contain graphs as for example,
# using list.copy() on a list of graphs will copy the list but continue to contain the same graphs
# e.g. b = 'abc',  a = [graph1, graph2, graph3, b]
# graph1.name = 'thisname'
# copy_test = a.copy()
# copy_test[0].name = 'thatname'
# a[0].name
#   'thatname'
# Using deepcopy instead prevents this behaviour
import pandas as pd


def debug_view_match(match_buffer_multi_output, mol_graph_1, mol_graph_2):
    """
    Function used to generate an image of two graphs and and a common fragment mapping found between them.
    Produces an image og both graphs with the mapping highlighted in red
    USed for debugging purposes, to change which mapping the fragment name, and mapping index (if there were several mappings)
        must be changed

    :param match_buffer_multi_output: <dict> output dict from match_buffer_multi that stores the node mapping data
    :param mol_graph_1: <networkx.classes.graph.Graph> One of the graphs used to generate the match_buffer_multi output
    :param mol_graph_2: <networkx.classes.graph.Graph> A different graph used to generate the match_buffer_multi_ouptut
    """
    f, axes = plt.subplots(1, 2, figsize=(10 * 2, 10))  # Create matplotlib axes and figure to they can be updated
    # sperately
    # test case with W1, W2, '2225_warfarin_node_id6_lv5'
    mol_1_layout = nx.spring_layout(mol_graph_1)  # Generate the layout of the graph
    mol_1_labels = nx.get_node_attributes(mol_graph_1, 'label')  # Pull the node labels (atom type) from the graph
    # into a list
    mol_1_mapped_nodes = match_buffer_multi_output['2225_warfarin_node_id6_lv5'][mol_graph_1.name][0]  # Retrieve the
    # specified mapping, here it uses the fragment '2225_warfarin_node_id6_lv5' with the 0th mapping ([0]).
    # note if the source of the fragment is the same as the mol_graph then there will only be 1 mapping, so [0] must be
    # used as the index
    mol_1_unmapped_nodes_color = {i: "#c0c1c2" for i in mol_graph_1.nodes}  # sets every node to be a light
    # grey shade using hexidecimal colours. Puts this information in a dictionary with {node_id: "#c0c1c2" ...}
    mol_1_mapped_nodes_color = {i: 'r' for i in mol_1_mapped_nodes}  # Same as above but only for mapped nodes
    # and sets them to red
    nx.draw_networkx_labels(mol_graph_1, mol_1_layout, mol_1_labels, font_size=16, ax=axes[0])  # Draws the graph labels
    # on one of the matplotlib axes
    nx.draw_networkx_edges(mol_graph_1, mol_1_layout, ax=axes[0])  # Draws the edges
    nx.draw_networkx_nodes(mol_graph_1, mol_1_layout, ax=axes[0],
                           node_color=list({**mol_1_unmapped_nodes_color, **mol_1_mapped_nodes_color}.values()),
                           nodelist=list({**mol_1_unmapped_nodes_color, **mol_1_mapped_nodes_color}.keys()))
    # Draws the nodes. When using node colour the node_colour dict is merged from the two colour dicts with the
    # (mol_1_unmapped_nodes_color and mol_1_mapped_nodes_color) with the mapped nodes taking priority, so the
    # red colour overwrites the grey. The same is done for the node_list to ensure the values and nodes line up, as
    # dictionaries can't be input, only lists. Thus the following format has to be ensured:
    # nodelist=[1,2,3,4], node_colour = [color1, color2, color3, color4]

    # See above comments for the below code
    mol_2_layout = nx.spring_layout(mol_graph_2)
    mol_2_labels = nx.get_node_attributes(mol_graph_2, 'label')
    mol_2_mapped_nodes = match_buffer_multi_output['2225_warfarin_node_id6_lv5'][mol_graph_2.name][0]
    mol_2_unmapped_nodes_color = {i: "#c0c1c2" for i in mol_graph_2.nodes}
    mol_2_mapped_nodes_color = {i: 'r' for i in mol_2_mapped_nodes}
    nx.draw_networkx_labels(mol_graph_2, mol_2_layout, mol_2_labels, font_size=16, ax=axes[1])
    nx.draw_networkx_edges(mol_graph_2, mol_2_layout, ax=axes[1])
    nx.draw_networkx_nodes(mol_graph_2, mol_2_layout, ax=axes[1],
                           node_color=list({**mol_2_unmapped_nodes_color, **mol_2_mapped_nodes_color}.values()),
                           nodelist=list({**mol_2_unmapped_nodes_color, **mol_2_mapped_nodes_color}.keys()))

    plt.show()  # Displays the plot


class NodeMatcher:
    """
    Class for storing node matching check functions used in get_isomorphisms
    """

    @staticmethod
    def label(n1, n2):
        """
        standard check for labels as atom type information is stored under "label"
        :param n1: <int> node_id 1
        :param n2: <int> node_id 2
        """
        return n1["label"] == n2["label"]


def graph_generation(pdb_path):
    """
    Function to generate networkx Graph objects from pdb files
    :param pdb_path: <str> path to pdb file
    :return: <networkx.classes.graph.Graph> Graph object of entered molecule, with appropriate node id's and labels,
    and name property corresponding to the file name used on the path.
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
    :param mute: <Boolean> silence printed information of this function
    :return: <list<dict>> TARGET node id's == KEYS, REFERENCE node id's == VALUES in the dictionaries
    Each dictionary is a single mapping found between the reference and the target
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
    return mappings  # As a generator is faster but single use. List is more versatile to use


def fragment(G, source_node_id, steps):
    """
    Finds a subgraph starting at source_node_id that extends the number of specified steps away
    :param G: <networkx.classes.graph.Graph>, must have been imported from a pdb using graph_generation and have the
    label attribute
    :param source_node_id: <int> Starting node
    :param steps: <int>, number of steps to take
    :return: <networkx.classes.graph.Graph> Returns the fragment as a graph
    """

    assert source_node_id in G.nodes  # checks the node id is valid

    def recursive_fragment(G, H, source_node, n):
        """
        recursive function used to walk through graph. Creates the fragment as it goes by copying data to a new graph
        :param source_node: <int> Current node id of the recursive function
        :param n: current step, counted down from param steps of parent function
        Returns NoneType as the result, H is mutable and changes are saved as they occur
        :return: NoneType
        """
        if n != 0:  # out of steps
            for i in G.neighbors(source_node):  # iterates over every nodes neighbours
                if not H.has_edge(source_node, i):  # stops the tree algorithm going back on itself by checking if
                    # it has already made a path
                    H.add_edge(source_node, i)  # this will also add the node to the graph H
                    H.nodes[i]['label'] = G.nodes[i]['label']  # copies over the element type
                    recursive_fragment(G, H, i, n - 1)  # rerun with one less step size

    H = nx.Graph({source_node_id: {}})  # Creates Graph of one node with given node id
    nx.set_node_attributes(H, '', 'label')  # initialises empty labels
    H.nodes[source_node_id]['label'] = G.nodes[source_node_id][
        'label']  # sets the correct element type of the initial node
    recursive_fragment(G, H, source_node_id, steps)  # runs the recursive function
    H.name = G.name + '_node_id' + str(source_node_id) + '_lv' + str(
        steps)  # sets the name to indicate how it was created, '_frag_id' changed to '_node_id'
    return H  # returns the fragment graph


def display_graph(graph, save_path=None):
    """
    Displays the graph with element types and optionally saves the file
    saves a few line of code
    used for debugging
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
    :param write: <boolean> If True, saves the output to a file
    :param mute: Allows/stops the printing of function progress updates
    :return: <dict> of mappings and writes file with mappings
    """

    # TODO: add check for identical fragments (node id and all)) all line should be good

    fragments = pd.DataFrame(columns=['name', 'reference_name', 'target_name', 'mapping', 'graph', 'source_node',
                                      'level'])  # Initialises empty Dataframe for the fragments as they are created
    # TODO: Consider moving this loop outside this function
    for mol_graph_ref in mol_graph_list:  # iterates over every inputted molecule
        for node_id in mol_graph_ref.nodes:  # iterates over every atom
            current_frag = fragment(mol_graph_ref, node_id, level)  # creates a fragment using the inputted depth
            # if not any([nx.is_isomorphic(x, current_frag) for x in list(fragments['graph'])]):
            # TODO: See if this previous line needs to be implemented
            # TODO: These lines deal with intermolecular mappings, so far just included the source molecule in the loop
            #  to compare against itself to account for intramolecular mappings
            for mol_graph_target in mol_graph_list:  # reiterates over every molecule
                fragments = fragments.append({'reference_name': mol_graph_ref.name,
                                              'target_name': mol_graph_target.name,
                                              'name': current_frag.name,
                                              'mapping': get_isomorphisms(mol_graph_target, current_frag, mute=mute),
                                              'graph': current_frag,
                                              'source_node': node_id,
                                              'level': level},
                                             ignore_index=True
                                             )

                # adds the mappings generated by get_isomorphisms and other required data assoicated with the molecule
                # to the DataFrame
                # currently does not remove symmetrical maps, adds the maps to the mapping list and is removed later
                # Not sure if all this data is needed at this point, something to look at when optimising
    fragments[['source_node', 'level']] = fragments[['source_node', 'level']].astype(
        int)  # changes these two columns to
    # be the correct variable type
    if write:  # see docstring
        # This section deals with determining the file name, adds (<int>) similar to saving files on windows
        if save_path is None:  # Creates a file name from the Graph names
            save_path = ''.join(x.name[0:5] for x in mol_graph_list) + "_lv_" + str(level) + '.xlsx'
        else:
            save_path += ".xlsx"
        i = 1  # Adding numbers to end of files, checks in a while loop
        while os.path.exists(save_path + " (%s).xlsx" % i):
            i += 1
        save_path = save_path + " (%s).xlsx" % i
        print(save_path)
        fragments.to_excel(save_path)  # uses pandas built in DataFrame to excel method
        print("Note Graph objects are not saved")  # The graph object data stored in the DataFrame cannot be
        # saved in this way

    return fragments  # Returns the fragments DataFrame


def remove_copies(fragment_DataFrame, save_path=None, write=False):
    """
    Passes by value. \n
    Checks for 'symmetric' mapping by checking if a mapping has the same set of target and reference maps, irrespective
    of how they are paired, i.e these dictionaries are copies:
    {1:2, 2:3, 3:1}, {1:3, 2:1, 3:2}.



    This makes assumptions about the way mappings work, it does not check if number occurs twice for example as they shouldn't.
    If it takes a mapping where this is the case it will not account for it.


    Assumptions:
     - nodes can't have two mappings in one mapping dict
     - graphs can't be structured such that a copied mapping can actually exist as unique structures
        (this is true for chemicals, see lab book)
     - # TODO demonstrate all previous points in lab book
    :param fragment_DataFrame: mapping output generated by fragment_similarity_iter, currently can't read a saved list
    excel sheet generated by same function # TODO add functionality to load from file
    :return <dict> of mappings and writes file with mappings:
    """

    def remove_copies_lambda(mappings_list):  # function applied to every row in the fragment_DataFrame
        if len(mappings_list) > 1:  # checks there are 2 or more mappings, otherwise this process is unneeded
            maps_cleaned = []
            # TODO Add visualisation in lab book for next step

            # Probably a better way to structure the following code
            for j, map_1 in enumerate(mappings_list):
                # j is the index out of the current mappings, map_1 is the mapping dictionary.
                # iterating over the original maps list
                inst_keyset = set(map_1.keys())
                # stores the unique set (in this case just orders it, as they should never double up)
                # of the keys which correspond to the nodes being mapped too (on the target from the fragment)

                to_check_cond = not inst_keyset in [set(x.keys()) for x in
                                                    [q for p, q in enumerate(mappings_list) if p != j]]
                # stores the condition, seeing if the key set is in a list of the key sets from the old mappings
                # this list [q for p, q in enumerate(i['mapping']) if p != j] is the same list of old mappings
                # excluding map_1
                # done by saying when p (index of new list) is not the same as j

                past_check_cond = not inst_keyset in [set(x.keys()) for x in maps_cleaned]
                # stores the condition, seeing if the keyset is in a list of the keysets already added to the new maps
                if to_check_cond or past_check_cond:  # combines the previous conditions
                    maps_cleaned.append(map_1)  # adds the current map if it passes the tests
            return maps_cleaned  # replaces the maps in the row being apllied to
        else:
            return mappings_list

    # Since different entries have the same node these need to be sorted by their source molecule and target molecule
    # It does this per entry as different fragments though identical may have different buffer regions
    # example code:
    # test_graph = nx.path_graph(5)
    # fragment1 = fragment(test_graph, 3, 2)
    # fragment2 = fragment(test_graph, 3, 6)
    # Though both fragments are the same graph fragment one would buffer the edges of the line but fragment 2 would not
    # TODO: Demonstrate this example

    fragment_DataFrame['mapping'] = fragment_DataFrame['mapping'].apply(remove_copies_lambda)
    # applies the previous function

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

    return fragment_DataFrame  # returns the 'cleaned' fragment_DataFrame


def match_buffer_multi(graph_list, fragment_buffer=1, min_depth=3, max_depth=5, mute=True):
    """
    :param graph_list: <list<networkx.classes.graph.Graph>> A list of the graph objects generated from the molecules
    being tested

    :param fragment_buffer: <int> How far from the central atom a fragment mapping is consider a constraint
    recommended value is min_depth-1

    :param min_depth: <int> minimum recursive depth for generating fragments

    :param max_depth: <int> maximum recursive depth for generating fragments

    :parm mute" <boolean> Specified if progress should be printed
    """

    if min_depth - fragment_buffer != 1:  # Warning about fragment buffer
        print('Warning, fragment_buffer is not the recommended value')

    fragment_DataFrame = pd.DataFrame(
        columns=['name', 'reference_name', 'target_name', 'mapping', 'graph', 'source_node', 'level'])
    # Initialises empty dataframe to store the fragment data that is generated by fragment_similarity_iter.
    # It run multiple times so the data needs to be concatenated into one object

    for depth in list(range(min_depth, max_depth + 1))[::-1]:  # Short albeit moderately unreadable code, creates
        # a reversed list to iterate through, starting at the maximum size of the molecule
        inst_DataFrame = remove_copies(fragment_similarity_iter(graph_list, depth, mute=mute))
        # DataFrame created using parameters (depth) of current loop
        fragment_DataFrame = fragment_DataFrame.append(inst_DataFrame, ignore_index=True)
        # updates the fragment_DataFrame with the data from this loop

    fragment_series = fragment_DataFrame['name'].unique()  # Not relevant for 2 molecules but is for more
    # creates a series with the different fragment names

    # TODO: Need to change the way this data is stored, the source fragment becomes irrelevant after this process

    # todo: Change so nodes are group not the actual fragments
    # This is still done later but would be more efficient to do it as the data is created

    # Instead of different entries for the same fragments, categorise by the fragments thus grouping together molecules
    # i.e From: fragmentA: molecule1_mapping1, molecule1_mapping2
    #           fragmentA: molecule2_mapping
    # to:       fragmentA: [molecule1: [mapping1, mapping2], molecule2: [mapping1]]

    match_data = {fragment_name: {**{molecule.name: [] for molecule in graph_list},
                                  **{'graph': None, 'source_node': None, 'level': None}} for fragment_name in
                  fragment_series}

    # initialises the empty dictionary for storing the grouped fragment data

    def grouping_lambda(entry, match_data):  # function to be applied to each entry of fragment_DataFrame
        # TARGET GRAPH == DICT Values
        # REFERENCE GRAPH == DICT KEYS  SWITCHED ON 13/1/20
        # These entries will have the same reference graphs as it is based on the fragment
        mappings = entry['mapping']  # Stored the current rows mappings
        if len(mappings) > 0:  # checks if the row has any mappings

            name = entry['name']  # Temporarily stores the rows data
            graph = entry['graph']
            source_node = entry['source_node']
            level = entry['level']
            ref_name = entry['reference_name']

            # Only the target will be changing
            target_name = entry['target_name']

            # TODO: make sure reference nodes aren't doubling up in lists, doesn't affect outcome though but creates
            #  cumbersome data
            #  This can be fixed by deleting identical fragments

            print("Matching entry %s" % name)

            sorted_keys = sorted(mappings[0].keys())  # creates a list of the sorted keys to sort the different molecule
            # mapping by
            if match_data[name][ref_name] != 1:
                match_data[name][ref_name].append(sorted_keys)  # TODO: Workaround needs to be fixed

            for mapping in mappings:  # TODO: create dataFrame for each node as better storage method
                sorted_values = [mapping[x] for x in sorted_keys]  # Sorts the target nodes based on the sorted keys
                match_data[name][target_name].append(sorted_values)  # adds these sorted values

            match_data[name]['graph'] = graph  # copies over the data to the new match_data object
            match_data[name]['source_node'] = source_node
            match_data[name]['level'] = level
            match_data[name]['reference_name'] = ref_name
        return entry

    for fragment_name in fragment_series:  # Iterates matching over the different fragments that are generated
        fragment_DataFrame.loc[fragment_DataFrame['name'] == fragment_name].apply(grouping_lambda, args=(match_data,),
                                                                                  axis=1)
        # .loc find the part of the data with the current fragment name and applies the previous function

    # clean the match data # TODO wrewrite entire script so this isn't needed
    for entry in match_data:  # iterates over every entry in the matching data
        for molecule in graph_list:  # does this process for every molecule being analysed
            set_of_mappings = set(tuple(x) for x in match_data[entry][molecule.name])
            # Creates a tuple from each list of mappings as these can be turned into a unique set
            match_data[entry][molecule] = [list(mapping) for mapping in set_of_mappings]
            # Converts this set back into a list

    mol_dict_template = {molecule.name: [] for molecule in graph_list}
    # A template to use for initialising the dictionary with lists of associated mappings

    node_groups = {}  # Initialises an empty dictionary to store the node group data in

    for fragment_name in fragment_series:  # becomes redundant at this point both fragment series and match_data
        # have unique list of names, easier to visualise though and helps with error catching
        # Remove the buffered area

        # Following code copies over the relevant information into temporary variables
        graph = match_data[fragment_name]['graph']
        source_node = match_data[fragment_name]['source_node']
        buffered_level = match_data[fragment_name]['level'] - fragment_buffer
        reference_name = match_data[fragment_name]['reference_name']
        buffered_fragment = fragment(graph, source_node, buffered_level)
        # The buffered fragment is a fragment created to be smaller then the original with the difference in steps being
        # the fragment_buffer variable

        # create new mapping with only new nodes
        # Does this by removing all node_id's not in the buffered fragment and removing their associated mappings

        nodes_for_deletion = [node for node in graph.nodes if node not in buffered_fragment.nodes]
        # Statement to check if the node id is in the buffered fragment, iterates over the nodes in the original
        # fragment
        for node in nodes_for_deletion:  # Separately iterates over these identified nodes
            index = match_data[fragment_name][reference_name][0].index(node)  # Finds the list index of the current node
            # e.g [0,5,2,3,7].index(5) returns 1
            for molecule in graph_list:  # Iterates over every molecule in the entry
                for mapping in match_data[fragment_name][molecule.name]:  # Does the following step for every mapping
                    del mapping[index]  # Deletes the nodes mapped to the removed node

        # Node grouping code
    group_id = 0  # Starts the group id's at 0
    for key in match_data.keys():  # iterates over every molecule, molecules are keys of this dictionary

        # copies out relevant data into temporary variables
        reference_name = match_data[key]['reference_name']
        ref_nodes = match_data[key][reference_name][0]  # Reference nodes are always one list
        for index, node in enumerate(ref_nodes):  # reference only ever has one list, index is used to index the other
            # mappings
            node_groups[group_id] = deepcopy(
                mol_dict_template)  # This deepcopy is really, really important (.copy() doesnt work), see import
            # section
            node_groups[group_id][reference_name].append(node) # adds the reference node to the appropriate molecule
            # section
            for molecule in graph_list: # repeats the same process for each other molecules,
                # the following if check should not be necessary in a fully optimised version
                if molecule.name != reference_name:  # skips the reference molecule as this is done previously
                    for mapping in match_data[key][molecule.name]:  # for every mapping in each molecule
                        node_groups[group_id][molecule.name].append(mapping[index])  # indexes the mappings associated
                        # with the reference node and adds them to the node group
            group_id += 1  # changes the group id each loop

    # Clean out entries with no information
    node_groups = {index: data for index, data in node_groups.items() if
                   sum(len(node_list) for node_list in data.values()) != 1}

    # remove node double ups (Shouldn't be necessary)
    node_groups = {index: {index2: list(set(data2)) for index2, data2 in data.items()} for index, data in
                   node_groups.items()}

    # remove identical entries

    node_groups = {key: node_groups[key] for i, key in enumerate(node_groups) if
                   node_groups[key] not in [node_groups[key2] for key2 in list(node_groups.keys())[i + 1:]]}

    # remove empty node list for next step in temp object,
    # the following step check id a node groups is a subset of another e.g.:
    # group 1: mol1:[2], mol2:[2], mol3:[] is a subset of group 1: mol1:[2], mol2:[2], mol3:[4]
    #
    # However these are not recognised, only the following case is recognised:
    # {mol1:[2], mol2:[2]} is a subset of {mol1:[2], mol2:[2], mol3:[4]}
    # so the empty list entries must be deleted

    node_groups_emptied = {key: {mol_name: node_groups[key][mol_name] for
                                 mol_name in node_groups[key] if node_groups[key][mol_name] != []} for key in
                           node_groups}

    def list_of_dicts_without(index):
        """
        Workaround function to return all the dictionaries without the specified index
        """
        return [value for key, value in node_groups.items() if key != index]


    # Removes the subgroup node_groups. Not removing these does not change the output of the charge fitting though
    node_groups = {key: node_groups[key] for key in node_groups if not
    any([node_groups_emptied[key].items() <= node_map.items() for node_map in list_of_dicts_without(key)])}

    # TODO: Remove duplicate node groups and node 'groups' which is just a single group of nodes on one molecule

    return match_data, node_groups


def write_node_groups(node_groups, file_name):
    """
    Writes the output to comply with formatting for Visscher's charge fitting code
    """
    with open(file_name, 'w') as file:
        for key in node_groups:
            file.write('group_' + str(key) + ':\n')
            for mol_name in node_groups[key]:
                file.write('\t')
                file.write(mol_name + ': ')
                file.write(str(node_groups[key][mol_name]))
                file.write('\n')


if __name__ == "__main__":

    # Some test Cases


    # Load in the warfarin molecules
    W1 = graph_generation(r'Warfarin_PDB_Files/2225_warfarin.pdb')
    W2 = graph_generation(r'Warfarin_PDB_Files/2222_warfarin.pdb')
    W3 = graph_generation(r'Warfarin_PDB_Files/2202_warfarin.pdb')
    W4 = graph_generation(r'Warfarin_PDB_Files/2227_warfarin.pdb')

    # test_frag = fragment(W1, 2, 3)
    # Test to generate a fragment



    match_data, node_groups = match_buffer_multi([W1, W2, W3, W4], fragment_buffer=3, min_depth=4, mute=True,
                                                 max_depth=5)
    # Compare the 4 loaded molecules, using depths of 4 to 5 and a buffer of 3. Output is supressed


    # debug_view_match(match_data, W1, W2)
    # Check a mapping between W1 and W2


    # node_groups_view = pd.DataFrame.from_dict(node_groups, orient='index')
    # Visualise the node_groups as its own DataFrame

    # save the formatted file with the constraints as 'W1W2W3W5.txt'
    write_node_groups(node_groups, 'W1W2W3W5.txt')
