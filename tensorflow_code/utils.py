#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/23 2:52
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

import networkx as nx
import numpy as np


def build_graph(train_data):
    """
    build the graph for all training data, and count the time of occurrence 
    of item u to item v for all session in all training data as weight. Then 
    update the weight by divided by the indegree of each node.
    """
    # Create an empty graph structure (a “null graph”) with no nodes and no edges.
    graph = nx.DiGraph() # Directed graphs with self loops
    for seq in train_data: # seq: a session list. [item_id, item_id,..., target_id]
        for i in range(len(seq) - 1):
            # get_edge_data: Return the attribute dictionary associated with edge (u,v)
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    # count the sum of weights of each nodes in graph
    for node in graph.nodes:
        sum = 0 
        # Return a list of the incoming edges.
        # j represent u in edge (u,v) and i represent v in edge (u,v)
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            # update the weights of rach nodes
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    """
    find the max length, len_max, in all sessions. Then we add 0 to the
    session which length shorter then len_max. After that, all sequences will
    have the same length. us_pois: output sequences. us_msks: mask list, only 
    consists of 1 and 0.
    """
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
        inputs = data[0] # the data which already split by timestep
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs) # size: (length, maxlen). trivago:(414670, 95)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs) # trivago: 414670
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        # for trivago data: slices : store the index from 0 to 414670
        # and type: list(size 4147) of numpy array(size (100,))
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        # Update the last slice to the last batch_size in real session data index
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        """
        items: list of list, shape = [batch_size, max_n_node]
        A_in: list of numpy array, list shape = batch-size, numpy array shape = (max_n_node, max_n_node)
        A_out: list of numpy array, list shape = batch-size, numpy array shape = (max_n_node, max_n_node)
        alias_inputs: list of list, # of batch-size list with size maxlen(95 for trivago)
        mask: numpy array, shape = (batch_size, maxlen). trivago = (100, 95)
        targets: numpy array, shape = (batch_size, )
        
        """
        if 1:
            # n_node: store the unique item num in every session
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]:
                n_node.append(len(np.unique(u_input)))
            max_n_node = np.max(n_node)
            if self.method == 'ggnn':
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.zeros((max_n_node, max_n_node))
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        # np.where will return (array([], dtype), )
                        # return the index of u_input[i] in node
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    u_sum_in = np.sum(u_A, 0) # add u_A row by row
                    u_sum_in[np.where(u_sum_in == 0)] = 1
                    u_A_in = np.divide(u_A, u_sum_in)
                    u_sum_out = np.sum(u_A, 1)
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)
                    
                    # u_A_in, u_A_out for a session row in a batch
                    # A_in, A_out for a batch
                    A_in.append(u_A_in)
                    A_out.append(u_A_out)
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]
            elif self.method == 'gat':
                A_in = []
                A_out = []
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.eye(max_n_node)
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    A_in.append(-1e9 * (1 - u_A))
                    A_out.append(-1e9 * (1 - u_A.transpose()))
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

        else:
            return self.inputs[index], self.mask[index], self.targets[index]