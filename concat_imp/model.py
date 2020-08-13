#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : model.py
# @Software: PyCharm
import tensorflow as tf
import math
import os

class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.item = tf.placeholder(dtype=tf.int32)   # 重新编号的序列构成的矩阵
        self.tar = tf.placeholder(dtype=tf.int32)
        self.alias_tar = tf.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
    
    def forward(self, re_embedding, train=True):
        """
        build session embedding
        Input:
            re_embedding : shape = (batch_size, max_n_node, hidden_size)
        """
        # mask shape = (batch_size, maxlen) => rm = (batch_size, )
        rm = tf.reduce_sum(self.mask, 1)
        # last_id : the last click id in each session. shape = (batch_size, )
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm)-1], axis=1))
        # last_h : shape = (100, 100), 100 session's in a batch and each item's embedding is (100,)
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        # seq_h: for each session in a batch, 
        # trivago: (100, 95, 100)
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) \
                          for i in range(self.batch_size)], axis=0) #batch_size * max_len * hidden_size
        
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        if not self.nonhybrid:
            # ma : Sl concat Sg. shape = (batch_size, 2*hidden_size)
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            #self.B : W3 in paper
            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            sess_emb = y1
            # logits = tf.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1)
            sess_emb = ma
            # logits = tf.matmul(ma, b, transpose_b=True)
        return sess_emb # shape = [batch_size, hidden_size]

# inherit model
class GGNN(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN,self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        # trivago: shape=[54608, 100]
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step # gnn propogation steps
        self.nonhybrid = nonhybrid
        # define parameter
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
    
    def ggnn(self):
        """
        get item embedding
        """
        # fin_state: for each element(item) in self.item, get the representative embedding
        # vector which shape = (hidden_size, ) from self.embedding. Therefore,
        # self.embedding shape = (n_node, hidden_size), self.item shape = (batch_size, max_n_node)
        # fin_state shape = (batch_size, max_n_node, hidden_size)
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        # The number of units in the GRU cell = self.out_size = hiddenSize
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                # reshape fin_state to (# of all items in a batch, hidden_size) and multiply
                # W_in (shape(hidden_size, hidden_size)), and add b_in shape(hidden_size, )).
                # So, we will get the result in shape(# of all items in a batch, hidden_size)
                # After that, reshape to (batch_size, max_n_node, hidden_size)
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                # self.adj_in shape = (batch_size, max_n_node, max_n_node)
                # after matmul of adj and fin_state, shape = (batch_size, max_n_node, hidden_size)
                # which can be regard as (max_n_node, max_n_node) x (max_n_node, hidden_size)
                # for each session sequence in a batch.
                # After concat => shape = (batch_size, max_n_node, 2*hidden_size)
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                # the input shape of dynamic_rnn = (batch_size * max_n_node, 1, 2 * hidden_size)
                # the shape of fin_state return from dynamic_rnn = (batch_size x max_n_node, cell.output_size)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])

class Impression_Embedding():
    def __init__(self, hidden_size, batch_size, embedding_len, i_node):
        self.emb_hidden_size = hidden_size
        self.emb_batch_size = batch_size # must be the same as GGNN batch_size
        self.embedding_len = embedding_len
        self.imps = tf.placeholder(dtype=tf.int32) # impressions
        self.i_stdv = 1.0 / math.sqrt(self.emb_hidden_size)
        self.imp_embedding = tf.get_variable(shape=[i_node, hidden_size], name='imp_embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.i_stdv, self.i_stdv))
        self.weight_h1 = tf.get_variable(shape=[hidden_size*embedding_len, hidden_size*embedding_len], name='EMB_w1', 
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-self.i_stdv, self.i_stdv))
        self.bias_h1 = tf.get_variable(shape=[hidden_size*embedding_len], name='EMB_b1', dtype=tf.float32, 
                                       initializer=tf.constant_initializer(0))
    
    def embedding_layer(self):
        # imp_state = (emb_batch_size, embedding_len, hidden_size)
        imp_state = tf.nn.embedding_lookup(self.imp_embedding, self.imps)
        layer_1 = tf.add(tf.matmul(tf.reshape(imp_state, [self.emb_batch_size, -1]), self.weight_h1), self.bias_h1)
        imp_state = tf.nn.relu(layer_1)
        return tf.reshape(imp_state, [self.emb_batch_size, -1, self.emb_hidden_size])
        
class GGNN_Concat_EMB(GGNN, Impression_Embedding):
    def __init__(self,hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False, 
                 embedding_len=25, i_node=None):
        GGNN.__init__(self, hidden_size, out_size, batch_size, n_node, 
                      lr, l2, step, decay, lr_dc, nonhybrid)
        Impression_Embedding.__init__(self, hidden_size, batch_size, embedding_len, i_node)
        self.NN_w1 = tf.get_variable(shape=[self.hidden_size*(self.embedding_len+1), (self.hidden_size/2)*self.embedding_len], 
                                            name='NN_w1', dtype=tf.float32, initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.NN_b1 = tf.get_variable(shape=[(self.hidden_size/2)*self.embedding_len], name='NN_b1', 
                                            dtype=tf.float32, initializer=tf.constant_initializer(0))
        self.NN_w2 = tf.get_variable(shape=[(self.hidden_size/2)*self.embedding_len, self.embedding_len], name='NN_w2', 
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.NN_b2 = tf.get_variable(shape=[self.embedding_len], name='NN_b2', dtype=tf.float32, 
                                       initializer=tf.constant_initializer(0))
        
        with tf.variable_scope('ggnn_model', reuse=None):
            # self.temp_re_embedding = self.ggnn()
            self.loss_train, _ = self.Concat_model()
        with tf.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.score_test = self.Concat_model(train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.8
        #session = tf.Session(config=config )
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer()) # initial all variables
    
    def Concat_model(self, train=True):
        sess_embedding = self.forward(self.ggnn(), train)
        imp_embedding = self.embedding_layer()
        nn_input = tf.concat([tf.reshape(sess_embedding, [self.batch_size, -1, self.hidden_size]), imp_embedding], 1)
        nn_layer1 = tf.add(tf.matmul(tf.reshape(nn_input, [self.batch_size, -1]), self.NN_w1), self.NN_b1)
        nn_layer1 = tf.nn.relu(nn_layer1)
        nn_layer2 = tf.add(tf.matmul(nn_layer1, self.NN_w2), self.NN_b2)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.alias_tar, logits=nn_layer2))
        
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, nn_layer2
    
    # fetches = [model.opt, model.loss_train, model.global_step]
    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask, imps, alias_tar):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in, self.adj_out: adj_out, 
                                                 self.alias: alias, self.mask: mask, self.imps: imps, self.alias_tar: alias_tar})