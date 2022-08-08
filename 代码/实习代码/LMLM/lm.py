#!/usr/bin/python3

import tensorflow as tf


class LM(object):
    """
    RNNLM with LSTM + Dropout
    """
    def __init__(self, args, is_training, is_rescoring=False):
        self.batch_size = batch_size = args['batch_size']
        self.cell_type = cell_type = args['cell_type']
        rnn_size = args['rnn_size']
        # biderectional = args['biderectional']
        num_layers = args['num_layers']
        use_projection = args['use_projection']
        projection_dim = args['projection_dim']
        cell_clip = args['cell_clip']
        proj_clip = args['proj_clip']
        use_peepholes = args['use_peepholes']
        hidden_size = args['hidden_size']
        word_vocab_size = args['vocab_size']
        keep_prob = args['keep_prob']
        out_vocab_size = word_vocab_size + 1
        # placeholders for data
        self._input_data = tf.placeholder(tf.int32, shape=[batch_size, None], name="frozen_input")

        self._input_data_nbest_1 = tf.placeholder(tf.int32, shape=[batch_size, None], name="frozen_input_nbest")
        self._input_data_nbest_2 = tf.placeholder(tf.int32, shape=[batch_size, None], name="frozen_input_nbest")
        self._input_data_nbest_3 = tf.placeholder(tf.int32, shape=[batch_size, None], name="frozen_input_nbest")
        self._input_data_nbest_4 = tf.placeholder(tf.int32, shape=[batch_size, None], name="frozen_input_nbest")
        # input(self._input_data)
        self._input_data_reverse = tf.placeholder(tf.int32, shape=[batch_size, None], name="frozen_input_reverse")
        input_mask = self._input_data > 0
        input_mask_nbest_1 = self._input_data_nbest_1 > 0
        input_mask_nbest_2 = self._input_data_nbest_2 > 0
        input_mask_nbest_3 = self._input_data_nbest_3 > 0
        input_mask_nbest_4 = self._input_data_nbest_4 > 0

        self._sequence_length = tf.reduce_sum(tf.cast(input_mask, tf.int32), axis=1)
        self._sequence_length_nbest_1 = tf.reduce_sum(tf.cast(input_mask_nbest_1, tf.int32), axis=1)
        self._sequence_length_nbest_2 = tf.reduce_sum(tf.cast(input_mask_nbest_2, tf.int32), axis=1)
        self._sequence_length_nbest_3 = tf.reduce_sum(tf.cast(input_mask_nbest_3, tf.int32), axis=1)
        self._sequence_length_nbest_4 = tf.reduce_sum(tf.cast(input_mask_nbest_4, tf.int32), axis=1)

        self._targets = tf.placeholder(tf.int32, shape=[batch_size, None])

        self._targets_nbest_1 = tf.placeholder(tf.int32, shape=[batch_size, None])
        self._targets_nbest_2 = tf.placeholder(tf.int32, shape=[batch_size, None])
        self._targets_nbest_3 = tf.placeholder(tf.int32, shape=[batch_size, None])
        self._targets_nbest_4 = tf.placeholder(tf.int32, shape=[batch_size, None])

        # input
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [out_vocab_size, hidden_size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
            inputs_nbest_1 = tf.nn.embedding_lookup(embedding, self._input_data_nbest_1)
            inputs_nbest_2 = tf.nn.embedding_lookup(embedding, self._input_data_nbest_2)
            inputs_nbest_3 = tf.nn.embedding_lookup(embedding, self._input_data_nbest_3)
            inputs_nbest_4 = tf.nn.embedding_lookup(embedding, self._input_data_nbest_4)
            if is_training and args['keep_prob'] < 1:
                self.debug_0 = inputs = tf.nn.dropout(inputs, keep_prob)
                self.debug_1 = inputs_nbest_1 = tf.nn.dropout(inputs_nbest_1, keep_prob)
                self.debug_2 = inputs_nbest_2 = tf.nn.dropout(inputs_nbest_2, keep_prob)
                self.debug_3 = inputs_nbest_3 = tf.nn.dropout(inputs_nbest_3, keep_prob)
                self.debug_4 = inputs_nbest_4 = tf.nn.dropout(inputs_nbest_4, keep_prob)

        # language model network
        # input(inputs_nbest)
        multi_cell = []
        for _ in range(num_layers):
            if use_projection:
                cell = tf.nn.rnn_cell.LSTMCell(rnn_size,
                                                use_peepholes=use_peepholes,
                                                num_proj=projection_dim,
                                                cell_clip=cell_clip,
                                                proj_clip=proj_clip)
            else:
                if cell_type.lower() == "gru":
                    cell = tf.nn.rnn_cell.GRUCell(rnn_size)
                elif cell_type.lower() == "lstm":
                    cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
                else:
                    cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
            if is_training and args['keep_prob'] < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            multi_cell.append(cell)
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell(multi_cell, state_is_tuple=True)
            #initialize cell state
        self._initial_lm_state = stacked_cell.zero_state(batch_size, tf.float32)
        # self._initial_lm_state_nbest = stacked_cell.zero_state(batch_size*4, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(stacked_cell,
                                            inputs,
                                            initial_state=self._initial_lm_state,
                                            sequence_length=self._sequence_length)
        outputs_nbest_1, state = tf.nn.dynamic_rnn(stacked_cell,
                                            inputs_nbest_1,
                                            initial_state=self._initial_lm_state,
                                            sequence_length=self._sequence_length_nbest_1)
        outputs_nbest_2, state = tf.nn.dynamic_rnn(stacked_cell,
                                            inputs_nbest_2,
                                            initial_state=self._initial_lm_state,
                                            sequence_length=self._sequence_length_nbest_2)
        outputs_nbest_3, state = tf.nn.dynamic_rnn(stacked_cell,
                                            inputs_nbest_3,
                                            initial_state=self._initial_lm_state,
                                            sequence_length=self._sequence_length_nbest_3)
        outputs_nbest_4, state = tf.nn.dynamic_rnn(stacked_cell,
                                            inputs_nbest_4,
                                            initial_state=self._initial_lm_state,
                                            sequence_length=self._sequence_length_nbest_4)
        #compute logits

        outputs = tf.reshape(outputs, [-1, projection_dim])
        outputs_nbest_1 = tf.reshape(outputs_nbest_1, [-1, projection_dim])
        outputs_nbest_2 = tf.reshape(outputs_nbest_2, [-1, projection_dim])
        outputs_nbest_3 = tf.reshape(outputs_nbest_3, [-1, projection_dim])
        outputs_nbest_4 = tf.reshape(outputs_nbest_4, [-1, projection_dim])
        softmax_w = tf.transpose(embedding, perm=[1, 0])
        softmax_b = tf.get_variable(name="softmax_b",
                                    shape=[out_vocab_size],
                                    initializer=tf.constant_initializer())
        self._logits = logits = tf.matmul(outputs, softmax_w) + softmax_b
        self._logits_nbest_1 = logits_nbest_1 = tf.matmul(outputs_nbest_1, softmax_w) + softmax_b
        self._logits_nbest_2 = logits_nbest_2 = tf.matmul(outputs_nbest_2, softmax_w) + softmax_b
        self._logits_nbest_3 = logits_nbest_3 = tf.matmul(outputs_nbest_3, softmax_w) + softmax_b
        self._logits_nbest_4 = logits_nbest_4 = tf.matmul(outputs_nbest_4, softmax_w) + softmax_b
        # self._softmax = softmax = tf.nn.softmax(logits, dim=1)
        # input(logits_nbest)
        self._final_state = state
        self.embedding = embedding
        # softmax
        if is_rescoring:
            self._softmax = softmax = tf.nn.softmax(logits, dim=1)
            frozen_softmax = tf.identity(softmax, name="frozen_softmax")
            return
        labels = tf.reshape(tf.one_hot(self._targets, out_vocab_size, dtype=tf.int32), [-1, out_vocab_size])
        labels_nbest_1 = tf.reshape(tf.one_hot(self._targets_nbest_1, out_vocab_size, dtype=tf.int32), [-1, out_vocab_size])
        labels_nbest_2 = tf.reshape(tf.one_hot(self._targets_nbest_2, out_vocab_size, dtype=tf.int32), [-1, out_vocab_size])
        labels_nbest_3 = tf.reshape(tf.one_hot(self._targets_nbest_3, out_vocab_size, dtype=tf.int32), [-1, out_vocab_size])
        labels_nbest_4 = tf.reshape(tf.one_hot(self._targets_nbest_4, out_vocab_size, dtype=tf.int32), [-1, out_vocab_size])
        self._labels = labels = tf.cast(labels, tf.float32)
        self._labels_nbest_1 = labels_nbest_1 = tf.cast(labels_nbest_1, tf.float32)
        self._labels_nbest_2 = labels_nbest_2 = tf.cast(labels_nbest_2, tf.float32)
        self._labels_nbest_3 = labels_nbest_3 = tf.cast(labels_nbest_3, tf.float32)
        self._labels_nbest_4 = labels_nbest_4 = tf.cast(labels_nbest_4, tf.float32)
        
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=tf.reshape(logits, [-1, out_vocab_size]))
        cost_nbest_1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels_nbest_1, logits=tf.reshape(logits_nbest_1, [-1, out_vocab_size]))
        cost_nbest_2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels_nbest_2, logits=tf.reshape(logits_nbest_2, [-1, out_vocab_size]))
        cost_nbest_3 = tf.nn.softmax_cross_entropy_with_logits(labels=labels_nbest_3, logits=tf.reshape(logits_nbest_3, [-1, out_vocab_size]))
        cost_nbest_4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels_nbest_4, logits=tf.reshape(logits_nbest_4, [-1, out_vocab_size]))
       
        
        cost = tf.reshape(cost, [-1, tf.shape(self._targets)[1]])
        cost_nbest_1 = tf.reshape(cost_nbest_1, [-1, tf.shape(self._targets_nbest_1)[1]])
        cost_nbest_2 = tf.reshape(cost_nbest_2, [-1, tf.shape(self._targets_nbest_2)[1]])
        cost_nbest_3 = tf.reshape(cost_nbest_3, [-1, tf.shape(self._targets_nbest_3)[1]])
        cost_nbest_4 = tf.reshape(cost_nbest_4, [-1, tf.shape(self._targets_nbest_4)[1]])

        self._mask = loss_mask = tf.sequence_mask(self._sequence_length, tf.shape(self._targets)[1], dtype=tf.float32)
        self._mask_nbest_1 = loss_mask_nbest_1 = tf.sequence_mask(self._sequence_length_nbest_1, tf.shape(self._targets_nbest_1)[1], dtype=tf.float32)
        self._mask_nbest_2 = loss_mask_nbest_2 = tf.sequence_mask(self._sequence_length_nbest_2, tf.shape(self._targets_nbest_2)[1], dtype=tf.float32)
        self._mask_nbest_3 = loss_mask_nbest_3 = tf.sequence_mask(self._sequence_length_nbest_3, tf.shape(self._targets_nbest_3)[1], dtype=tf.float32)
        self._mask_nbest_4 = loss_mask_nbest_4 = tf.sequence_mask(self._sequence_length_nbest_4, tf.shape(self._targets_nbest_4)[1], dtype=tf.float32)

        cost *= loss_mask
        cost_nbest_1 *= loss_mask_nbest_1
        cost_nbest_2 *= loss_mask_nbest_2
        cost_nbest_3 *= loss_mask_nbest_3
        cost_nbest_4 *= loss_mask_nbest_4

        self._cost = tf.reduce_sum(cost,1)
        self._cost_nbest_1 = tf.reduce_sum(cost_nbest_1,1)
        self._cost_nbest_2 = tf.reduce_sum(cost_nbest_2,1)
        self._cost_nbest_3 = tf.reduce_sum(cost_nbest_3,1)
        self._cost_nbest_4 = tf.reduce_sum(cost_nbest_4,1)

        loss1 = 5 - tf.subtract(self._cost,self._cost_nbest_1)
        ones = tf.ones_like(loss1)
        zeros = tf.zeros_like(loss1)
        loss1_1 = tf.where(loss1>0, ones, zeros)
        loss1 =  tf.multiply(loss1,loss1_1)

        loss2 = 5 - tf.subtract(self._cost,self._cost_nbest_2)
        ones = tf.ones_like(loss2)
        zeros = tf.zeros_like(loss2)
        loss2_2 = tf.where(loss2>0, ones, zeros)
        loss2 =  tf.multiply(loss2,loss2_2)

        loss3 = 5 - tf.subtract(self._cost,self._cost_nbest_3)
        ones = tf.ones_like(loss3)
        zeros = tf.zeros_like(loss3)
        loss3_3 = tf.where(loss3>0, ones, zeros)
        loss3 =  tf.multiply(loss3,loss3_3)

        loss4 = 5 - tf.subtract(self._cost,self._cost_nbest_4)
        ones = tf.ones_like(loss4)
        zeros = tf.zeros_like(loss4)
        loss4_4 = tf.where(loss4>0, ones, zeros)
        loss4 =  tf.multiply(loss4,loss4_4)

        t1 = tf.add(loss1,loss2)
        t2 = tf.add(t1,loss3)
        t3 = tf.add(t2,loss4)

        self._loss = tf.reduce_sum(t3)

    # def hinge_loss(self,log_probs,log_probs_nbest):
        
    #     loss = 0
    #     for i in range(0,len(log_probs)):
    #         for j in range(0,4):
    #             loss+= max(0,log_probs[i] - log_probs_nbest[4*i+j])
    #     return loss

    @property
    def loss(self):
        return self._loss

    @property
    def mask(self):
        return self._mask

    def mask_nbest_1(self):
        return self._mask_nbest_1

    def mask_nbest_2(self):
        return self._mask_nbest_2

    def mask_nbest_3(self):
        return self._mask_nbest_3

    def mask_nbest_4(self):
        return self._mask_nbest_4

    @property
    def cost(self):
        return self._cost

    @property
    def cost_nbest_1(self):
        return self._cost_nbest_1

    @property
    def cost_nbest_2(self):
        return self._cost_nbest_2

    @property
    def cost_nbest_3(self):
        return self._cost_nbest_3

    @property
    def cost_nbest_4(self):
        return self._cost_nbest_4

    @property
    def labels(self):
        return self._labels

    @property
    def labels_nbest_1(self):
        return self._labels_nbest_1

    @property
    def labels_nbest_2(self):
        return self._labels_nbest_2

    @property
    def labels_nbest_3(self):
        return self._labels_nbest_3

    @property
    def labels_nbest_4(self):
        return self._labels_nbest_4

    @property
    def logits(self):
        return self._logits

    @property
    def logits_nbest_1(self):
        return self._logits_nbest_1

    @property
    def logits_nbest_2(self):
        return self._logits_nbest_2

    @property
    def logits_nbest_3(self):
        return self._logits_nbest_3

    @property
    def logits_nbest_4(self):
        return self._logits_nbest_4

    @property
    def input_data(self):
        return self._input_data

    @property
    def input_data_nbest_1(self):
        return self._input_data_nbest_1

    @property
    def input_data_nbest_2(self):
        return self._input_data_nbest_2

    @property
    def input_data_nbest_4(self):
        return self._input_data_nbest_4

    @property
    def input_data_nbest_3(self):
        return self._input_data_nbest_3



    @property
    def input_data_reverse(self):
        return self._input_data_reverse

    @property
    def targets(self):
        return self._targets

    @property
    def targets_nbest_1(self):
        return self._targets_nbest_1

    @property
    def targets_nbest_2(self):
        return self._targets_nbest_2

    @property
    def targets_nbest_3(self):
        return self._targets_nbest_3

    @property
    def targets_nbest_4(self):
        return self._targets_nbest_4



    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def sequence_length_nbest_1(self):
        return self._sequence_length_nbest_1

    @property
    def sequence_length_nbest_2(self):
        return self._sequence_length_nbest_2

    @property
    def sequence_length_nbest_3(self):
        return self._sequence_length_nbest_3

    @property
    def sequence_length_nbest_4(self):
        return self._sequence_length_nbest_4



    @property
    def initial_lm_state(self):
        return self._initial_lm_state

    @property
    def softmax(self):
        return self._softmax

    @property
    def final_state(self):
        return self._final_state
