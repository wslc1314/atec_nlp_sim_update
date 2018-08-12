# coding: utf-8

import tensorflow as tf


def entry_stop_gradients(target, mask):
    mask_h = 1-mask
    mask = tf.cast(mask, dtype=target.dtype)
    mask_h = tf.cast(mask_h, dtype=target.dtype)
    return tf.stop_gradient(mask_h * target) + mask * target


def embedded(X,embeddings,oov_mask,trainable,scope="embedded",reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        # embedding_matrix=tf.get_variable(name="embedding_matrix", trainable=trainable,
        #                                  shape=embeddings.shape,dtype=tf.float32,
        #                                  initializer=tf.constant_initializer(value=embeddings))
        embedding_matrix=tf.get_variable(name="embedding_matrix", trainable=True,
                                         shape=embeddings.shape,dtype=tf.float32,
                                         initializer=tf.constant_initializer(value=embeddings))
        if not trainable:
            embedding_matrix=entry_stop_gradients(embedding_matrix,oov_mask)
        X_embedded = tf.nn.embedding_lookup(embedding_matrix, X)
        return X_embedded,embeddings.shape[1]


def build_loss(labels,logits,focal=True,alpha=0.75,gamma=2):
    logits=tf.reshape(logits,[-1,])
    ce_loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,dtype=tf.float32),logits=logits)
    if focal:
        probs = tf.sigmoid(logits)
        alpha_t = tf.ones_like(logits) * alpha
        alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
        probs_t = tf.where(labels > 0, probs, 1.0 - probs)
        weight_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
        loss = tf.reduce_sum(weight_matrix * ce_loss)
    else:
        loss=tf.reduce_sum(ce_loss)
    return loss


def build_summaries():
    loss, f1, acc, pre, recall = None, None, None, None, None
    summaries = tf.Summary()
    summaries.value.add(tag='Loss', simple_value=loss)
    summaries.value.add(tag='F1-score', simple_value=f1)
    summaries.value.add(tag='Accuracy', simple_value=acc)
    summaries.value.add(tag='Precision', simple_value=pre)
    summaries.value.add(tag='Recall', simple_value=recall)
    return summaries


def attention_han(inputs, attention_size,
                  initializer=tf.glorot_uniform_initializer(),
                  scope="attention_han",reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        hidden_size = inputs.shape[-1].value
        w_omega = tf.get_variable(name="weights", shape=[hidden_size, attention_size],
                                  dtype=tf.float32, initializer=initializer)
        b_omega = tf.get_variable(name="biases", shape=[attention_size,],
                                  dtype=tf.float32, initializer=tf.zeros_initializer())
        u_omega = tf.get_variable(name="context_vector", shape=[attention_size,],
                                  dtype=tf.float32, initializer=initializer)
        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        output = tf.reduce_sum(v * tf.expand_dims(alphas, -1), axis=1)
        return output,attention_size


def gru(inputs, inputs_len, state_size_list, return_mode,
        initializer=tf.glorot_uniform_initializer(),keep_prob=1.0,
        scope='gru', reuse=False):
    """:param return_mode:
    0 - 返回序列
    1 - 返回序列最后一个时间步的输出
    """
    assert return_mode in [0, 1], "Invalid return mode!"
    with tf.variable_scope(scope, reuse=reuse):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        cells_fw= []
        for i in range(len(state_size_list)):
            state_size = state_size_list[i]
            cell_fw = tf.nn.rnn_cell.GRUCell(state_size,kernel_initializer=initializer)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
        else:
            cells_fw= cells_fw[0]
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cells_fw,inputs=inputs,
                                                     sequence_length=inputs_len,
                                                     dtype=tf.float32)
        rnn_outputs_dim = state_size_list[-1]
        if return_mode == 0:
            pass
        else:
            rnn_outputs = tf.gather_nd(params=rnn_outputs,
                                       indices=tf.stack([tf.range(batch_size), inputs_len - 1], axis=1))
        return rnn_outputs, rnn_outputs_dim


def bi_gru(inputs, inputs_len, state_size_list, return_mode,
           initializer=tf.glorot_uniform_initializer(),keep_prob=1.0,
           scope='bi_gru', reuse=False):
    """:param return_mode:
    0 - 分别返回前向、后向序列
    1 - 返回拼接后序列
    2 - 返回拼接后序列最后一个时间步的输出
    """
    assert return_mode in [0,1,2],"Invalid return mode!"
    with tf.variable_scope(scope,reuse=reuse):
        batch_size,seq_len=tf.shape(inputs)[0],tf.shape(inputs)[1]
        cells_fw,cells_bw = [],[]
        for i in range(len(state_size_list)):
            state_size=state_size_list[i]
            cell_fw = tf.nn.rnn_cell.GRUCell(state_size,kernel_initializer=initializer)
            cell_bw = tf.nn.rnn_cell.GRUCell(state_size,kernel_initializer=initializer)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
        else:
            cells_fw,cells_bw= cells_fw[0],cells_bw[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw,
                                            inputs=inputs, sequence_length=inputs_len,
                                            dtype=tf.float32)
        rnn_outputs_dim = 2 * state_size_list[-1]
        if return_mode==0:
            rnn_outputs=(rnn_outputs_fw,rnn_outputs_bw)
        elif return_mode==1:
            rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
        else:
            rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
            rnn_outputs = tf.gather_nd(params=rnn_outputs,indices=tf.stack([tf.range(batch_size), inputs_len - 1], axis=1))
        return rnn_outputs, rnn_outputs_dim


def conv_with_max_pool(X_embedded,filter_size_list, filter_num,with_max_pooling,
                       activation=tf.nn.selu,initializer=tf.glorot_uniform_initializer(),
                       scope="conv_with_max_pool",reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        batch_size,seq_len=tf.shape(X_embedded)[0],tf.shape(X_embedded)[1]
        h_total = []
        for filter_size in filter_size_list:
            h = tf.layers.conv1d(inputs=X_embedded, filters=filter_num, kernel_size=filter_size,
                                 strides=1, padding='same', data_format='channels_last',
                                 activation=activation, use_bias=True,
                                 kernel_initializer=initializer)
            if with_max_pooling:
                h=tf.reduce_max(h,axis=-2)
            h_total.append(h)
        out_dim=filter_num*len(h_total)
        if len(h_total) > 1:
            h = tf.concat(h_total, axis=-1)
            if with_max_pooling:
                h = tf.reshape(h, shape=[batch_size, out_dim])
            else:
                h = tf.reshape(h, shape=[batch_size, seq_len, out_dim])
        else:
            h = h_total[0]
        return h,out_dim


def attention_to(Q, A,
                 initializer=tf.glorot_uniform_initializer(),
                 scope="attention_to", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        Q_dim = Q.shape[-1].value
        A_dim = A.shape[-1].value
        assert Q_dim == A_dim
        weights = tf.get_variable(name="weights", shape=[Q_dim, A_dim], dtype=tf.float32,
                                  initializer=initializer)
        biases = tf.get_variable(name="biases", shape=[A_dim, ], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        G = tf.nn.softmax(
            tf.matmul(
                A, tf.map_fn(lambda x: tf.matmul(x, weights) + biases, Q), transpose_b=True))
        H = tf.matmul(G, Q)
    return H,A_dim


def linear_transform(inputs,out_dim,
                     activation=None,initializer=tf.glorot_uniform_initializer(),
                     scope="linear_transform",reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        in_dim=inputs.shape[-1].value
        W = tf.get_variable(name="weights", shape=[in_dim, out_dim],
                            dtype=tf.float32, initializer=initializer)
        b = tf.get_variable(name="biases", shape=[out_dim],
                            dtype=tf.float32, initializer=tf.zeros_initializer())
        outputs = tf.tensordot(inputs, W, axes=1) + b
        if activation is None:
            return outputs,out_dim
        else:
            return activation(outputs),out_dim


def crnn(X_embedded, X_len,
         filter_size_list=(3,),filter_num=128,state_size_list=(128,),
         activation=tf.nn.selu,initializer=tf.glorot_uniform_initializer(),keep_prob=1.0,
         scope="crnn", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h,h_dim=conv_with_max_pool(X_embedded,filter_size_list,filter_num,False,activation,initializer,"conv")
        batch_size,seq_len=tf.shape(X_embedded)[0],tf.shape(X_embedded)[1]
        h = tf.reshape(h, shape=[batch_size, seq_len, h_dim])
        out,out_dim=bi_gru(h,X_len,state_size_list,2,initializer,keep_prob,"bi_gru")
        return out,out_dim


def rcnn(X_embedded,  X_len,
         state_size_list=(128,),hidden_size=256,
         activation=tf.nn.tanh, initializer=tf.glorot_uniform_initializer(),keep_prob=1.0,
         scope="rcnn", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(X_embedded)[0]
        cells_fw, cells_bw, cells_fw_init, cells_bw_init = [], [], [], []
        for i in range(len(state_size_list)):
            state_size = state_size_list[i]
            cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_units=state_size)
            cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_units=state_size)
            init_fw_ = tf.get_variable(name="cell_fw_init_state_" + str(i),
                                       dtype=tf.float32, shape=[1, state_size],
                                       trainable=True, initializer=initializer)
            init_fw = tf.tile(init_fw_, multiples=[batch_size, 1])
            init_bw_ = tf.get_variable(name="cell_bw_init_state_" + str(i),
                                       dtype=tf.float32, shape=[1, state_size],
                                       trainable=True, initializer=initializer)
            init_bw = tf.tile(init_bw_, multiples=[batch_size, 1])
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
            cells_fw_init.append(init_fw)
            cells_bw_init.append(init_bw)
        if len(cells_fw) > 1:
            cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
            cells_fw_init = tf.nn.rnn_cell.MultiRNNCell(cells_fw_init)
            cells_bw_init = tf.nn.rnn_cell.MultiRNNCell(cells_bw_init)
        else:
            cells_fw, cells_bw, cells_fw_init, cells_bw_init = cells_fw[0], cells_bw[0], cells_fw_init[0], \
                                                               cells_bw_init[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw,
                                            inputs=X_embedded, sequence_length=X_len,
                                            initial_state_fw=cells_fw_init, initial_state_bw=cells_bw_init)
        rnn_outputs_fw = tf.concat([tf.expand_dims(cells_fw_init, axis=1), rnn_outputs_fw[:, :-1, :]],
                                   axis=1)
        rnn_outputs_bw = tf.concat([rnn_outputs_bw[:, 1:, :], tf.expand_dims(cells_bw_init, axis=1)],
                                   axis=1)
        h = tf.concat([rnn_outputs_fw, X_embedded, rnn_outputs_bw], axis=-1)
        h,h_dim = linear_transform(h,hidden_size,activation,initializer,"linear_transform")
        out = tf.reduce_max(h, axis=-2)
        return out,h_dim


def han(X_embedded,  X_len,
        state_size_list=(64,),attention_dim=128,
        initializer=tf.glorot_uniform_initializer(),keep_prob=1.0,
        scope="han", reuse=False):
    """
    Only 1-level attention is used here.
    """
    with tf.variable_scope(scope, reuse=reuse):
        h,h_dim=bi_gru(X_embedded,X_len,state_size_list,1,initializer,keep_prob,"bi_gru")
        return attention_han(h, attention_dim,initializer,"attention")
