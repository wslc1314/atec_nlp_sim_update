# coding: utf-8

import tensorflow as tf, numpy as np
from config import Config as BaseConfig
from models.base import Model as BaseModel
from utils.wv_utils import load_global_embedding_matrix
from models.model_ops import embedded,bi_gru,build_loss,build_summaries
EPSILON = 1e-6


class Config(BaseConfig):
    # model
    wv_config = {"path_w": "wv/glove/atec_word-2-300", "train_w": False,
                 "path_c": "wv/glove/atec_char-2-300", "train_c": False}

    initial = "uniform"
    char_dim = 50
    bi_dim = 64
    un_dim = 128
    mp_dim=16

    log_dir = "logs/SeqMatchSeq_BiMPM"
    save_dir = "checkpoints/SeqMatchSeq_BiMPM"

    modeC = 9


class Model(BaseModel):
    def __init__(self,config=Config):
        super(Model).__init__(config)
        self.config = config
        assert self.config.initial in ["uniform", "normal"]
        if self.config.initial == "uniform":
            self.initializer = tf.glorot_uniform_initializer()
        else:
            self.initializer = tf.glorot_normal_initializer()
        self.embeddings_w, self.embeddings_c = load_global_embedding_matrix(
            self.config.wv_config['path_w'], self.config.wv_config['path_c'], self.config.global_dict)
        self.build_graph()

    def _preprocess(self,Xw,Xw_len,Xc,Xc_len,scope="preprocess_layers",reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            Xw_embedded, size_w = embedded(Xw, self.embeddings_w[0], self.embeddings_w[1],
                                           self.config.wv_config["train_w"],
                                           scope="embedded_w")
            Xc_embedded, size_c = embedded(Xc, self.embeddings_c[0], self.embeddings_c[1],
                                           self.config.wv_config["train_c"],
                                           scope="embedded_c")
            batch_size, seq_len = tf.shape(Xw)[0], tf.shape(Xw)[1]
            Xc_embedded = tf.reshape(Xc_embedded, shape=[batch_size * seq_len, -1, size_c])
            Xc_len = tf.reshape(Xc_len, shape=[batch_size * seq_len, ])
            Xc_embedded, size_c = bi_gru(Xc_embedded, Xc_len, (self.config.char_dim,), 2,
                                         self.initializer, 1.0,"bi_gru_c2w")
            Xc_embedded = tf.reshape(Xc_embedded, shape=[batch_size, seq_len, size_c])
            X_embedded = tf.concat([Xw_embedded, Xc_embedded], axis=-1)
            (out_f,out_b), out_size = bi_gru(tf.nn.dropout(X_embedded,self.keep_prob),
                                             Xw_len, (self.config.bi_dim,),0,
                                             self.initializer, self.keep_prob , "bi_gru__wc")
            return out_f,out_b

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("placeholders"):
                self.X1w = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent1w_ph")
                self.X2w = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent2w_ph")
                self.X1c = tf.placeholder(dtype=tf.int32, shape=[None, None,None], name="sent1c_ph")
                self.X2c = tf.placeholder(dtype=tf.int32, shape=[None, None,None], name="sent2c_ph")
                self.y = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label_ph")
                self.keep_prob=tf.placeholder_with_default(1.0,shape=[],name="keep_prob_ph")
                self.X1w_mask = tf.sign(self.X1w, name="sent1w_mask")
                self.X2w_mask = tf.sign(self.X2w, name="sent2w_mask")
                self.X1c_mask = tf.sign(self.X1c, name="sent1c_mask")
                self.X2c_mask = tf.sign(self.X2c, name="sent2c_mask")
                self.X1w_l = tf.reduce_sum(self.X1w_mask, axis=-1, name="sent1w_len")
                self.X2w_l = tf.reduce_sum(self.X2w_mask, axis=-1, name="sent2w_len")
                self.X1c_l = tf.reduce_sum(self.X1c_mask, axis=-1, name="sent1c_len")
                self.X2c_l = tf.reduce_sum(self.X2c_mask, axis=-1, name="sent2c_len")

            X1_f,X1_b=self._preprocess(self.X1w,self.X1w_l,self.X1c,self.X1c_l,scope="preprocess_layers")
            X2_f,X2_b=self._preprocess(self.X2w,self.X2w_l,self.X2c,self.X2c_l,scope="preprocess_layers",reuse=True)

            with tf.variable_scope("match_layers"):
                # Shapes: (batch_size, num_sentence_words, 8*multi-perspective_dims)
                match_1_to_2_out, match_2_to_1_out = bilateral_matching(
                    X1_f, X1_b,X2_f, X2_b,self.X1w_mask,self.X2w_mask,
                    self.keep_prob,self.config.mp_dim)
                
            # Aggregate the representations from the matching functions.
            with tf.variable_scope("aggregate_layers"):
                seq_1_fb,_=bi_gru(match_1_to_2_out,self.X1w_l,(self.config.bi_dim,),2,
                                  self.initializer,1.0,"bi_gru")
                seq_2_fb,_= bi_gru(match_2_to_1_out, self.X2w_l, (self.config.bi_dim,),2,
                                   self.initializer,1.0,"bi_gru",reuse=True)
                combined_aggregated_representation = tf.concat([seq_1_fb,seq_2_fb], -1)

            with tf.variable_scope("fc_layers"):
                h = tf.nn.dropout(combined_aggregated_representation,keep_prob=self.keep_prob)
                h = tf.layers.dense(h, self.config.un_dim, activation=tf.nn.selu,
                                    kernel_initializer=self.initializer)
                h = tf.nn.dropout(h, keep_prob=self.keep_prob)
                pi = 0.01
                self.logits = tf.layers.dense(h, 1,
                                              kernel_initializer=self.initializer,
                                              bias_initializer=tf.constant_initializer(-np.log((1 - pi) / pi)))
            self.pos_prob = tf.nn.sigmoid(self.logits)
            self.var_list = [v for v in tf.global_variables()]
            if self.config.fine_tune:
                self.var_list_trainable = [v for v in tf.trainable_variables()
                                           if "embedded" in v.name or "fc" in v.name]
            else:
                self.var_list_trainable = [v for v in tf.trainable_variables()]

            with tf.name_scope("Loss"):
                self.loss_op = build_loss(labels=self.y, logits=self.logits, focal=self.config.focal,
                                          alpha=self.config.alpha, gamma=self.config.gamma)

            with tf.name_scope("Optimize"):
                self.adam_op = tf.train.AdamOptimizer(learning_rate=self.config.init_learning_rate). \
                    minimize(self.loss_op, var_list=self.var_list_trainable)
                self.sgd_op = tf.train.MomentumOptimizer(learning_rate=self.config.init_learning_rate, momentum=0.9). \
                    minimize(self.loss_op, var_list=self.var_list_trainable)

            with tf.name_scope("Prediction"):
                self.predicted = tf.cast(tf.greater_equal(self.pos_prob, self.config.threshold), dtype=tf.int32)

            with tf.name_scope("Summary"):
                self.summaries = build_summaries()

    def _get_train_feed_dict(self, batch):
        feed_dict = {self.X1w: np.asarray(batch["sen1w"].tolist()),
                     self.X2w: np.asarray(batch["sen2w"].tolist()),
                     # self.X1w_l: np.asarray(batch["sen1w_len"].tolist()),
                     # self.X2w_l: np.asarray(batch["sen2w_len"].tolist()),
                     self.X1c: np.asarray(batch["sen1c"].tolist()),
                     self.X2c: np.asarray(batch["sen2c"].tolist()),
                     # self.X1c_l: np.asarray(batch["sen1c_len"].tolist()),
                     # self.X2c_l: np.asarray(batch["sen2c_len"].tolist()),
                     self.y: np.asarray(batch["label"].tolist()),
                     self.keep_prob:1-self.config.dropout}
        return feed_dict

    def _get_valid_feed_dict(self, batch):
        feed_dict = {self.X1w: np.asarray(batch["sen1w"].tolist()),
                     self.X2w: np.asarray(batch["sen2w"].tolist()),
                     # self.X1w_l: np.asarray(batch["sen1w_len"].tolist()),
                     # self.X2w_l: np.asarray(batch["sen2w_len"].tolist()),
                     self.X1c: np.asarray(batch["sen1c"].tolist()),
                     self.X2c: np.asarray(batch["sen2c"].tolist()),
                     # self.X1c_l: np.asarray(batch["sen1c_len"].tolist()),
                     # self.X2c_l: np.asarray(batch["sen2c_len"].tolist()),
                     self.y: np.asarray(batch["label"].tolist())}
        return feed_dict

    def _get_test_feed_dict(self, batch):
        feed_dict = {self.X1w: np.asarray(batch["sen1w"].tolist()),
                     self.X2w: np.asarray(batch["sen2w"].tolist()),
                     # self.X1w_l: np.asarray(batch["sen1w_len"].tolist()),
                     # self.X2w_l: np.asarray(batch["sen2w_len"].tolist()),
                     self.X1c: np.asarray(batch["sen1c"].tolist()),
                     self.X2c: np.asarray(batch["sen2c"].tolist())}
                     # self.X1c_l: np.asarray(batch["sen1c_len"].tolist()),
                     # self.X2c_l: np.asarray(batch["sen2c_len"].tolist())}
        return feed_dict


def bilateral_matching(sentence_one_fw_representation, sentence_one_bw_representation,
                       sentence_two_fw_representation, sentence_two_bw_representation,
                       sentence_one_mask, sentence_two_mask,
                       keep_prob, multi_perspective_dims,
                       with_full_match=True, with_pool_match=True,
                       with_attentive_match=True, with_max_attentive_match=True):
    """Given the representations of a sentence from a BiRNN, apply four bilateral
    matching functions between sentence_one and sentence_two in both directions
    (sentence_one to sentence_two, and sentence_two to sentence_one).
    """
    # Match each word of sentence one to the entirety of sentence two.
    with tf.variable_scope("match_one_to_two"):
        match_one_to_two_output = match_sequences(
            sentence_one_fw_representation,
            sentence_one_bw_representation,
            sentence_two_fw_representation,
            sentence_two_bw_representation,
            sentence_one_mask,
            sentence_two_mask,
            multi_perspective_dims=multi_perspective_dims,
            with_full_match=with_full_match,
            with_pool_match=with_pool_match,
            with_attentive_match=with_attentive_match,
            with_max_attentive_match=with_max_attentive_match)

    # Match each word of sentence two to the entirety of sentence one.
    with tf.variable_scope("match_two_to_one"):
        match_two_to_one_output = match_sequences(
            sentence_two_fw_representation,
            sentence_two_bw_representation,
            sentence_one_fw_representation,
            sentence_one_bw_representation,
            sentence_two_mask,
            sentence_one_mask,
            multi_perspective_dims=multi_perspective_dims,
            with_full_match=with_full_match,
            with_pool_match=with_pool_match,
            with_attentive_match=with_attentive_match,
            with_max_attentive_match=with_max_attentive_match)

    # Shapes: (batch_size, num_sentence_words, 13*multi_perspective_dims)
    match_one_to_two_representations = tf.concat(
        match_one_to_two_output, 2)
    match_two_to_one_representations = tf.concat(
        match_two_to_one_output, 2)

    # Apply dropout to the matched representations.
    # Shapes: (batch_size, num_sentence_words, 13*multi_perspective_dims)
    match_one_to_two_representations = tf.nn.dropout(
        match_one_to_two_representations,keep_prob=keep_prob,
        name="match_one_to_two_dropout")
    match_two_to_one_representations = tf.nn.dropout(
        match_two_to_one_representations,keep_prob=keep_prob,
        name="match_two_to_one_dropout")
    # Shapes: (batch_size, num_sentence_words, 8*multi_perspective_dims)
    return match_one_to_two_representations, match_two_to_one_representations


def match_sequences(sentence_a_fw, sentence_a_bw, sentence_b_fw, sentence_b_bw,
                    sentence_a_mask, sentence_b_mask, multi_perspective_dims,
                    with_full_match, with_pool_match, with_attentive_match,with_max_attentive_match):
    """Given the representations of a sentence from a BiRNN, apply four bilateral
    matching functions from sentence_a to sentence_b (so each time step of sentence_a is
    matched with the the entirety of sentence_b).
    """
    matched_representations = []
    sentence_b_len = tf.reduce_sum(sentence_b_mask, 1)
    sentence_encoding_dim = sentence_a_fw.shape[-1].value
    # Calculate the cosine similarity matrices for
    # fw and bw representations, used in the attention-based matching functions.
    # Shapes: (batch_size, num_sentence_words, num_sentence_words)
    fw_similarity_matrix = calculate_cosine_similarity_matrix(sentence_b_fw,sentence_a_fw)
    fw_similarity_matrix = mask_similarity_matrix(fw_similarity_matrix,sentence_b_mask,sentence_a_mask)
    bw_similarity_matrix = calculate_cosine_similarity_matrix(sentence_b_bw,sentence_a_bw)
    bw_similarity_matrix = mask_similarity_matrix(bw_similarity_matrix,sentence_b_mask,sentence_a_mask)
    # Apply the multi_perspective matching functions.
    if multi_perspective_dims > 0:
        # Apply forward and backward full matching
        if with_full_match:
            # Forward full matching: each time step of sentence_a_fw vs last output of sentence_b_fw.
            with tf.variable_scope("forward_full_matching"):
                # Shape: (batch_size, rnn_hidden_size)
                last_output_sentence_b_fw = last_relevant_output(sentence_b_fw, sentence_b_len,fw=True)
                # The weights for the matching function.
                fw_full_match_params = tf.get_variable(
                    "forward_full_matching_params",shape=[multi_perspective_dims, sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dims)
                fw_full_match_output = full_matching(sentence_a_fw,last_output_sentence_b_fw,fw_full_match_params)
            matched_representations.append(fw_full_match_output)
            # Backward full matching: each time step of sentence_a_bw vs last output of sentence_b_bw.
            with tf.variable_scope("backward_full_matching"):
                # Shape: (batch_size, rnn_hidden_size)
                last_output_sentence_b_bw = last_relevant_output(sentence_b_bw, sentence_b_len,fw=False)
                # The weights for the matching function.
                bw_full_match_params = tf.get_variable(
                    "backward_full_matching_params",shape=[multi_perspective_dims, sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dims)
                bw_full_match_output = full_matching(sentence_a_bw,last_output_sentence_b_bw,bw_full_match_params)
            matched_representations.append(bw_full_match_output)
        # Apply forward and backward pool matching.
        if with_pool_match:
            # Forward Pooling-Matching: each time step of sentence_a_fw vs
            # each element of sentence_b_fw, then taking the element-wise max.
            with tf.variable_scope("forward_pooling_matching"):
                # The weights for the matching function.
                fw_pooling_params = tf.get_variable(
                    "forward_pooling_matching_params",
                    shape=[multi_perspective_dims, sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dims)
                fw_pooling_match_output = max_pooling_matching(sentence_a_fw,sentence_b_fw,fw_pooling_params)
                matched_representations.append(fw_pooling_match_output)
            # Backward Pooling-Matching: each time step of sentence_a_bw vs.
            # each element of sentence_b_bw, then taking the element-wise mean.
            with tf.variable_scope("backward_pooling_matching"):
                # The weights for the matching function
                bw_pooling_params = tf.get_variable(
                    "backward_pooling_matching_params",
                    shape=[multi_perspective_dims, sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dims)
                bw_pooling_match_output = max_pooling_matching(sentence_a_bw,sentence_b_bw,bw_pooling_params)
                matched_representations.append(bw_pooling_match_output)
        # Apply forward and backward attentive matching.
        # Using the cosine distances between the sentence representations, we use a weighted
        # sum across the entire sentence to generate an attention vector.
        if with_attentive_match:
            # Forward Attentive Matching
            with tf.variable_scope("forward_attentive_matching"):
                # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
                sentence_b_fw_att = weight_sentence_by_similarity(sentence_b_fw,fw_similarity_matrix)
                # The weights for the matching function.
                fw_attentive_params = tf.get_variable(
                    "forward_attentive_matching_params",
                    shape=[multi_perspective_dims, sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dim)
                fw_attentive_matching_output = attentive_matching(sentence_a_fw,sentence_b_fw_att,fw_attentive_params)
                matched_representations.append(fw_attentive_matching_output)
            # Backward Attentive Matching
            with tf.variable_scope("backward_attentive_matching"):
                sentence_b_bw_att = weight_sentence_by_similarity(sentence_b_bw,bw_similarity_matrix)
                bw_attentive_params = tf.get_variable(
                    "backward_attentive_matching_params",
                    shape=[multi_perspective_dims, sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dim)
                bw_attentive_matching_output = attentive_matching(sentence_a_bw,sentence_b_bw_att,bw_attentive_params)
                matched_representations.append(bw_attentive_matching_output)
        # Apply forward and backward max attentive matching.
        # Use the time step of the sentence_b with the highest cosine similarity
        # to cosine b as an attention vector.
        if with_max_attentive_match:
            # Forward max attentive-matching
            with tf.variable_scope("forward_attentive_matching"):
                # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
                sentence_b_fw_max_att = max_sentence_similarity(sentence_b_fw,fw_similarity_matrix)
                # The weights for the matching function.
                fw_max_attentive_params = tf.get_variable(
                    "fw_max_attentive_params",
                    shape=[multi_perspective_dims,sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dim)
                fw_max_attentive_matching_output = attentive_matching(sentence_a_fw,sentence_b_fw_max_att,
                                                                      fw_max_attentive_params)
                matched_representations.append(fw_max_attentive_matching_output)
            # Backward max attentive-matching
            with tf.variable_scope("backward_attentive_matching"):
                # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
                sentence_b_bw_max_att = max_sentence_similarity(sentence_b_bw,bw_similarity_matrix)
                # The weights for the matching function.
                bw_max_attentive_params = tf.get_variable(
                    "bw_max_attentive_params",
                    shape=[multi_perspective_dims,sentence_encoding_dim],dtype="float")
                # Shape: (batch_size, num_sentence_words, multi_perspective_dim)
                bw_max_attentive_matching_output = attentive_matching(sentence_a_bw,sentence_b_bw_max_att,
                                                                      bw_max_attentive_params)
                matched_representations.append(bw_max_attentive_matching_output)
    return matched_representations


def max_sentence_similarity(sentence_input, similarity_matrix):
    """Parameters
    ----------
    sentence_input: Tensor
        Tensor of shape (batch_size, num_sentence_words1, rnn_hidden_dim).
    similarity_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words2, num_sentence_words1).
    """
    def single_instance(inputs):
        single_sentence = inputs[0]
        argmax_index = inputs[1]
        # Shape: (num_sentence_words2, rnn_hidden_dim)
        return tf.gather(single_sentence, argmax_index)
    question_index = tf.argmax(similarity_matrix, 2)
    elems = (sentence_input, question_index)
    # Shape: (batch_size, num_sentence_words2, rnn_hidden_dim)
    return tf.map_fn(single_instance, elems, dtype="float")


def attentive_matching(input_sentence, att_matrix, weights):
    """Parameters
    ----------
    input_sentence: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)
    att_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)
    """
    def single_instance(inputs):
        # Shapes: (num_sentence_words, rnn_hidden_dim)
        sentence_a_single = inputs[0]
        sentence_b_single_att = inputs[1]
        # Shapes: (num_sentence_words, multi_perspective_dims, rnn_hidden_dim)
        expanded_sentence_a_single = multi_perspective_expand_for_2d(sentence_a_single, weights)
        expanded_sentence_b_single_att = multi_perspective_expand_for_2d(sentence_b_single_att, weights)
        # Shape: (num_sentence_words, multi_perspective_dims)
        return cosine_distance(expanded_sentence_a_single,expanded_sentence_b_single_att)
    elems = (input_sentence, att_matrix)
    # Shape: (batch_size, num_sentence_words, multi_perspective_dims)
    return tf.map_fn(single_instance, elems, dtype="float")


def weight_sentence_by_similarity(input_sentence, cosine_matrix,normalize=False):
    """Parameters
    ----------
    input_sentence: Tensor
        Tensor of shape (batch_size, num_sentence_words1, rnn_hidden_dim)
    cosine_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words2, num_sentence_words1)
    """
    if normalize:
        cosine_matrix = tf.nn.softmax(cosine_matrix)
    # Shape: (batch_size, num_sentence_words2, num_sentence_words1, 1)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1)
    # Shape: (batch_size, 1, num_sentence_words1, rnn_hidden_dim)
    weighted_question_words = tf.expand_dims(input_sentence, axis=1)
    # Shape: (batch_size, num_sentence_words2, rnn_hidden_dim)
    weighted_question_words = tf.reduce_sum(tf.multiply(weighted_question_words, expanded_cosine_matrix), axis=2)
    if not normalize:
        weighted_question_words = tf.div(
            weighted_question_words,
            tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),EPSILON),axis=-1))
    # Shape: (batch_size, num_sentence_words2, rnn_hidden_dim)
    return weighted_question_words


def max_pooling_matching(sentence_a_representation,sentence_b_representation,weights):
    """Parameters
    ----------
    sentence_a_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words1, rnn_hidden_dim)
    sentence_b_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words2, rnn_hidden_dim)
    weights: Tensor
        Tensor of shape (multi_perspective_dims, rnn_hidden_dim)
    """
    def single_instance(inputs):
        # Shape: (num_sentence_words1, rnn_hidden_dim)
        sentence_a_representation_single = inputs[0]
        # Shape: (num_sentence_words2, rnn_hidden_dim)
        sentence_b_representation_single = inputs[1]
        # Shape: (num_sentence_words1, multi_perspective_dims, rnn_hidden_dim)
        sentence_a_expanded = multi_perspective_expand_for_2d(sentence_a_representation_single, weights)
        # Shape: (num_sentence_words2, multi_perspective_dims, rnn_hidden_dim)
        sentence_b_expanded = multi_perspective_expand_for_2d(sentence_b_representation_single, weights)
        # Shape: (num_sentence_words1, 1, multi_perspective_dims,rnn_hidden_dim)
        sentence_a_expanded = tf.expand_dims(sentence_a_expanded, 1)
        # Shape: (1, num_sentence_words2, multi_perspective_dims,rnn_hidden_dim)
        sentence_b_expanded = tf.expand_dims(sentence_b_expanded, 0)
        # Shape: (num_sentence_words1, num_sentence_words2, multi_perspective_dims)
        return cosine_distance(sentence_a_expanded,sentence_b_expanded)
    elems = (sentence_a_representation, sentence_b_representation)
    # Shape: (batch_size, num_sentence_words1, num_sentence_words2,multi_perspective_dims)
    matching_matrix = tf.map_fn(single_instance, elems, dtype="float")
    # Take the max pool of the matching matrix.
    # Shape: (batch_size, num_sentence_words1, multi_perspective_dims)
    return tf.reduce_max(matching_matrix, axis=2)


def full_matching(sentence_a_representation,sentence_b_last_output,weights):
    """Match each time step of sentence_a with the last output of sentence_b
    by passing them both through the multi_perspective matching function.
    """
    def single_instance(inputs):
        # Shape: (num_sentence_words, rnn_hidden_dim)
        sentence_a_representation_single = inputs[0]
        # Shape: (rnn_hidden_dim)
        sentence_b_last_output_single = inputs[1]
        # Shape: (num_sentence_words, multi_perspective_dims, rnn_hidden_dim)
        sentence_a_single_expanded = multi_perspective_expand_for_2d(sentence_a_representation_single,weights)
        # Shape: (multi_perspective_dims, rnn_hidden_dim)
        sentence_b_last_output_expanded = multi_perspective_expand_for_1d(sentence_b_last_output_single,weights)
        # Shape: (1, multi_perspective_dims, rnn_hidden_dim)
        sentence_b_last_output_expanded = tf.expand_dims(sentence_b_last_output_expanded, 0)
        # Shape: (num_sentence_words, multi_perspective_dims)
        return cosine_distance(sentence_a_single_expanded,sentence_b_last_output_expanded)
    elems = (sentence_a_representation, sentence_b_last_output)
    # Shape: (batch_size, num_sentence_words, multi_perspective_dims)
    return tf.map_fn(single_instance, elems, dtype="float")


def last_relevant_output(output, sequence_length,fw=True):
    """Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).
    """
    with tf.variable_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        if fw:
            relevant = tf.gather_nd(params=output,
                                    indices=tf.stack([tf.range(batch_size), sequence_length - 1], axis=1))
        else:
            # bw
            relevant = tf.gather_nd(params=output,
                                    indices=tf.stack([tf.range(batch_size), tf.zeros_like(sequence_length)], axis=1))
        return relevant


def multi_perspective_expand_for_3d(in_tensor, weights):
    # Shape: (batch_size, num_sentence_words, 1, rnn_hidden_dim)
    in_tensor_expanded = tf.expand_dims(in_tensor, axis=2)
    # Shape: (1, 1, multi_perspective_dims, rnn_hidden_dim)
    weights_expanded = tf.expand_dims(tf.expand_dims(weights, axis=0),axis=0)
    # Shape: (batch_size, num_sentence_words, multi_perspective_dims, rnn_hidden_dim)
    return tf.multiply(in_tensor_expanded, weights_expanded)


def multi_perspective_expand_for_2d(in_tensor, weights):
    """Given a 2d input tensor and weights of the appropriate shape,
    weight the input tensor by the weights by multiplying them
    together.
    """
    # Shape: (num_sentence_words, 1, rnn_hidden_dim)
    in_tensor_expanded = tf.expand_dims(in_tensor, axis=1)
    # Shape: (1, multi_perspective_dims, rnn_hidden_dim)
    weights_expanded = tf.expand_dims(weights, axis=0)
    # Shape: (num_sentence_words, multi_perspective_dims, rnn_hidden_dim)
    return tf.multiply(in_tensor_expanded, weights_expanded)


def multi_perspective_expand_for_1d(in_tensor, weights):
    """Given a 1d input tensor and weights of the appropriate shape,
    weight the input tensor by the weights by multiplying them
    together.
    """
    # Shape: (1, rnn_hidden_dim)
    in_tensor_expanded = tf.expand_dims(in_tensor, axis=0)
    # Shape: (multi_perspective_dims, rnn_hidden_dim)
    return tf.multiply(in_tensor_expanded, weights)


def mask_similarity_matrix(similarity_matrix, mask_a, mask_b):
    """Given the mask of the two sentences, apply the mask to the similarity
    matrix.
    Parameters
    ----------
    similarity_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words2, num_sentence_words1).
    mask_a: Tensor
        Tensor of shape (batch_size, num_sentence_words1). This mask should
        correspond to the first vector (v1) used to calculate the similarity
        matrix.
    mask_b: Tensor
        Tensor of shape (batch_size, num_sentence_words2). This mask should
        correspond to the second vector (v2) used to calculate the similarity
        matrix.
    """
    similarity_matrix = tf.multiply(similarity_matrix,
                                    tf.expand_dims(tf.cast(mask_a, "float"), 1))
    similarity_matrix = tf.multiply(similarity_matrix,
                                    tf.expand_dims(tf.cast(mask_b, "float"), 2))
    return similarity_matrix


def calculate_cosine_similarity_matrix(v1, v2):
    """Calculate the cosine similarity matrix between two
    sentences.
    """
    # Shape: (batch_size, 1, num_sentence_words1, rnn_hidden_size)
    expanded_v1 = tf.expand_dims(v1, 1)
    # Shape: (batch_size, num_sentence_words2, 1, rnn_hidden_size)
    expanded_v2 = tf.expand_dims(v2, 2)
    # Shape: (batch_size, num_sentence_words2, num_sentence_words1)
    cosine_relevancy_matrix = cosine_distance(expanded_v1,expanded_v2)
    return cosine_relevancy_matrix


def cosine_distance(v1, v2):
    """Calculate the cosine distance between the representations of the
    words of the two sentences.

    Parameters
    ----------
    v1: Tensor
        Tensor of shape (batch_size, 1, num_sentence_words1, context_rnn_hidden_size)
        representing the first sentence to take the cosine similarity with.

    v2: Tensor
        Tensor of shape (batch_size, num_sentence_words2, 1, context_rnn_hidden_size)
        representing the second sentence to take the cosine similarity with.
    """
    # The product of the two vectors is shape
    # (batch_size, num_sentence_words2, num_sentence_words1, rnn_hidden_size)
    # Taking the sum over the last axis results in shape:
    # (batch_size, num_sentence_words2, num_sentence_words1)
    cosine_numerator = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)
    # Shape: (batch_size, 1, num_sentence_words1)
    v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1), axis=-1),EPSILON))
    # Shape: (batch_size, num_sentence_words2, 1)
    v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2), axis=-1),EPSILON))
    # Shape: (batch_size, num_sentence_words2, num_sentence_words1)
    return cosine_numerator / v1_norm / v2_norm
