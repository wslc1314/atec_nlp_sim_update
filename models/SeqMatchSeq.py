# coding: utf-8

import tensorflow as tf, numpy as np
from config import Config as BaseConfig
from models.base import Model as BaseModel
from utils.wv_utils import load_global_embedding_matrix
from models.model_ops import embedded,bi_gru,attention_to,linear_transform,conv_with_max_pool,build_loss,build_summaries


class Config(BaseConfig):
    wv_config = {"path_w": "wv/glove/atec_word-2-300", "train_w": False,
                 "path_c": "wv/glove/atec_char-2-300", "train_c": False}

    initial = "uniform"
    char_bi_dim=50
    bi_dim=64
    un_dim=128

    log_dir = "logs/SeqMatchSeq"
    save_dir = "checkpoints/SeqMatchSeq"

    modeC=9


class Model(BaseModel):
    def __init__(self, config=Config):
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
            batch_size,seq_len=tf.shape(Xw)[0],tf.shape(Xw)[1]
            Xc_embedded=tf.reshape(Xc_embedded,shape=[batch_size*seq_len,-1,size_c])
            Xc_len=tf.reshape(Xc_len,shape=[batch_size*seq_len,])
            Xc_embedded,size_c=bi_gru(Xc_embedded,Xc_len,(self.config.char_bi_dim,),2,
                                      self.initializer,1.0,"bi_gru_c2w")
            Xc_embedded=tf.reshape(Xc_embedded,shape=[batch_size,seq_len,size_c])
            X_embedded=tf.concat([Xw_embedded,Xc_embedded],axis=-1)
            out_w, out_w_size = bi_gru(X_embedded, Xw_len, (self.config.bi_dim,),1,
                                       self.initializer, self.keep_prob, "bi_gru__wc")
            return out_w,out_w_size

    def _compare(self,seq1,seq2,scope="compare_layers",reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            sub=(seq1-seq2)*(seq1-seq2)
            mul=seq1*seq2
            abs=tf.abs(seq1-seq2)
            max=tf.maximum(seq1*seq1,seq2*seq2)
            sm=tf.concat([sub,mul,abs,max],axis=-1)
            sm=tf.nn.dropout(sm,self.keep_prob)
            sm,sm_size=linear_transform(sm,self.config.un_dim,tf.nn.selu,
                                        self.initializer,"linear_transform")
            return sm

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("placeholders"):
                self.X1w = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent1w_ph")
                self.X2w = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent2w_ph")
                self.X1c = tf.placeholder(dtype=tf.int32, shape=[None, None,None], name="sent1c_ph")
                self.X2c = tf.placeholder(dtype=tf.int32, shape=[None, None,None], name="sent2c_ph")
                self.X1w_mask = tf.sign(self.X1w, name="sent1w_mask")
                self.X2w_mask = tf.sign(self.X2w, name="sent2w_mask")
                self.X1c_mask = tf.sign(self.X1c, name="sent1c_mask")
                self.X2c_mask = tf.sign(self.X2c, name="sent2c_mask")
                self.X1w_l = tf.reduce_sum(self.X1w_mask, axis=-1, name="sent1w_len")
                self.X2w_l = tf.reduce_sum(self.X2w_mask, axis=-1, name="sent2w_len")
                self.X1c_l = tf.reduce_sum(self.X1c_mask, axis=-1, name="sent1c_len")
                self.X2c_l = tf.reduce_sum(self.X2c_mask, axis=-1, name="sent2c_len")
                # self.X1w_l = tf.placeholder(dtype=tf.int32, shape=[None, ], name="sent1w_len_ph")
                # self.X2w_l = tf.placeholder(dtype=tf.int32, shape=[None, ], name="sent2w_len_ph")
                # self.X1c_l = tf.placeholder(dtype=tf.int32, shape=[None, ], name="sent1c_len_ph")
                # self.X2c_l = tf.placeholder(dtype=tf.int32, shape=[None, ], name="sent2c_len_ph")
                self.y = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label_ph")
                self.keep_prob = tf.placeholder_with_default(1.0, shape=[], name="keep_prob_ph")

            # preprocess
            Q,q_size = self._preprocess(self.X1w, self.X1w_l, self.X1c, self.X1c_l, scope="preprocess_layers")
            A,a_size = self._preprocess(self.X2w, self.X2w_l, self.X2c, self.X2c_l, scope="preprocess_layers",reuse=True)

            # attention
            H,_= attention_to(Q, A,self.initializer,"attention_to_a_layers")
            K,_= attention_to(A, Q,self.initializer,"attention_to_q_layers")

            # comparison
            out_qa= self._compare(H,A,"compare_to_a_layers")
            out_aq= self._compare(K,Q,"compare_to_q_layers")
            out_qa=tf.nn.dropout(out_qa,self.keep_prob)
            out_aq=tf.nn.dropout(out_aq,self.keep_prob)

            # aggregation
            out_qa, _ = conv_with_max_pool(out_qa,(2,3,4,5),self.config.un_dim//4,True,
                                           tf.nn.selu,self.initializer,"conv_with_max_pooling")
            out_aq, _ = conv_with_max_pool(out_aq,(2,3,4,5),self.config.un_dim//4,True,
                                           tf.nn.selu,self.initializer,"conv_with_max_pooling",reuse=True)
            out=tf.concat([out_qa,out_aq],axis=-1)

            with tf.variable_scope("fc"):
                out = tf.nn.dropout(out, self.keep_prob)
                pi = 0.01
                self.logits = tf.layers.dense(out, 1,
                                              kernel_initializer=tf.glorot_uniform_initializer(),
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
                     self.keep_prob: 1 - self.config.dropout}
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
