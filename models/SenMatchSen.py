# coding: utf-8

import tensorflow as tf, numpy as np
from config import Config as BaseConfig
from models.base import Model as BaseModel
from utils.wv_utils import load_global_embedding_matrix
from models.model_ops import embedded,attention_han,bi_gru,conv_with_max_pool,build_loss,build_summaries


class Config(BaseConfig):
    wv_config = {"path_w": "wv/glove/atec_word-2-300", "train_w": False,
                 "path_c": "wv/glove/atec_char-2-300", "train_c": False}

    initial="uniform"
    un_dim=128
    bi_dim=64

    log_dir = "logs/SenMatchSen"
    save_dir = "checkpoints/SenMatchSen"

    modeC=0


class Model(BaseModel):
    def __init__(self,config=Config):
        super(Model).__init__(config)
        self.config=config
        assert self.config.initial in ["uniform","normal"]
        if self.config.initial == "uniform":
            self.initializer=tf.glorot_uniform_initializer()
        else:
            self.initializer=tf.glorot_normal_initializer()
        self.embeddings_w,self.embeddings_c = load_global_embedding_matrix(
            self.config.wv_config['path_w'],self.config.wv_config['path_c'],self.config.global_dict)
        self.build_graph()

    def _encode(self,Xw,Xw_l,Xc,Xc_l,scope="encode_layers",reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            Xw_embedded,size_w=embedded(Xw,self.embeddings_w[0],self.embeddings_w[1],self.config.wv_config["train_w"],
                                        scope="embedded_w")
            Xc_embedded,size_c=embedded(Xc,self.embeddings_c[0],self.embeddings_c[1],self.config.wv_config["train_c"],
                                        scope="embedded_c")
            batch_size=tf.shape(Xw)[0]
            # char
            v0,v0_size=attention_han(Xc_embedded,self.config.un_dim,self.initializer,"attention_han_c")
            v1,v1_size=bi_gru(Xc_embedded,Xc_l,(self.config.bi_dim,),2,self.initializer,1.0,"bi_gru_c")
            char_v=tf.reshape(tf.concat([v0,v1],axis=-1),[batch_size,v0_size+v1_size])
            # word
            v0,v0_size=attention_han(Xw_embedded,self.config.un_dim,self.initializer,"attention_han_w")
            v1,v1_size=bi_gru(Xw_embedded,Xw_l,(self.config.bi_dim,),2,self.initializer,1.0,"bi_gru_w")
            word_v=tf.reshape(tf.concat([v0,v1],axis=-1),[batch_size,v0_size+v1_size])
            # phrase
            Xp_embedded,size_p=conv_with_max_pool(Xw_embedded,(2,3,4,5),size_w//4,False,
                                                  tf.nn.selu,self.initializer,"conv_w2p")
            v0,v0_size=attention_han(Xp_embedded,self.config.un_dim,self.initializer,"attention_han_p")
            v1,v1_size=bi_gru(Xp_embedded,Xw_l,(self.config.bi_dim,),2,self.initializer,1.0,"bi_gru_p")
            phrase_v=tf.reshape(tf.concat([v0,v1],axis=-1),[batch_size,v0_size+v1_size])
            return char_v,word_v,phrase_v

    def _match(self,h1,h2,
               mah=True,euc=True,cos=True,maxi=True,
               scope="match_layers",reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            h=[]
            if mah:
                h.append(tf.abs(h1-h2))
            if euc:
                h.append((h1-h2)*(h1-h2))
            if cos:
                h.append(h1*h2)
            if maxi:
                h.append(tf.maximum(h1*h1,h2*h2))
            h=tf.concat(h,axis=-1)
            return h

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("placeholders"):
                self.X1w = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent1w_ph")
                self.X2w = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent2w_ph")
                self.X1c = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent1c_ph")
                self.X2c = tf.placeholder(dtype=tf.int32, shape=[None, None], name="sent2c_ph")
                self.X1w_mask = tf.sign(self.X1w, name="sent1w_mask")
                self.X2w_mask = tf.sign(self.X2w, name="sent2w_mask")
                self.X1c_mask = tf.sign(self.X1c, name="sent1c_mask")
                self.X2c_mask = tf.sign(self.X2c, name="sent2c_mask")
                self.X1w_l = tf.reduce_sum(self.X1w_mask, axis=-1, name="sent1w_len")
                self.X2w_l = tf.reduce_sum(self.X2w_mask, axis=-1, name="sent2w_len")
                self.X1c_l = tf.reduce_sum(self.X1c_mask, axis=-1, name="sent1c_len")
                self.X2c_l = tf.reduce_sum(self.X2c_mask, axis=-1, name="sent2c_len")
                self.y = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label_ph")
                self.keep_prob = tf.placeholder_with_default(1.0, shape=[], name="keep_prob_ph")

            # encode
            X1c,X1w,X1p = self._encode(self.X1w, self.X1w_l, self.X1c, self.X1c_l,scope="encode_layers_1")
            X2c,X2w,X2p = self._encode(self.X2w, self.X2w_l, self.X2c, self.X2c_l,scope="encode_layers_2")

            # match
            match_c = self._match(X1c, X2c, scope="match_layers_c")
            match_w = self._match(X1w, X2w, scope="match_layers_w")
            match_p = self._match(X1p, X2p, scope="match_layers_p")

            with tf.variable_scope("fc"):
                h=tf.nn.dropout(tf.concat([match_c,match_w,match_p],axis=-1),self.keep_prob)
                h1=tf.layers.dense(h,self.config.un_dim,activation=tf.nn.selu,
                                   kernel_initializer=self.initializer)
                h2=tf.layers.dense(h,self.config.un_dim,activation=tf.nn.sigmoid,
                                   kernel_initializer=self.initializer)
                h=tf.concat([h1,h2],axis=-1)
                h=tf.nn.dropout(h,keep_prob=self.keep_prob)
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
                self.var_list_trainable=[v for v in tf.trainable_variables()]

            with tf.name_scope("Loss"):
                self.loss_op= build_loss(labels=self.y, logits=self.logits,focal=self.config.focal,
                                         alpha=self.config.alpha,gamma=self.config.gamma)

            with tf.name_scope("Optimize"):
                self.adam_op = tf.train.AdamOptimizer(learning_rate=self.config.init_learning_rate).\
                    minimize(self.loss_op,var_list=self.var_list_trainable)
                self.sgd_op = tf.train.MomentumOptimizer(learning_rate=self.config.init_learning_rate,momentum=0.9).\
                    minimize(self.loss_op,var_list=self.var_list_trainable)

            with tf.name_scope("Prediction"):
                self.predicted = tf.cast(tf.greater_equal(self.pos_prob, self.config.threshold), dtype=tf.int32)

            with tf.name_scope("Summary"):
                self.summaries = build_summaries()

    def _get_train_feed_dict(self,batch):
        feed_dict = {self.X1w: np.asarray(batch["sen1w"].tolist()),
                     self.X2w: np.asarray(batch["sen2w"].tolist()),
                     # self.X1w_l: np.asarray(batch["sen1w_len"].tolist()),
                     # self.X2w_l: np.asarray(batch["sen2w_len"].tolist()),
                     self.X1c: np.asarray(batch["sen1c"].tolist()),
                     self.X2c: np.asarray(batch["sen2c"].tolist()),
                     # self.X1c_l: np.asarray(batch["sen1c_len"].tolist()),
                     # self.X2c_l: np.asarray(batch["sen2c_len"].tolist()),
                     self.y:np.asarray(batch["label"].tolist()),
                     self.keep_prob:1-self.config.dropout}
        return feed_dict
    
    def _get_valid_feed_dict(self,batch):
        feed_dict = {self.X1w: np.asarray(batch["sen1w"].tolist()),
                     self.X2w: np.asarray(batch["sen2w"].tolist()),
                     # self.X1w_l: np.asarray(batch["sen1w_len"].tolist()),
                     # self.X2w_l: np.asarray(batch["sen2w_len"].tolist()),
                     self.X1c: np.asarray(batch["sen1c"].tolist()),
                     self.X2c: np.asarray(batch["sen2c"].tolist()),
                     # self.X1c_l: np.asarray(batch["sen1c_len"].tolist()),
                     # self.X2c_l: np.asarray(batch["sen2c_len"].tolist()),
                     self.y:np.asarray(batch["label"].tolist())}
        return feed_dict

    def _get_test_feed_dict(self,batch):
        feed_dict = {self.X1w: np.asarray(batch["sen1w"].tolist()),
                     self.X2w: np.asarray(batch["sen2w"].tolist()),
                     # self.X1w_l: np.asarray(batch["sen1w_len"].tolist()),
                     # self.X2w_l: np.asarray(batch["sen2w_len"].tolist()),
                     self.X1c: np.asarray(batch["sen1c"].tolist()),
                     self.X2c: np.asarray(batch["sen2c"].tolist()),}
                     # self.X1c_l: np.asarray(batch["sen1c_len"].tolist()),
                     # self.X2c_l: np.asarray(batch["sen2c_len"].tolist())}
        return feed_dict
