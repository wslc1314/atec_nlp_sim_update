# coding: utf-8

import numpy as np
import os
from gensim import models
from data.data_utils import loadDict


def load_global_embedding_matrix(wv_path_w,wv_path_c,global_dict_path="data/atec/training-2-2.json"):
    wv_path_list=[wv_path_w,wv_path_c]
    try:
        wv_type_w = wv_path_w.split("/")[-2]
    except AttributeError:
        wv_type_w=None
    assert wv_type_w in ["glove", "word2vec", "fasttext",None]
    try:
        wv_type_c = wv_path_c.split("/")[-2]
    except AttributeError:
        wv_type_c=None
    assert wv_type_c in ["glove", "word2vec", "fasttext",None]
    wv_type_list = [wv_type_w,wv_type_c]
    try:
        wv_name_w = wv_path_w.split("/")[-1].split(".")[0]
    except AttributeError:
        wv_name_w=None
    try:
        wv_name_c = wv_path_c.split("/")[-1].split(".")[0]
    except AttributeError:
        wv_name_c=None
    wv_name_list=[wv_name_w,wv_name_c]

    embeddings_list=[]
    for i,wv_level in enumerate(["word", "char"]):
        wv_path=wv_path_list[i]
        if isinstance(wv_path,str):
            wv_type = wv_type_list[i]
            assert wv_type in ["glove", "word2vec", "fasttext"]
            wv_name = wv_name_list[i].replace("wc", wv_level)
            embed_path=global_dict_path.replace(".json","_"+wv_type+'_'+wv_name+".npy")
            embed_path_oov=global_dict_path.replace(".json","_"+wv_type+'_'+wv_name+".oov.npy")
            if os.path.exists(embed_path):
                embeddings=np.load(embed_path)
                oov_mask=np.load(embed_path_oov)
                assert embeddings.shape[0]==oov_mask.shape[0]
            else:
                oov_mask=[]
                i2v=loadDict(global_dict_path)[wv_level]["i2v"]
                vocab_size=len(i2v)
                embedding_size=int(wv_name.split("-")[-1])
                embeddings = np.random.uniform(low=-0.1,high=0.1,size=(vocab_size, embedding_size))
                if wv_type == "word2vec":
                    model = models.Word2Vec.load(wv_path)
                else:
                    model = models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
                n_oov=0
                for i in range(vocab_size):
                    word=i2v[str(i)]
                    try:
                        embeddings[i] = model[word]
                        oov_mask.append(0)
                    except:
                        n_oov+=1
                        print("Not in wv: id: %d, vocab: %s"%(i,word))
                        oov_mask.append(1)
                print("Size for oov: %d!"%n_oov)
                np.save(embed_path,embeddings)
                oov_mask=np.asarray(oov_mask,dtype=np.int).reshape((vocab_size,1))
                np.save(embed_path_oov,oov_mask)
            embeddings_list.append((embeddings,oov_mask))
        elif isinstance(wv_path,int):
            i2v = loadDict(global_dict_path)[wv_level]["i2v"]
            vocab_size = len(i2v)
            embedding_size = int(wv_path)
            embeddings = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, embedding_size))
            oov_mask = [0]*vocab_size
            oov_mask = np.asarray(oov_mask, dtype=np.int).reshape((vocab_size, 1))
            embeddings_list.append((embeddings, oov_mask))
        else:
            print("Unsupported type for wv_path!")
    return embeddings_list[0],embeddings_list[1]


if __name__=="__main__":

    for i in [300]:
        load_global_embedding_matrix("../wv/glove/atec_word-2-"+str(i),
                                     "../wv/glove/atec_char-2-"+str(i),
                                     "../data/atec/training-2-2.json")
        # load_global_embedding_matrix("../wv/glove/wiki_word-"+str(i),
        #                              "../wv/glove/wiki_char-"+str(i),
        #                              "../data/atec/training-2-2.json")
    load_global_embedding_matrix("../wv/fasttext/wc-300.vec",
                                 "../wv/fasttext/wc-300.vec",
                                 "../data/atec/training-2-2.json")
