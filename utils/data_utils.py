# coding: utf-8

from data.data_utils import read_cut_file,saveDict,loadDict
import os


def createGlobalWCDict(trainFile = "data/atec/training.csv",min_count_w=2,min_count_c=2):
    """
    记录全局动态字/词-序号、序号-字/词对应关系
    """
    data=read_cut_file(trainFile,with_label=True)
    sentences=data["sent1w"]+data["sent2w"]
    sentences_c=data["sent1c"]+data["sent2c"]
    savePath=os.path.join(os.path.dirname(trainFile),os.path.basename(trainFile).split(".")[0])
    words={}
    chars={}
    for sentence in sentences:
        for word in sentence:
            try:
                words[word]+=1
            except:
                words[word]=1
    for sentence in sentences_c:
        for char in sentence:
            try:
                chars[char] += 1
            except:
                chars[char] = 1
    vocab=[w for w in words.keys() if words[w]>=min_count_w]
    vocab_c=[c for c in chars.keys() if chars[c]>=min_count_c]

    int_to_vocab = dict(enumerate(['<pad>']+['<unk>']+vocab))
    vocab_to_int = dict(zip(int_to_vocab.values(), int_to_vocab.keys()))
    print("id for <pad>: ",vocab_to_int['<pad>'])
    print("id for <unk>: ",vocab_to_int['<unk>'])
    print("total vocab size: ",len(list(int_to_vocab.keys())))
    word_dict={"i2v":int_to_vocab,"v2i":vocab_to_int}

    int_to_vocab = dict(enumerate(['<pad>']+['<unk>']+vocab_c))
    vocab_to_int = dict(zip(int_to_vocab.values(), int_to_vocab.keys()))
    print("id for <pad>: ",vocab_to_int['<pad>'])
    print("id for <unk>: ",vocab_to_int['<unk>'])
    print("total vocab size: ",len(list(int_to_vocab.keys())))
    char_dict={"i2v":int_to_vocab,"v2i":vocab_to_int}

    cw_dict={"char":char_dict,"word":word_dict}
    saveDict(cw_dict,savePath+"-"+str(min_count_w)+"-"+str(min_count_c)+".json")


def createLocalWCDict(trainFile,min_count_w=2,min_count_c=2,
                      global_dict_path="data/atec/training-2-2.json"):
    """
    根据训练数据的不同生成不同的动态序号-词/字、词/字-序号字典
    """
    global_dict = loadDict(global_dict_path)
    global_w_v2i=global_dict["word"]["v2i"]
    global_c_v2i=global_dict["char"]["v2i"]
    data=read_cut_file(trainFile,with_label=True)
    sentences=data["sent1w"]+data["sent2w"]
    sentences_c=data["sent1c"]+data["sent2c"]
    savePath = os.path.join(os.path.dirname(trainFile), os.path.basename(trainFile).split(".")[0])
    words,chars = {},{}
    for sentence in sentences:
        for word in sentence:
            try:
                words[word] += 1
            except:
                words[word] = 1
    for sentence in sentences_c:
        for char in sentence:
            try:
                chars[char] += 1
            except:
                chars[char] = 1
    print("Size for text words: ", len(words.keys()))
    print("Size for global words: ", len(global_w_v2i.keys()))
    print("Size for text chars: ", len(chars.keys()))
    print("Size for global chars: ", len(global_c_v2i.keys()))
    vocab=[w for w in words.keys() if words[w]>=min_count_w]
    vocab_c=[c for c in chars.keys() if chars[c]>=min_count_c]
    vocab = ['<pad>']  + ['<unk>']+ vocab
    vocab_c=['<pad>']  + ['<unk>']+ vocab_c
    v2i,i2v = {},{}
    for word in vocab:
        id = global_w_v2i[word]
        v2i[word] = id
        i2v[id] = word
    print("id for <pad>: ",v2i['<pad>'])
    print("id for <unk>: ",v2i['<unk>'])
    print("total vocab size: ",len(v2i.keys()))
    w_dict={"v2i":v2i,"i2v":i2v}
    v2i,i2v = {},{}
    for word in vocab_c:
        id = global_c_v2i[word]
        v2i[word] = id
        i2v[id] = word
    print("id for <pad>: ",v2i['<pad>'])
    print("id for <unk>: ",v2i['<unk>'])
    print("total vocab size: ",len(v2i.keys()))
    c_dict={"v2i":v2i,"i2v":i2v}
    d={"word":w_dict,"char":c_dict}
    saveDict(d, savePath + "-" + str(min_count_w) + "-" + str(min_count_c) + ".json")
    return d


if __name__=="__main__":
    # createGlobalWCDict("../data/atec/training.csv",2,2)
    # createLocalWCDict("../data/atec/10/train0.csv",2,2,"../data/atec/training-2-2.json")

    global_dict=loadDict("../data/atec/training-2-2.json")
    c_len={}
    for w in global_dict["word"]["v2i"].keys():
        c_len[w]=len(w)
    print(sorted(c_len.items(),key=lambda x:x[1],reverse=True)) # 9、7、6、6、6、6、5
