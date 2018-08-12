# coding: utf-8

from gensim import models
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
from data.data_utils import read_cut_file


def visualize_wv(wv_path):
    """"
    可视化词向量并通过类比关系评价词向量。
    """
    wv_name=wv_path.split("/")[-1]
    level_type=wv_name.split("_")[-1].split("-")[0]
    assert level_type in ["char","word","wc"]
    file_type=wv_path.split("/")[-2]
    assert file_type in ["glove","word2vec","fasttext"]
    if file_type == "word2vec":
        model = models.Word2Vec.load(wv_path)
    else:
        model = models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
    print("Path for wv: %s"%wv_path)
    print("---------")
    if level_type in ["word","wc"]:
        word = model.most_similar(u"花呗",topn=5)
        print('与“花呗”相似的单词有：')
        for t in word:
            print(t[0], t[1])
        print("---------")
        print('“付款”和“付钱”的相似度是：')
        print(model.similarity(u'付款', u'付钱'))
        print("---------")
        print('“付款”和“推迟”的相似度是：')
        print(model.similarity(u'付款', u'收款'))
        print("---------")
        visualizeWords = [
            "低","最低","高","最高",
            "降低","提高","提前","延迟","推迟","打开","关闭","好", "坏", "无法","没法","成功", "失败",
            "花呗","支付宝","余额宝","天猫","淘宝","单车","共享",
            "扫码","分期","转账","信用","红包","旅行","火车","机票",
            "这", "那"]
        visualize_wv_helper(model,visualizeWords,
                            file_type+"-"+wv_name.split(".")[0].replace("_","-").replace("wc","word"),
                            wv_path.split('.')[0].replace("wc","word")+".png")
    if level_type in ["char","wc"]:
        word = model.most_similar(u"钱", topn=5)
        print('与“钱”相似的字有：')
        for t in word:
            print(t[0], t[1])
        print("---------")
        print('“款”和“钱”的相似度是：')
        print(model.similarity(u'款', u'钱'))
        print("---------")
        print('“款”和“迟”的相似度是：')
        print(model.similarity(u'款', u'迟'))
        print("---------")
        visualizeWords = [
            "低", "高", "降", "提",
            "前", "后", "迟", "开", "闭", "关","无", "没",
            "花", "借","呗","支","付","宝", "余","额", "车", "享","扫", "分","期",
            "转","账", "信","用", "红","包", "旅","行", "火","车", "机","票",
            "这", "那"]
        visualize_wv_helper(model,visualizeWords,
                            file_type+"-"+wv_name.split(".")[0].replace("_","-").replace("wc","char"),
                            wv_path.split('.')[0].replace("wc","char")+".png")


def visualize_wv_helper(model,visualizeWords,plt_title,save_path):
    visualizeVecs = []
    visualizeWords_ = []
    for i in visualizeWords:
        try:
            visualizeVecs.append(model[i])
            visualizeWords_.append(i)
        except:
            continue
    visualizeVecs = np.array(visualizeVecs).reshape((len(visualizeVecs), -1))
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(visualizeVecs)
    fig = plt.figure(figsize=(16, 9))
    plt.plot([Y[0, 0], Y[1, 0]], [Y[0, 1], Y[1, 1]], color='r')
    plt.plot([Y[2, 0], Y[3, 0]], [Y[2, 1], Y[3, 1]], color='b')
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(visualizeWords_, Y[:, 0], Y[:, 1]):
        plt.text(x, y, label, bbox=dict(facecolor='green', alpha=0.1))
    plt.xlim((np.min(Y[:, 0]) - 10, np.max(Y[:, 0]) + 10))
    plt.ylim((np.min(Y[:, 1]) - 10, np.max(Y[:, 1]) + 10))
    plt.title(plt_title)
    plt.show()
    fig.savefig(save_path)
    plt.close()


def analyze_wv_vocab_coverage(wv_path,global_train_path="../data/atec/training.csv",min_count=2):
    """分析词向量与训练数据间的词汇覆盖情况
    """
    wv_name = wv_path.split("/")[-1]
    level_type = wv_name.split("_")[-1].split("-")[0]
    assert level_type in ["char", "word","wc"]
    file_type = wv_path.split("/")[-2]
    assert file_type in ["glove", "word2vec","fasttext"]
    if file_type == "word2vec":
        model = models.Word2Vec.load(wv_path)
    else:
        model = models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
    wv_vocab = model.wv.vocab

    raw_data=read_cut_file(global_train_path,True)
    if level_type in ["word","wc"]:
        sent1,sent2=raw_data["sent1w"],raw_data["sent2w"]
        analyze_wv_vocab_coverage_helper(sent1,sent2,min_count,wv_vocab,
                                         "-".join(wv_path.split("-")[0:-1]).replace("wc","word")
                                         +"-"+str(min_count) + "_vc.png")
    if level_type in ["char","wc"]:
        sent1, sent2 = raw_data["sent1c"], raw_data["sent2c"]
        analyze_wv_vocab_coverage_helper(sent1,sent2,min_count,wv_vocab,
                                         "-".join(wv_path.split("-")[0:-1]).replace("wc","char")
                                         +"-"+str(min_count) + "_vc.png")


def analyze_wv_vocab_coverage_helper(sent1,sent2,min_count,wv_vocab,savePath):
    sentences = sent1 + sent2
    words_all = {}
    for sentence in sentences:
        for word in sentence:
            try:
                words_all[word] += 1
            except:
                words_all[word] = 1
    words={k:v for k,v in words_all.items() if v>=min_count}
    print("Size for global train vocab: ", len(words.keys()))
    print("Size for wv vocab: ", len(wv_vocab))
    vocab = set(wv_vocab) & set(words.keys())
    print("Size for their intersection: ", len(vocab))
    vocab_count=[words[w] for w in vocab]
    rest=set(words.keys())-vocab
    print("Size for the rest in global train vocab: ", len(rest))
    print("Example for the rest in global train vocab: ",list(rest)[:10])
    rest_count=[words[w] for w in rest]
    print("Example for the count of the rest in global train vocab: ", rest_count[:10])
    fig = plt.figure(figsize=(32, 9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.hist(vocab_count, bins=50)
    ax1.set_title("共有词汇频数分布")
    ax2.hist(rest_count, bins=50)
    ax2.set_title("独有词汇频数分布")
    plt.show()
    fig.savefig(savePath)
    plt.close()
