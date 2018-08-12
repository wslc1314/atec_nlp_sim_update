# coding: utf-8

import codecs,pandas as pd
from data.data_utils import str_to_list,read_cut_file,saveDict
import matplotlib.pyplot as plt
import re
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False


def combine_train_files(dest_path="atec/atec_nlp_sim_train_all.csv",
                        src_path_list=("atec/atec_nlp_sim_train.csv","atec/atec_nlp_sim_train_add.csv")):
    final_file=codecs.open(dest_path,'w',encoding="utf-8")
    line_id=0
    for id,src_path in enumerate(src_path_list):
        if id==0:
            with codecs.open(src_path,'r',encoding="utf-8") as f:
                line=f.readline()
                while line:
                    final_file.write(line)
                    line_id+=1
                    line = f.readline()
        else:
            print(line_id) # 39346
            with codecs.open(src_path,'r',encoding="utf-8") as f:
                line=f.readline()
                now_id=None
                while line:
                    line=str_to_list(line,"\t","utf-8","utf-8")
                    now_id=str(int(line[0])+line_id)
                    final_file.write("\t".join([now_id]+line[1:])+"\n")
                    line = f.readline()
                line_id=int(now_id)
    final_file.close()
    with codecs.open(dest_path,'r',"utf-8") as f:
        data=f.readlines()
        print(len(data)) # 102477
        print(data[0])
        print(data[0].strip().split("\t"))
        print(str_to_list(data[0]))


def get_t2s_dict(t_file="atec/atec_nlp_sim_train_all.csv",
                 s_file="atec/atec_nlp_sim_train_all.simp.csv",
                 save_path="atec/t2s_dict.json"):
    t2s={}
    with codecs.open(t_file,'r',"utf-8") as f:
        raw_data_t=f.readlines()
    with codecs.open(s_file,'r',"utf-8") as f:
        raw_data_s=f.readlines()
    assert len(raw_data_t)==len(raw_data_s)
    for l1,l2 in zip(raw_data_t,raw_data_s):
        l1=''.join(l1.split('\t')[1:3])
        l2=''.join(l2.split('\t')[1:3])
        assert len(l1)==len(l2)
        for t,s in zip(l1,l2):
            if t!=s:
                if t not in t2s.keys():
                    print("%s -> %s" % (t, s))
                    t2s[t]=s
                else:
                    assert s==t2s[t]
    saveDict(t2s,save_path)


def show_not_ch(file_path="atec/atec_nlp_sim_train_all.simp.csv",
                only_en=False,between_word=""):
    not_ch = {}
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        print('open a file.')
        lineNum = 1
        line = f.readline()
        while line:
            print('---processing ', lineNum, ' article---')
            line = str_to_list(line)
            if between_word=="":
                line1,line2=line[1],line[2]
            else:
                line1,line2="".join(str_to_list(line[1],between_word)),"".join(str_to_list(line[2],between_word))
            if only_en:
                line1 = re.findall('[a-zA-Z]+', line1)
                line2 = re.findall('[a-zA-Z]+', line2)
            else:
                line1 = re.findall('[^\u4e00-\u9fa5]+', line1)
                line2 = re.findall('[^\u4e00-\u9fa5]+', line2)
            for w in line1+line2:
                try:
                    not_ch[w]+=1
                except KeyError:
                    not_ch[w]=1
            lineNum+=1
            line = f.readline()
    print(sorted(not_ch.items(),key=lambda x:x[1],reverse=True))


def label_distribution(trainFile="atec/training.csv"):
    """
    分析训练数据中标签分布情况
    """
    labels=read_cut_file(file_path=trainFile,with_label=True)["label"]
    neg_count=labels.count(0)
    pos_count=labels.count(1)
    assert neg_count+pos_count==len(labels)
    counts=[neg_count,pos_count]
    labels=["不同义","同义"]
    fig=plt.figure(figsize=(9,9))
    # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.pie(counts, labels=labels, autopct='%1.2f%%')
    plt.title("标签分布", bbox={'facecolor': '0.6', 'pad': 5})
    plt.show()
    savePath=trainFile.split(".")[0]+"_ld.png"
    fig.savefig(savePath)
    plt.close()


def sentence_length_distribution(trainFile="atec/training.csv"):
    """
    分析训练数据中句子长度分布
    """
    raw_data = read_cut_file(file_path=trainFile, with_label=True)
    df=pd.DataFrame(raw_data)
    level=["w","c"]
    for l in level:
        s1="sent1"+l+"_len"
        print(df.loc[df[s1].argmax()])
        print(df[s1].describe())
        s2="sent2"+l+"_len"
        print(df.loc[df[s2].argmax()])
        print(df[s2].describe())
        df_=pd.DataFrame({s1:df[s1],s2:df[s2]})
        fig=plt.figure(figsize=(32,18))
        df_.boxplot()
        plt.legend()
        plt.show()
        fig.savefig(trainFile.replace(".csv","_sl_"+l+".png"))


def get_corpus(file_path="atec/training.csv",corpus_path="atec/atec"):
    """
    根据给定数据生成字级和词级语料库
    """
    target_char = codecs.open(corpus_path+"_char", 'w', encoding='utf-8')
    target_word = codecs.open(corpus_path + "_word", 'w', encoding='utf-8')
    raw_data=read_cut_file(file_path,True)
    w1,w2,c1,c2=raw_data["sent1w"],raw_data["sent2w"],raw_data["sent1c"],raw_data["sent2c"]
    target_char.writelines([" ".join(c)+"\n" for c in c1])
    target_char.writelines([" ".join(c)+"\n" for c in c2])
    target_word.writelines([" ".join(w)+"\n" for w in w1])
    target_word.writelines([" ".join(w)+"\n" for w in w2])
    print('well done.')
    target_char.close()
    target_word.close()


if __name__=="__main__":

    # combine_train_files()

    # get_t2s_dict()

    # show_not_ch()

    # show_not_ch(only_en=True)

    # from data.data_utils import participle
    # participle("atec/atec_nlp_sim_train_all.simp.csv","atec/training.csv",True,None)

    # show_not_ch(file_path="atec/training.csv",between_word="|")

    # show_not_ch(file_path="atec/training.csv",only_en=True,between_word="|")

    # label_distribution()

    # sentence_length_distribution()

    # from data.data_utils import split_train_val
    # split_train_val("atec/training.csv",10,6)

    get_corpus()
