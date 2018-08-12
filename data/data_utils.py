# coding: utf-8

import codecs,jieba,os,json,re
import numpy as np
from sklearn.model_selection import KFold


def str_to_list(orig_str,delimiter="\t",
                encoding="utf-8",decoding="utf-8-sig"):
    return [s.encode(encoding).decode(decoding) for s in orig_str.strip().split(delimiter)]


def process_str(st,replace_space="，",
                remove_repetitive_punctuation=True,remove_digit=True,
                remove_continuous_rep_subst=True,min_substr_len=1,min_rep_num=3,
                show_len_reduce=10):
    orig_st=st[:]
    if replace_space is not None:
        st=st.replace(" ",replace_space)
    if remove_repetitive_punctuation:
        punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】" \
               "〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
        st=re.sub(u"([%s])+" % punc, r"\1", st)
    if remove_digit:
        st=''.join(i for i in st if not i.isdigit())
    if remove_continuous_rep_subst:
        i=0

        def replace_special_char(s):
            for sc in ".^$*+?\[]|{}()":
                s=s.replace(sc,"\%s"%sc)
            return s

        while i<len(st)-min_substr_len*min_rep_num:
            j = i + min_substr_len
            while j<len(st)-min_substr_len*(min_rep_num-1):
                if st[i]==st[j] and len(st)-i >= min_rep_num*(j-i):
                    st=re.sub(u"(%s){%d,}"%(replace_special_char(st[i:j]),min_rep_num), r"\1", st)
                    i=j
                    j = i + min_substr_len
                else:
                    j+=1
            i+=1
    if len(orig_st)-len(st)>show_len_reduce:
        print("Original: ",orig_st)
        print("Processed: ",st)
    return st


def participle(in_path,out_path,with_label=False,t2s_path=None):
    """对形如id 句子1 句子2 (标签)的文件进行分词并保存
    :param in_path: 单个文件路径
    :param out_path: 分词后所得文件的保存地址
    :param with_label: 输入文件是否带有标签
    :param t2s_path: 繁体转简体字典路径
    :return:
    """
    t2s=None
    if isinstance(t2s_path, str):
        t2s=loadDict(t2s_path)
    try:
        jieba.load_userdict("data/myDict.txt")
    except IOError:
        jieba.load_userdict("myDict.txt")
    else:
        print("Here needs correct path for dict!")
    target = codecs.open(out_path, 'w', encoding='utf-8')
    en2ch={"huabei":"花呗","jiebei":"借呗","mayi":"蚂蚁","xiugai": "修改", "zhifu": "支付",
           "zhifubao":"支付宝","mobike": "摩拜","zhebi":"这笔","xinyong":"信用","neng":"能",
           "buneng":"不能","keyi":"可以","tongguo":"通过","changshi":"尝试","bunengyongle":"不能用了",
           "mobie": "摩拜","feichang":"非常","huankuan":"还款","huanqian":"还钱","jieqian":"借钱",
           "shouqian":"收钱","shoukuan":"收款"}
    with codecs.open(in_path, 'r', encoding='utf-8') as f:
        print('open a file.')
        lineNum = 1
        line = f.readline()
        while line:
            print('---processing ', lineNum, ' article---')
            if isinstance(t2s_path,str):
                for k,v in t2s.items():
                    line=line.replace(k,v)
            for k,v in sorted(en2ch.items(),key=lambda x:len(x[0]),reverse=True):
                line = line.replace(k, v)
            line = str_to_list(line)
            # # 保留中英文
            # p = re.compile(u'[^\u4e00-\u9fa5a-z]')  # 中文的编码范围是：\u4e00到\u9fa5
            # line1 = " ".join(p.split(line[1])).strip()
            # line2 = " ".join(p.split(line[2])).strip()
            line1=process_str(line[1])
            line2=process_str(line[2])
            sent1 = '|'.join([w.strip() for w in jieba.cut(line1) if len(w.strip())>0])
            sent2 = '|'.join([w.strip() for w in jieba.cut(line2) if len(w.strip())>0])
            line_ = line[0] + '\t' + sent1 + '\t' + sent2
            if with_label:
                line_+='\t'+line[3]
            target.write(line_+'\n')
            lineNum = lineNum + 1
            line = f.readline()
    print('well done.')
    target.close()


def ensure_dir_exist(dir):
    if dir.strip()!="" and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def split_train_val(trainFile, num_split=10,random_state=19941229):
    """将训练数据划分为训练集和验证集
    """
    with codecs.open(trainFile, 'r',encoding='utf-8') as f:
        raw_data = f.readlines()
    kf=KFold(n_splits=num_split,shuffle=True,random_state=random_state)
    save_dir=ensure_dir_exist(os.path.join(os.path.dirname(trainFile),str(num_split)))
    count=0
    for train_index, test_index in kf.split(raw_data):
        train=np.asarray(raw_data)[train_index].tolist()
        test=np.asarray(raw_data)[test_index].tolist()
        with codecs.open(save_dir+"/train"+str(count)+".csv", 'w',encoding='utf-8') as f:
            f.writelines(train)
        with codecs.open(save_dir+"/valid"+str(count)+".csv", 'w',encoding='utf-8') as f:
            f.writelines(test)
        count+=1


def saveDict(dicts,saveFile):
    with open(saveFile,'w') as f:
        json.dump(dicts,f)


def loadDict(loadFile):
    with open(loadFile,'r') as f:
        dicts=json.load(f)
    return dicts


def read_cut_file(file_path,with_label=False,dictPathW=None,dictPathC=None,modeC=0):
    """对形如id 句子1 句子2 (标签)的分词后的文件进行读取
    :param modeC: 0 字表示句子；1 字表示词，词表示句子；>1 modeC个数的字表示词，词表示句子
    """
    index, label = [], []
    sent1, sent2,sent1_len,sent2_len=[],[],[],[]
    sent1c,sent2c,sent1c_len,sent2c_len=[],[],[],[]
    v2i_w,v2i_c=None,None
    if isinstance(dictPathW,str):
        v2i_w=loadDict(dictPathW)["word"]["v2i"]
    if isinstance(dictPathC,str):
        v2i_c = loadDict(dictPathC)["char"]["v2i"]
    with codecs.open(file_path,'r',encoding="utf-8") as f:
        raw_data=f.readlines()
        for line in raw_data:
            line=str_to_list(line)
            index.append(int(line[0]))
            tmp1 = [t.strip() for t in str_to_list(line[1],'|') if len(t.strip())>0]
            tmp2 = [t.strip() for t in str_to_list(line[2],'|') if len(t.strip())>0]
            if isinstance(dictPathW,str):
                sent1_=list(map(lambda s:int(v2i_w.get(s,v2i_w['<unk>'])),tmp1))
                sent2_=list(map(lambda s:int(v2i_w.get(s,v2i_w['<unk>'])),tmp2))
            else:
                sent1_ = tmp1[:]
                sent2_ = tmp2[:]
            if modeC==0:
                if isinstance(dictPathC, str):
                    sent1c_ = list(map(lambda s: int(v2i_c.get(s, v2i_c['<unk>'])), [_ for _ in ''.join(tmp1)]))
                    sent2c_ = list(map(lambda s: int(v2i_c.get(s, v2i_c['<unk>'])), [_ for _ in ''.join(tmp2)]))
                else:
                    sent1c_ = list([_ for _ in ''.join(tmp1)])
                    sent2c_ = list([_ for _ in ''.join(tmp2)])
                sent1c_len.append(len(sent1c_))
                sent2c_len.append(len(sent2c_))
            else:
                if isinstance(dictPathC, str):
                    sent1c_ = [list(map(lambda s: int(v2i_c.get(s, v2i_c['<unk>'])), t)) for t in tmp1]
                    sent2c_ = [list(map(lambda s: int(v2i_c.get(s, v2i_c['<unk>'])), t)) for t in tmp2]
                else:
                    sent1c_ = [[t_ for t_ in t] for t in tmp1]
                    sent2c_ = [[t_ for t_ in t] for t in tmp2]
                sent1c_len.append([len(s) for s in sent1c_])
                sent2c_len.append([len(s) for s in sent2c_])
                if modeC>1:
                    if isinstance(dictPathC,str):
                        sent1c_ = [(t+modeC*[v2i_c["<pad>"]])[:modeC] for t in sent1c_]
                        sent2c_ = [(t+modeC*[v2i_c["<pad>"]])[:modeC] for t in sent2c_]
                    else:
                        sent1c_ = [(t+ modeC * ["<pad>"])[:modeC] for t in sent1c_]
                        sent2c_ = [(t+ modeC * ["<pad>"])[:modeC] for t in sent2c_]
                    assert np.allclose(np.asarray([len(s) for s in sent1c_]),np.asarray([modeC]))
                    assert np.allclose(np.asarray([len(s) for s in sent2c_]),np.asarray([modeC]))
            sent1.append(sent1_)
            sent2.append(sent2_)
            sent1_len.append(len(sent1_))
            sent2_len.append(len(sent2_))
            sent1c.append(sent1c_)
            sent2c.append(sent2c_)
            if with_label:
                label.append(int(line[3]))
            else:
                label.append(None)
    res={"id":index,"label":label,
         "sent1w": sent1, "sent2w": sent2,"sent1w_len":sent1_len,"sent2w_len":sent2_len,
         "sent1c":sent1c,"sent2c":sent2c,"sent1c_len":sent1c_len,"sent2c_len":sent2c_len}
    return res
