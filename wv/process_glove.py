# coding: utf-8

import shutil,codecs


def getFileLineNums(filename):
    """
    计算行数，就是单词数。
    """
    f = codecs.open(filename, 'r',"utf-8")
    count = 0
    for _ in f:
        count += 1
    return count


def prepend_line(infile, outfile, line):
    """
    打开词向量文件，在开始增加一行。
    """
    with codecs.open(infile, 'r',"utf-8") as old:
        with codecs.open(outfile, 'w',"utf-8") as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def processGloVe(filename):
    """
    将用GloVe训练生成的词向量转成能用gensim打开的形式。
    用gensim打开glove词向量需要在向量的开头增加一行：所有的单词数 词向量的维度。
    """
    num_lines = getFileLineNums(filename)
    filename_=filename.replace(".txt","")
    size=int(filename_.split("-")[-1])
    first_line = "{} {}".format(num_lines, size)
    prepend_line(filename, filename_, first_line)


if __name__ == "__main__":

    """atec
    """
    for size in [300]:
        filename="glove/atec_char-2-"+str(size)+".txt"
        processGloVe(filename)
        filename = "glove/atec_word-2-" + str(size) + ".txt"
        processGloVe(filename)

    """分析词向量
    """
    from wv.wv_utils import visualize_wv
    wv_path = "glove/"
    for i in [300]:
        visualize_wv(wv_path + "atec_char-2-" + str(i))
    for i in [300]:
        visualize_wv(wv_path + "atec_word-2-" + str(i))

    from wv.wv_utils import analyze_wv_vocab_coverage
    analyze_wv_vocab_coverage("glove/atec_char-2-300","../data/atec/training.csv",2)
    analyze_wv_vocab_coverage("glove/atec_word-2-300","../data/atec/training.csv",2)
