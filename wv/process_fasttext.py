# coding: utf-8


if __name__=="__main__":
    """分析词向量
    """
    wv_path = "fasttext/wc-300.vec"
    # from wv.wv_utils import visualize_wv
    # visualize_wv(wv_path)

    from wv.wv_utils import analyze_wv_vocab_coverage
    analyze_wv_vocab_coverage(wv_path,"../data/atec/training.csv",2)
