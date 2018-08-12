# coding: utf-8

from data.data_utils import ensure_dir_exist
import pandas as pd,numpy as np,logging,os


def my_logger(logging_path):
    # 生成日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.handlers = []
    assert len(logger.handlers) == 0
    handler = logging.FileHandler(logging_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def get_num_params(var_list,exclude_embedding_matrix=True):
    if exclude_embedding_matrix:
        return np.sum([np.prod(v.get_shape().as_list()) for v in var_list
                       if "embedding_matrix" not in v.name])
    else:
        return np.sum([np.prod(v.get_shape().as_list()) for v in var_list])


def update_history_summary(mode, history, summary, avg_l, avg_f, avg_a, avg_p, avg_r):
    assert mode in ["train", "valid"]
    history[mode + "_loss"].append(avg_l)
    history[mode + "_f1"].append(avg_f)
    history[mode + "_acc"].append(avg_a)
    history[mode + "_pre"].append(avg_p)
    history[mode + "_rec"].append(avg_r)
    if summary is not None:
        summary.value[0].simple_value = avg_l
        summary.value[1].simple_value = avg_f
        summary.value[2].simple_value = avg_a
        summary.value[3].simple_value = avg_p
        summary.value[4].simple_value = avg_r
    return history, summary


def WriteToSubmission(res,fileName):
    ensure_dir_exist(os.path.dirname(fileName))
    if isinstance(res,dict):
        res = [[int(key), int(value)] for (key, value) in res.items()]
    tmp=pd.DataFrame(res,columns=["id","label"])
    tmp=tmp.sort_values(by="id",axis=0,ascending=True)
    print(tmp)
    tmp.to_csv(fileName,sep='\t',header=False,index=False)
