# coding: utf-8

from data.data_utils import read_cut_file
import pandas as pd, numpy as np


def DataIterator(filePath,with_label,dictPathW,dictPathC,modeC,
                 is_train=False):
    assert modeC==0 or modeC>1,"Unsupported modeC!"
    # return PaddedDataIteratorCut(filePath,with_label,dictPathW,dictPathC,modeC,is_train,20,40)
    return BucketedDataIteratorSimple(filePath,with_label,dictPathW,dictPathC,modeC,is_train,5)


class BucketedDataIteratorSimple(object):
    def __init__(self, filePath,with_label,dictPathW,dictPathC,modeC,
                 is_train=False,num_buckets=5):
        raw_data = read_cut_file(filePath, with_label, dictPathW,dictPathC,modeC)
        id, la = raw_data['id'], raw_data['label']
        s1w, s2w, s1wl, s2wl = raw_data["sent1w"], raw_data['sent2w'], raw_data['sent1w_len'], raw_data['sent2w_len']
        s1c, s2c, s1cl, s2cl = raw_data["sent1c"], raw_data['sent2c'], raw_data['sent1c_len'], raw_data['sent2c_len']
        if is_train:
            # 对调句1和句2进行数据扩充
            self.df = pd.DataFrame({"id": id + id, "label": la + la,
                                    "sen1w": s1w + s2w, "sen2w": s2w + s1w,
                                    "sen1w_len": s1wl + s2wl, "sen2w_len": s2wl + s1wl,
                                    "sen1c": s1c + s2c, "sen2c": s2c + s1c,
                                    "sen1c_len": s1cl + s2cl, "sen2c_len": s2cl + s1cl,
                                    })
        else:
            self.df = pd.DataFrame({"id": id, "label": la,
                                    "sen1w": s1w, "sen2w": s2w, "sen1w_len": s1wl, "sen2w_len": s2wl,
                                    "sen1c": s1c, "sen2c": s2c, "sen1c_len": s1cl, "sen2c_len": s2cl})
        if with_label:
            df_pos = self.df[self.df["label"] == 1]
            df_neg = self.df[self.df["label"] == 0]
            pn_rate_orig = len(df_pos) / float(len(df_neg))
            print("pn_rate_orig: %f"%pn_rate_orig)
        df = self.df.sort_values("sen1w_len").reset_index(drop=True)
        self.total_size = len(df)
        part_size = self.total_size // num_buckets
        self.dfs = []
        for i in range(num_buckets):
            self.dfs.append(df.ix[i * part_size:(i + 1) * part_size - 1])
        self.dfs[num_buckets-1]=self.dfs[num_buckets - 1].append(df.ix[num_buckets * part_size:self.total_size - 1])
        self.num_buckets = num_buckets
        self.cursor = np.array([0] * num_buckets)
        self.p_list = [1 / self.num_buckets] * self.num_buckets
        self.loop = -1
        self.shuffle()
        self.modeC=modeC

    def shuffle(self):
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
        self.p_list = [1 / self.num_buckets] * self.num_buckets
        self.loop+=1

    def next(self, batch_size, need_all=False):
        for i in range(self.num_buckets):
            if need_all:
                if self.cursor[i]>=len(self.dfs[i]):
                    self.p_list[i]=0
            else:
                if self.cursor[i]+batch_size>=len(self.dfs[i]):
                    self.p_list[i] = 0
        if sum(self.p_list) == 0:
            self.shuffle()
        else:
            times = 1 / sum(self.p_list)
            self.p_list = [times * p for p in self.p_list]
        selected=np.random.choice(a=np.arange(self.num_buckets),size=1,p=self.p_list)[0]
        if need_all:
            batch_size=min(batch_size,len(self.dfs[selected])-self.cursor[selected])
        res=self.dfs[selected].ix[self.cursor[selected]:self.cursor[selected]+batch_size-1,:]
        self.cursor[selected]+=batch_size
        max_len1=max(res["sen1w_len"].values)
        max_len2 =max(res["sen2w_len"].values)
        res1=np.zeros(shape=[batch_size,max_len1],dtype=np.int32)
        res2=np.zeros(shape=[batch_size,max_len2],dtype=np.int32)
        if self.modeC==0:
            max_len1_c = max(res["sen1c_len"].values)
            max_len2_c = max(res["sen2c_len"].values)
            res1_c = np.zeros(shape=[batch_size, max_len1_c], dtype=np.int32)
            res2_c = np.zeros(shape=[batch_size, max_len2_c], dtype=np.int32)
            for idx in range(batch_size):
                # 少的pad。
                res1[idx,:res["sen1w_len"].values[idx]]=res["sen1w"].values[idx]
                res2[idx,:res["sen2w_len"].values[idx]]=res["sen2w"].values[idx]
                res1_c[idx, :res["sen1c_len"].values[idx]] = res["sen1c"].values[idx]
                res2_c[idx, :res["sen2c_len"].values[idx]] = res["sen2c"].values[idx]
        else:
            res1_c = np.zeros(shape=[batch_size, max_len1,self.modeC], dtype=np.int32)
            res2_c = np.zeros(shape=[batch_size, max_len2,self.modeC], dtype=np.int32)
            for idx in range(batch_size):
                # 少的pad。
                res1[idx, :res["sen1w_len"].values[idx]] = res["sen1w"].values[idx]
                res2[idx, :res["sen2w_len"].values[idx]] = res["sen2w"].values[idx]
                for jdx in range(res["sen1w_len"].values[idx]):
                    res1_c[idx,jdx] = res["sen1c"].values[idx][jdx]
                for jdx in range(res["sen2w_len"].values[idx]):
                    res2_c[idx,jdx] = res["sen2c"].values[idx][jdx]
        final_res = pd.DataFrame({"id": res["id"].values, "label": res["label"].values,
                                  "sen1w": res1.tolist(), "sen2w": res2.tolist(),
                                  # "sen1w_len": res["sen1w_len"].values, "sen2w_len": res["sen2w_len"].values,
                                  "sen1c": res1_c.tolist(), "sen2c": res2_c.tolist(),})
                                  # "sen1c_len": res["sen1c_len"].values, "sen2c_len": res["sen2c_len"].values})
        return final_res


class PaddedDataIteratorSimple(object):
    def __init__(self, filePath,with_label,dictPathW,dictPathC,modeC,
                 is_train=False):
        raw_data = read_cut_file(filePath, with_label, dictPathW,dictPathC,modeC)
        id, la = raw_data['id'], raw_data['label']
        s1w, s2w, s1wl, s2wl = raw_data["sent1w"], raw_data['sent2w'], raw_data['sent1w_len'], raw_data['sent2w_len']
        s1c, s2c, s1cl, s2cl = raw_data["sent1c"], raw_data['sent2c'], raw_data['sent1c_len'], raw_data['sent2c_len']
        if is_train:
            # 对调句1和句2进行数据扩充
            self.df = pd.DataFrame({"id": id + id, "label": la + la,
                                    "sen1w": s1w + s2w, "sen2w": s2w + s1w,
                                    "sen1w_len": s1wl + s2wl, "sen2w_len": s2wl + s1wl,
                                    "sen1c": s1c + s2c, "sen2c": s2c + s1c,
                                    "sen1c_len": s1cl + s2cl, "sen2c_len": s2cl + s1cl,
                                    })
        else:
            self.df = pd.DataFrame({"id": id, "label": la,
                                    "sen1w": s1w, "sen2w": s2w, "sen1w_len": s1wl, "sen2w_len": s2wl,
                                    "sen1c": s1c, "sen2c": s2c, "sen1c_len": s1cl, "sen2c_len": s2cl})
        if with_label:
            df_pos = self.df[self.df["label"] == 1]
            df_neg = self.df[self.df["label"] == 0]
            pn_rate_orig = len(df_pos) / float(len(df_neg))
            print("pn_rate_orig: %f"%pn_rate_orig)
        self.total_size = len(self.df)
        self.cursor = 0
        self.loop = -1
        self.shuffle()
        self.modeC=modeC

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0
        self.loop+=1

    def next(self, batch_size, need_all=False):
        if need_all:
            if self.cursor>=self.total_size:
                self.shuffle()
        else:
            if self.cursor+batch_size>=self.total_size:
                self.shuffle()
        if need_all:
            batch_size=min(batch_size,self.total_size-self.cursor)
        res=self.df.ix[self.cursor:self.cursor+batch_size-1,:]
        self.cursor+=batch_size
        max_len1=max(res["sen1w_len"].values)
        max_len2 =max(res["sen2w_len"].values)
        res1=np.zeros(shape=[batch_size,max_len1],dtype=np.int32)
        res2=np.zeros(shape=[batch_size,max_len2],dtype=np.int32)
        if self.modeC==0:
            max_len1_c = max(res["sen1c_len"].values)
            max_len2_c = max(res["sen2c_len"].values)
            res1_c = np.zeros(shape=[batch_size, max_len1_c], dtype=np.int32)
            res2_c = np.zeros(shape=[batch_size, max_len2_c], dtype=np.int32)
            for idx in range(batch_size):
                # 少的pad。
                res1[idx,:res["sen1w_len"].values[idx]]=res["sen1w"].values[idx]
                res2[idx,:res["sen2w_len"].values[idx]]=res["sen2w"].values[idx]
                res1_c[idx, :res["sen1c_len"].values[idx]] = res["sen1c"].values[idx]
                res2_c[idx, :res["sen2c_len"].values[idx]] = res["sen2c"].values[idx]
        else:
            res1_c = np.zeros(shape=[batch_size, max_len1,self.modeC], dtype=np.int32)
            res2_c = np.zeros(shape=[batch_size, max_len2,self.modeC], dtype=np.int32)
            for idx in range(batch_size):
                # 少的pad。
                res1[idx, :res["sen1w_len"].values[idx]] = res["sen1w"].values[idx]
                res2[idx, :res["sen2w_len"].values[idx]] = res["sen2w"].values[idx]
                for jdx in range(res["sen1w_len"].values[idx]):
                    res1_c[idx,jdx] = res["sen1c"].values[idx][jdx]
                for jdx in range(res["sen2w_len"].values[idx]):
                    res2_c[idx,jdx] = res["sen2c"].values[idx][jdx]
        final_res = pd.DataFrame({"id": res["id"].values, "label": res["label"].values,
                                  "sen1w": res1.tolist(), "sen2w": res2.tolist(),
                                  # "sen1w_len": res["sen1w_len"].values, "sen2w_len": res["sen2w_len"].values,
                                  "sen1c": res1_c.tolist(), "sen2c": res2_c.tolist(),})
                                  # "sen1c_len": res["sen1c_len"].values, "sen2c_len": res["sen2c_len"].values})
        return final_res


class PaddedDataIteratorCut(object):
    def __init__(self, filePath,with_label,dictPathW,dictPathC,modeC,
                 is_train=False,max_len_w=20,max_len_c=40):
        raw_data = read_cut_file(filePath, with_label, dictPathW,dictPathC,modeC)
        id, la = raw_data['id'], raw_data['label']
        s1w, s2w, s1wl, s2wl = raw_data["sent1w"], raw_data['sent2w'], raw_data['sent1w_len'], raw_data['sent2w_len']
        s1c, s2c, s1cl, s2cl = raw_data["sent1c"], raw_data['sent2c'], raw_data['sent1c_len'], raw_data['sent2c_len']
        if is_train:
            # 对调句1和句2进行数据扩充
            self.df = pd.DataFrame({"id": id + id, "label": la + la,
                                    "sen1w": s1w + s2w, "sen2w": s2w + s1w,
                                    "sen1w_len": s1wl + s2wl, "sen2w_len": s2wl + s1wl,
                                    "sen1c": s1c + s2c, "sen2c": s2c + s1c,
                                    "sen1c_len": s1cl + s2cl, "sen2c_len": s2cl + s1cl,
                                    })
        else:
            self.df = pd.DataFrame({"id": id, "label": la,
                                    "sen1w": s1w, "sen2w": s2w, "sen1w_len": s1wl, "sen2w_len": s2wl,
                                    "sen1c": s1c, "sen2c": s2c, "sen1c_len": s1cl, "sen2c_len": s2cl})
        if with_label:
            df_pos = self.df[self.df["label"] == 1]
            df_neg = self.df[self.df["label"] == 0]
            pn_rate_orig = len(df_pos) / float(len(df_neg))
            print("pn_rate_orig: %f"%pn_rate_orig)
        self.total_size = len(self.df)
        res1 = np.zeros(shape=[self.total_size, max_len_w], dtype=np.int32)
        res2 = np.zeros(shape=[self.total_size, max_len_w], dtype=np.int32)
        self.df["sen1w_len"] = min_ele_array(self.df["sen1w_len"].values, max_len_w)
        self.df["sen2w_len"] = min_ele_array(self.df["sen2w_len"].values, max_len_w)
        if modeC == 0:
            res1_c = np.zeros(shape=[self.total_size, max_len_c], dtype=np.int32)
            res2_c = np.zeros(shape=[self.total_size, max_len_c], dtype=np.int32)
            self.df["sen1c_len"] = min_ele_array(self.df["sen1c_len"].values, max_len_c)
            self.df["sen2c_len"] = min_ele_array(self.df["sen2c_len"].values, max_len_c)
            for idx in range(self.total_size):
                # 少的pad。
                res1[idx, :self.df["sen1w_len"].values[idx]] = \
                    self.df["sen1w"].values[idx][:self.df["sen1w_len"].values[idx]]
                res2[idx, :self.df["sen2w_len"].values[idx]] = \
                    self.df["sen2w"].values[idx][:self.df["sen2w_len"].values[idx]]
                res1_c[idx, :self.df["sen1c_len"].values[idx]] = \
                    self.df["sen1c"].values[idx][:self.df["sen1c_len"].values[idx]]
                res2_c[idx, :self.df["sen2c_len"].values[idx]] = \
                    self.df["sen2c"].values[idx][:self.df["sen2c_len"].values[idx]]
        else:
            res1_c = np.zeros(shape=[self.total_size, max_len_w, modeC], dtype=np.int32)
            res2_c = np.zeros(shape=[self.total_size, max_len_w, modeC], dtype=np.int32)
            for idx in range(self.total_size):
                # 少的pad。
                res1[idx, :self.df["sen1w_len"].values[idx]] = \
                    self.df["sen1w"].values[idx][:self.df["sen1w_len"].values[idx]]
                res2[idx, :self.df["sen2w_len"].values[idx]] = \
                    self.df["sen2w"].values[idx][:self.df["sen2w_len"].values[idx]]
                for jdx in range(self.df["sen1w_len"].values[idx]):
                    res1_c[idx, jdx] = self.df["sen1c"].values[idx][jdx]
                for jdx in range(self.df["sen2w_len"].values[idx]):
                    res2_c[idx, jdx] = self.df["sen2c"].values[idx][jdx]
        self.df["sen1w"]=res1.tolist()
        self.df["sen2w"]=res2.tolist()
        self.df["sen1c"]=res1_c.tolist()
        self.df["sen2c"]=res2_c.tolist()
        self.cursor = 0
        self.loop = -1
        self.shuffle()
        self.modeC=modeC

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0
        self.loop+=1

    def next(self, batch_size, need_all=False):
        if need_all:
            if self.cursor>=self.total_size:
                self.shuffle()
        else:
            if self.cursor+batch_size>=self.total_size:
                self.shuffle()
        if need_all:
            batch_size=min(batch_size,self.total_size-self.cursor)
        res=self.df.ix[self.cursor:self.cursor+batch_size-1,:]
        self.cursor+=batch_size
        final_res = pd.DataFrame({"id": res["id"].values, "label": res["label"].values,
                                  "sen1w": res["sen1w"].values, "sen2w": res["sen2w"].values,
                                  "sen1c": res["sen1c"].values, "sen2c": res["sen2c"].values})
        return final_res


def min_ele_array(a,val):
    def f(a_ele,value):
        if a_ele>value:
            return value
        else:
            return a_ele
    v_f=np.vectorize(f)
    return v_f(a,val)
