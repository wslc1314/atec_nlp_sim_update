# coding: utf-8

import os, tensorflow as tf, pandas as pd, numpy as np
from utils.data_utils import createLocalWCDict
from utils.data_gens import DataIterator
from data.data_utils import ensure_dir_exist,read_cut_file
from models.model_utils import my_logger,get_num_params,update_history_summary,WriteToSubmission
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score


class Model:

    def __init__(self,config):
        self.config=None
        self.initializer=None
        self.graph=None
        self.X1w,self.X1c,self.X1w_l,self.X1c_l=None,None,None,None
        self.X2w,self.X2c,self.X2w_l,self.X2c_l=None,None,None,None
        self.X1w_mask,self.X2w_mask,self.X1c_mask,self.X2c_mask=None,None,None,None
        self.y=None
        self.keep_prob=None
        self.logits=None
        self.pos_prob=None
        self.loss_op=None
        self.var_list=None
        self.var_list_trainable=None
        self.train_op=None
        self.sgd_op,self.adam_op=None,None
        self.predicted=None
        self.summaries=None

    def build_graph(self):
        raise NotImplementedError

    def _get_train_feed_dict(self,batch):
        raise NotImplementedError

    def _get_valid_feed_dict(self,batch):
        raise NotImplementedError

    def _get_test_feed_dict(self,batch):
        raise NotImplementedError

    def fit(self, trainFile=None, validFile=None, with_validation=True,
            load_path=None, log_dir=None, save_dir=None, max_to_keep=5,
            num_epochs = 500, steps_every_epoch = 100, batch_size = 128,
            save_epochs = 10, early_stopping = 10, epoch_adam_to_sgd=501):
        self.config.with_validation = with_validation
        if trainFile is None:
            if with_validation:
                trainFile = self.config.local_train_file
                validFile = self.config.local_valid_file
            else:
                trainFile = self.config.global_train_file
                validFile = None
        else:
            if with_validation:
                if validFile is None:
                    validFile = trainFile.replace("train", "valid")
            else:
                validFile = None
        self.config.train_file,self.config.valid_file=trainFile,validFile
        # 训练过程中的日志保存文件以及模型保存路径
        if log_dir is None:
            log_dir = self.config.log_dir
        if save_dir is None:
            save_dir = self.config.save_dir
        if with_validation:
            log_dir = ensure_dir_exist(log_dir + "/trainval")
            train_dir = os.path.join(log_dir, "train")
            val_dir = os.path.join(log_dir, "valid")
            save_dir = ensure_dir_exist(save_dir + "/trainval")
        else:
            log_dir = ensure_dir_exist(log_dir + "/train")
            train_dir = log_dir
            val_dir = None
            save_dir = ensure_dir_exist(save_dir + "/train")
        self.config.log_dir, self.config.save_dir = log_dir, save_dir
        self.config.load_path = load_path
        self.config.max_to_keep=max_to_keep
        self.config.num_epochs=num_epochs
        self.config.steps_every_epoch=steps_every_epoch
        self.config.batch_size=batch_size
        self.config.save_epochs=save_epochs
        self.config.early_stopping=early_stopping
        self.config.epoch_adam_to_sgd=epoch_adam_to_sgd
        # 生成日志
        logger = my_logger(log_dir + "/log_fit.txt")
        msg_dict = {}
        msg_dict.update(self.config.__dict__)
        msg = "\n".join(["--" + key + ": %s" % value for (key, value) in msg_dict.items() if key[0] != '_'])
        logger.info(msg)
        # 定义数据生成器
        dictPath = trainFile.split(".")[0] + "-"+"-".join([str(i) for i in self.config.min_count_wc])+".json"
        if os.path.exists(dictPath):
            pass
        else:
            createLocalWCDict(trainFile, global_dict_path=self.config.global_dict)
        if self.config.wv_config["train_w"]:
            dictPathW=dictPath
        else:
            dictPathW=self.config.global_dict
        if self.config.wv_config["train_c"]:
            dictPathC=dictPath
        else:
            dictPathC=self.config.global_dict
        train_generator = DataIterator(trainFile, True, dictPathW,dictPathC,self.config.modeC,
                                       True)
        val_generator = None if validFile is None else DataIterator(validFile, True, dictPathW,dictPathC,self.config.modeC)

        history = {"train_loss": [], "train_f1": [], "train_acc": [], "train_pre": [], "train_rec": [],
                   "valid_loss": [], "valid_f1": [], "valid_acc": [], "valid_pre": [], "valid_rec": []}

        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=config, graph=self.graph) as sess:
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)
            val_writer = None if val_dir is None else tf.summary.FileWriter(val_dir)
            saver = tf.train.Saver(max_to_keep=self.config.max_to_keep,var_list=self.var_list)
            sess.run(tf.global_variables_initializer())
            start = 0
            if isinstance(load_path, str):
                if os.path.isdir(load_path):
                    if os.listdir(load_path):
                        ckpt = tf.train.get_checkpoint_state(load_path)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        start = ckpt.model_checkpoint_path.split("-")[-1]
                else:
                    saver.restore(sess, load_path)
                    start = load_path.split("-")[-1]
                logger.info("Loading successfully, loading epoch is %s" % start)
            logger.info("The total number of trainable variables(embedding matrix excluded): %s"
                        % get_num_params(self.var_list_trainable,True))
            cur_early_stopping = 0
            cur_f1 = [0.]*max_to_keep
            cur_max_f1=0.
            start = int(start)
            logger.info('******* start training with %d *******' % start)
            cur_steps = self.config.steps_every_epoch
            for epoch in range(start, self.config.num_epochs):
                if epoch+1<epoch_adam_to_sgd:
                    self.train_op=self.adam_op
                else:
                    self.train_op=self.sgd_op
                avg_loss_t, avg_f1_t, avg_acc_t, avg_p_t, avg_r_t = 0, 0, 0, 0, 0
                for step in range(cur_steps):
                    batch = train_generator.next(self.config.batch_size)
                    _, loss_t, pred = sess.run([self.train_op, self.loss_op, self.predicted],
                                               feed_dict=self._get_train_feed_dict(batch))
                    avg_loss_t += loss_t
                    avg_f1_t += f1_score(y_true=batch["label"], y_pred=pred)
                    avg_acc_t += accuracy_score(y_true=batch["label"], y_pred=pred)
                    avg_p_t += precision_score(y_true=batch["label"], y_pred=pred)
                    avg_r_t += recall_score(y_true=batch["label"], y_pred=pred)
                avg_loss_t /= cur_steps
                avg_f1_t /= cur_steps
                avg_acc_t /= cur_steps
                avg_p_t /= cur_steps
                avg_r_t /= cur_steps
                history, self.summaries = update_history_summary("train", history, self.summaries,
                                                                 avg_loss_t,avg_f1_t,avg_acc_t, avg_p_t, avg_r_t)
                train_writer.add_summary(summary=self.summaries, global_step=epoch + 1)
                avg_loss_v, avg_f1_v, avg_acc_v, avg_p_v, avg_r_v = 0, 0, 0, 0, 0
                if with_validation:
                    counts=val_generator.total_size//self.config.batch_size
                    for _ in range(counts):
                        batch = val_generator.next(self.config.batch_size)
                        loss_v, pred = sess.run([self.loss_op, self.predicted],
                                                feed_dict=self._get_valid_feed_dict(batch))
                        avg_loss_v += loss_v
                        avg_f1_v += f1_score(y_true=batch["label"], y_pred=pred)
                        avg_acc_v += accuracy_score(y_true=batch["label"], y_pred=pred)
                        avg_p_v += precision_score(y_true=batch["label"], y_pred=pred)
                        avg_r_v += recall_score(y_true=batch["label"], y_pred=pred)
                    avg_loss_v /= counts
                    avg_f1_v /= counts
                    avg_acc_v /= counts
                    avg_p_v /= counts
                    avg_r_v /= counts
                    history, self.summaries = update_history_summary("valid", history, self.summaries,
                                                                     avg_loss_v,avg_f1_v, avg_acc_v, avg_p_v, avg_r_v)
                    val_writer.add_summary(summary=self.summaries, global_step=epoch + 1)
                    logger.info("[%05d/%05d], "
                                "T-L: %.4f, T-F1: %.4f,T-A: %.4f,T-P: %.4f,T-R: %.4f, "
                                "V-L: %.4f, V-F1: %.4f,V-A: %.4f,V-P: %.4f,V-R: %.4f"
                                % (epoch + 1, self.config.num_epochs,
                                   avg_loss_t, avg_f1_t, avg_acc_t, avg_p_t, avg_r_t,
                                   avg_loss_v, avg_f1_v, avg_acc_v, avg_p_v, avg_r_v))
                    if avg_f1_v>min(cur_f1):
                        cur_early_stopping = 0
                        cur_f1.append(avg_f1_v)
                        cur_f1=cur_f1[1:]
                        assert len(cur_f1) == max_to_keep
                        if avg_f1_v > cur_max_f1:
                            cur_max_f1=avg_f1_v
                            logger.info("Saving model-%s" % (epoch + 1))
                            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=epoch + 1)
                    else:
                        cur_early_stopping += 1
                    if cur_early_stopping > self.config.early_stopping:
                        logger.info("Early stopping after epoch %s !" % epoch)
                        break
                else:
                    logger.info("[%05d/%05d], "
                                "T-L: %.4f, T-F1: %.4f,T-A: %.4f,T-P: %.4f,T-R: %.4f"
                                % (epoch + 1, self.config.num_epochs,
                                   avg_loss_t, avg_f1_t, avg_acc_t, avg_p_t, avg_r_t))
                    if (epoch - start + 1) % self.config.save_steps == 0:
                        logger.info("Saving model-%s" % (epoch + 1))
                        saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=epoch + 1)

    def evaluate(self, validFile=None, dictPath=None, load_path=None):
        """
        :param dictPath: 模型训练数据对应的local dict path
        """
        assert validFile is not None and dictPath is not None and load_path is not None
        if self.config.wv_config["train_w"]:
            dictPathW = dictPath
        else:
            dictPathW = self.config.global_dict
        if self.config.wv_config["train_c"]:
            dictPathC = dictPath
        else:
            dictPathC = self.config.global_dict
        val_generator = DataIterator(validFile, True, dictPathW,dictPathC,self.config.modeC)
        load_dir = load_path if os.path.isdir(load_path) else os.path.dirname(load_path)
        log_dir = ensure_dir_exist(load_dir.replace("checkpoints", "logs"))
        logger = my_logger(log_dir + "/log_evaluate.txt")
        logger.info("Evaluating with file: %s, local dict: %s..."%(validFile,dictPath))

        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=config, graph=self.graph) as sess:
            logger.info("Loading model...")
            saver = tf.train.Saver(self.var_list)
            if os.path.isdir(load_path):
                ckpt = tf.train.get_checkpoint_state(load_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("-")[-1]
            else:
                saver.restore(sess, load_path)
                global_step = load_path.split("-")[-1]
            logger.info("Loading successfully, loading epoch is %s" % global_step)
            batch = val_generator.next(1024, need_all=True)
            res={}
            while val_generator.loop == 0:
                pos_prob,pred = sess.run([self.pos_prob,self.predicted],
                                         feed_dict=self._get_valid_feed_dict(batch))
                for (id, p,la,pr) in zip(batch["id"], pos_prob,batch["label"],pred):
                    res[id] = [float(p),int(la),int(pr)]
                batch = val_generator.next(1024, need_all=True)
            res = [[int(key), float(value[0]),int(value[1]),int(value[2])] for (key, value) in res.items()]
            tmp = pd.DataFrame(res, columns=["id", "pos_prob","label","pred"])
            tmp = tmp.sort_values(by="id", axis=0, ascending=True)
            id = np.asarray(tmp["id"].values, dtype=np.int)
            id_v = read_cut_file(validFile, True)["id"]
            assert np.allclose(np.sort(id), np.array(id_v)), "Inconsistent indices!"
            for t in np.arange(0,1,0.05):
                pred=np.greater_equal(tmp["pos_prob"].values,np.asarray([t]))
                pred=np.asarray(pred,dtype=np.int)
                if t==0.5:
                    assert np.allclose(pred, tmp["pred"].values), "Inconsistent prediction!"
                f1=f1_score(y_pred=pred,y_true=tmp["label"])
                acc=accuracy_score(y_pred=pred,y_true=tmp["label"])
                pre=precision_score(y_pred=pred,y_true=tmp["label"])
                rec=recall_score(y_pred=pred,y_true=tmp["label"])
                logger.info("Threshold: %02f, F1: %.4f, A: %.4f, P: %.4f, R: %.4f"%(t,f1,acc,pre,rec))

    def predict(self, testFile=None, dictPath=None, load_path=None, is_save=True, resPath=None):
        assert testFile is not None and dictPath is not None and load_path is not None
        if self.config.wv_config["train_w"]:
            dictPathW = dictPath
        else:
            dictPathW = self.config.global_dict_path
        if self.config.wv_config["train_c"]:
            dictPathC = dictPath
        else:
            dictPathC = self.config.global_dict_path
        test_generator = DataIterator(testFile, False, dictPathW,dictPathC,self.config.modeC)
        load_dir = load_path if os.path.isdir(load_path) else os.path.dirname(load_path)

        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver(self.var_list)
            if os.path.isdir(load_path):
                ckpt = tf.train.get_checkpoint_state(load_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("-")[-1]
            else:
                saver.restore(sess, load_path)
                global_step = load_path.split("-")[-1]
            print("Loading successfully, loading epoch is %s" % global_step)

            batch = test_generator.next(1024, need_all=True)
            res = {}
            while test_generator.loop == 0:
                predicted = sess.run(self.predicted, feed_dict=self._get_test_feed_dict(batch))
                for (id, label) in zip(batch["id"], predicted):
                    res[id] = int(label)
                batch = test_generator.next(1024, need_all=True)
            if is_save:
                if resPath is None:
                    res_dir = ensure_dir_exist(load_dir.replace("checkpoints", "results"))
                    resPath = os.path.join(res_dir, "predicted.csv-" + str(global_step))
                # 用于存放测试识别结果
                WriteToSubmission(fileName=resPath, res=res)
        return res
