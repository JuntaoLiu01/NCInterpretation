#_*_coding:utf-8_*_
import os
import sys
import json
import time
import random
import keras
# import umap
import numpy as np
import pandas as pd
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from bert4keras.snippets import DataGenerator,sequence_padding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.layers import Loss
from mlayer import Slice

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR,"data")
MODEL_DIR = os.path.join(ROOT_DIR,"models")

N = 23
max_text = 24

BERT_DIR = "bert"

config_path = os.path.join(MODEL_DIR,BERT_DIR,"bert_config.json")
checkpoint_path = os.path.join(MODEL_DIR,BERT_DIR,"bert_model.ckpt")
dict_path = os.path.join(MODEL_DIR,BERT_DIR,"vocab.txt")

tokenizer = Tokenizer(dict_path,do_lower_case=True)

def load_data(fn,shuffle=True):
    D = []
    with open(fn,"r",encoding="utf-8") as rf:
        for line in rf:
            line = json.loads(line)
            D.append([line["pair"],line["pos_sent"],line["neg_sents"]])
    if shuffle:
        random.shuffle(D)
    return D

class DG(DataGenerator):
    def __init__(self,data,batch_size,**kwargs):
        super(DG,self).__init__(data,batch_size,**kwargs)

    def __iter__(self,random=False):
        X1,X2,X3,X4,X5,X6 = [],[],[],[],[],[]
        for is_end,(pair,pos_sent,neg_sents) in self.sample(random=random):
            s,o = pair.split("##")
            compound_noun = o + s
            ## compound noun encoding
            token_ids,segment_ids = tokenizer.encode(compound_noun,maxlen=max_text)
            X1.append(token_ids)
            X2.append(segment_ids)
            ## postive sentence encoding
            s_token_ids,s_segment_ids = tokenizer.encode(pos_sent,maxlen=max_text)
            X3.append(s_token_ids)
            X4.append(s_segment_ids)

            ## negative sentence encoding
            text_list = []
            segment_list = []
            for neg_sent in neg_sents[:N]:
                n_token_ids,n_segment_ids = tokenizer.encode(neg_sent,maxlen=max_text)
                n_token_ids = n_token_ids + [0] * (max_text-len(n_token_ids))
                n_segment_ids = n_segment_ids + [0] *  (max_text-len(n_segment_ids))
                text_list += n_token_ids
                segment_list += n_segment_ids
            if len(neg_sents) < N:
                for _ in range(N-len(neg_sents)):
                    text_list += [101,102] + [0] * (max_text-2)
                    segment_list += [0] * max_text
            X5.append(text_list)
            X6.append(segment_list)
            
            if is_end or len(X1) == self.batch_size:
                X1 = sequence_padding(X1)
                X2 = sequence_padding(X2)
                X5 = sequence_padding(X5)
                X6 = sequence_padding(X6)
                X3 = sequence_padding(X3,length=max_text)
                X4 = sequence_padding(X4,length=max_text)
                yield [X1,X2,X3,X4,X5,X6],None
                X1,X2,X3,X4,X5,X6 = [],[],[],[],[],[]

class CPLoss(Loss):
    def __init__(self,temperature=0.1,cosine_sim=True,output_axis=1,**kwargs):
        self.temperature = temperature
        self.cosine_sim = cosine_sim
        super(CPLoss,self).__init__(output_axis,**kwargs)

    def compute_cosine_similarity(self,x1,x2):
        assert len(K.int_shape(x1)) == 2
        assert len(K.int_shape(x2)) == 2 or len(K.int_shape(x2)) == 3
        
        d2 = K.batch_dot(x1,x1)
        if len(K.int_shape(x2)) == 2:
            d1 = K.batch_dot(x1,x2)
            d3 = K.batch_dot(x2,x2)
        else:
            d1 = K.batch_dot(x1,K.permute_dimensions(x2,(0,2,1)))
            s_1,s_2 = K.int_shape(x2)[1:]
            tmp = K.reshape(x2,(-1,s_2))
            d3 = K.batch_dot(tmp,tmp)
            d3 = K.reshape(d3,(-1,s_1))
        denominator = K.maximum(K.sqrt(d2 * d3),K.epsilon())
        simliarity = d1/denominator
        return simliarity

    def compute_loss(self,inputs,mask=None):
        x,pos_x,neg_xx = inputs
        if not self.cosine_sim:
            x1 = K.batch_dot(x,pos_x)/self.temperature
            x2 = K.batch_dot(x,K.permute_dimensions(neg_xx,(0,2,1)))/self.temperature
        else:
            x1 = self.compute_cosine_similarity(x,pos_x)/self.temperature
            x2 = self.compute_cosine_similarity(x,neg_xx)/self.temperature
        tmp = K.concatenate([x1,x2],axis=-1)
        max_val = K.max(tmp,axis=1,keepdims=True)
        x1 = K.exp(x1-max_val)
        x2 = K.exp(x2-max_val)

        x1 = K.squeeze(x1,1)
        x2 = K.sum(x2,axis=-1)
        x3 = x1 + x2
        loss = K.mean(-K.log(x1/x3 + K.epsilon()))
        return loss

class SaveBest(keras.callbacks.Callback):
    def __init__(self,models,mfns,fn=None,**kwargs):
        self.pair_mlm = models[0]
        self.sent_mlm = models[1]
        self.pair_mfn = mfns[0]
        self.sent_mfn = mfns[1]
        self.val_loss = 100.
        data = []
        fn = os.path.join(DATA_DIR,"PA/valid.json")
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                data.append([d["pair"],d["pos_sent"],d["neg_sents"]])
        self.data = data
        super(SaveBest,self).__init__(**kwargs)

    def cal_accuracy(self,):
        acc_count = 0
        for d in self.data:
            s,o = d[0].split("##")
            token_ids,segment_ids = tokenizer.encode(o+s)
            pair_cls = self.pair_mlm.predict([np.array([token_ids]),np.array([segment_ids])])[0,0,:]
            cur_res = []
            candidates = [d[1]] + d[2]
            for sent in candidates:
                sent_token,sent_segment = tokenizer.encode(sent)
                sent_cls = self.sent_mlm.predict([np.array([sent_token]),np.array([sent_segment])])[0,0,:]
                cur_res.append(np.dot(pair_cls,sent_cls))
            cur_res = np.array(cur_res)
            y_pred = int(np.argmax(cur_res))
            if y_pred == 0:
                acc_count += 1
        return acc_count/len(self.data)

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get("val_loss")
        print("Epoch",epoch,": Accuracy: %f" % self.cal_accuracy())
        if val_loss < self.val_loss:
            self.val_loss = val_loss
            self.pair_mlm.save_weights(self.pair_mfn)
            self.sent_mlm.save_weights(self.sent_mfn)

class CPModel:
    def __init__(self,epochs,batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.pair_MLM = None
        self.sent_MLM = None

    def build(self,compile=True):
        in_1 = keras.layers.Input(shape=(None,))
        in_2 = keras.layers.Input(shape=(None,))
        in_3 = keras.layers.Input(shape=(max_text,))
        in_4 = keras.layers.Input(shape=(max_text,))

        in_5 = keras.layers.Input(shape=(max_text * N,))
        in_6 = keras.layers.Input(shape=(max_text * N,))

        self.pair_MLM = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path)
        self.sent_MLM = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path)
        CLS = keras.layers.Lambda(lambda x:x[:,0])
        TEXT_SLICE = Slice(N,max_text)
        
        NN_x = self.pair_MLM([in_1,in_2])
        NN_x = CLS(NN_x)
        POS_x = self.sent_MLM([in_3,in_4])
        POS_x = CLS(POS_x)

        in_5_slice  = TEXT_SLICE(in_5)
        in_6_slice  = TEXT_SLICE(in_6)
        mlm_outs = []
        for i in range(N):
            out = self.sent_MLM([in_5_slice[i],in_6_slice[i]])
            out = CLS(out)
            mlm_outs.append(out)
        NEG_x = keras.layers.Concatenate(axis=-1)(mlm_outs)
        NEG_x = keras.layers.Reshape(target_shape=(N,-1))(NEG_x)
        loss = CPLoss(0.1,True,1)([NN_x,POS_x,NEG_x])
        self.model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_6],outputs=loss)
        self.model.summary()
        if compile:
            self.model.compile(Adam(1e-5))

    def train(self,train_data,valid_data,mfns=None,eval_fn=None,plot=True):
        if not self.model:
            self.build(compile=True)
        if mfns:
            self.pair_MLM.load_weights(mfns[0])
            self.sent_MLM.load_weights(mfns[1])
        
        train_D = DG(train_data,self.batch_size)
        valid_D = DG(valid_data,self.batch_size)
        save_best = SaveBest([self.pair_MLM,self.sent_MLM],[os.path.join(MODEL_DIR,"model_CP_pair_v2.h5"),os.path.join(MODEL_DIR,"model_CP_sent_v2.h5")])
        callbacks = [save_best]
        history = self.model.fit_generator(
            train_D.forfit(),
            steps_per_epoch=len(train_D),
            epochs=self.epochs,
            validation_data=valid_D.forfit(),
            validation_steps=len(valid_D),
            callbacks=callbacks,
        )
        if plot:
            # print(history.history.keys())
            self.plot_results(history)

    def inference(self,fn,out_fn,mfns=None,infer_type="PAIR"):
        """
        infer_type:
            PAIR: samples with same pair are integrated as one case
            INSTANCE: samples with same pair and pos_sentence are integrated as one case
        """
        if not self.model:
            self.build(compile=False)
        if mfns:
            self.pair_MLM.load_weights(mfns[0])
            self.sent_MLM.load_weights(mfns[1])
        data_dict = {}
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                if not d["pair"] in data_dict:
                    data_dict[d["pair"]] = {
                        "pos_sents":[d["pos_sent"]],
                        "neg_sents":d["neg_sents"]
                    }
                else:
                    data_dict[d["pair"]]["pos_sents"].append(d["pos_sent"])
        data = []
        for pair in data_dict:
            if infer_type == "PAIR":
                data.append([pair,data_dict[pair]["pos_sents"],data_dict[pair]["neg_sents"]])
            else:
                for pos_sent in data_dict[pair]["pos_sents"]:
                    data.append([pair,[pos_sent],data_dict[pair]["neg_sents"]])
        res = []
        for d in data:
            s,o = d[0].split("##")
            token_ids,segment_ids = tokenizer.encode(o+s)
            pair_cls = self.pair_MLM.predict([np.array([token_ids]),np.array([segment_ids])])[0,0,:]
            cur_res = []
            tmp = []
            candidates = d[1] + d[2]
            for sent in candidates:
                sent_token,sent_segment = tokenizer.encode(sent)
                sent_cls = self.sent_MLM.predict([np.array([sent_token]),np.array([sent_segment])])[0,0,:]
                cur_res.append(np.dot(pair_cls,sent_cls))
                tmp.append((sent,float(np.dot(pair_cls,sent_cls))))
            cur_res = np.array(cur_res)
            y_pred = int(np.argmax(cur_res))
            x = d + [candidates[y_pred]]
            tmp = sorted(tmp,key=lambda x:x[1],reverse=True)
            x.append(tmp)
            res.append(x)
        with open(out_fn,"w",encoding="utf-8") as wf:
            for x in res:
                d = {
                    "pair":x[0],
                    "pos_sents":x[1],
                    "neg_sents":x[2],
                    "pred_sent":x[3],
                    "results":x[4]
                }
                wf.write(json.dumps(d,ensure_ascii=False)+"\n")

    def evaluate(self,fn,show_error=False):
        rank = []
        ## top 1,3,5,10
        topk = [0,0,0,0]
        count = 0
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                count += 1
                f_rank = False
                f_1 = False
                f_3 = False
                f_5 = False
                f_10 = False
                d = json.loads(line)
                for i,x in enumerate(d["results"]):
                    sent,score = x
                    if sent in d["pos_sents"]:
                        if not f_rank:
                            f_rank = True
                            rank.append(i+1)
                        if i < 1 and not f_1:
                            f_1 = True
                            topk[0] += 1
                        if i < 3 and not f_3:
                            f_3 = True
                            topk[1] += 1
                        if i < 5 and not f_5:
                            f_5 = True
                            topk[2] += 1
                        if i < 10 and not f_10:
                            f_10 = True
                            topk[3] += 1
        print("Rank: %f" % (sum(rank)/len(rank)))
        print("MRR: %f" % (len(rank)/sum(rank)))
        print("top@1: %f" % (topk[0]/count))
        print("top@3: %f" % (topk[1]/count))
        print("top@5: %f" % (topk[2]/count))
        print("top@10: %f" % (topk[3]/count))
        # if show_error:
        #     _d = os.path.dirname(fn)
        #     _f = os.path.basename(fn).replace(".","_error.")
        #     error_fn = os.path.join(_d,_f)
        #     with open(error_fn,"w",encoding="utf-8") as wf:
        #         for x in res:
        #             wf.write(json.dumps(x,ensure_ascii=False)+"\n")
                    
    def plot_results(self,history):
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.legend(["train","valid"],loc="upper left")
        plt.title("Model Loss");plt.xlabel("epoch");plt.ylabel("loss")
        plt.savefig(os.path.join(DATA_DIR,"imgs","CP_loss.pdf"))
        plt.close()

    def visualize(self,fn,out_fn,mfns=None):
        if not self.model:
            self.build(compile=False)
        if mfns:
            self.pair_MLM.load_weights(mfns[0])
            self.sent_MLM.load_weights(mfns[1])
        data_dict = {}
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                if not d["pair"] in data_dict:
                    data_dict[d["pair"]] = {
                        "pos_sents":[d["pos_sent"]],
                        "neg_sents":d["neg_sents"]
                    }
                else:
                    data_dict[d["pair"]]["pos_sents"].append(d["pos_sent"])
        data = []
        for pair in data_dict:
            data.append([pair,data_dict[pair]["pos_sents"],data_dict[pair]["neg_sents"]])
        embedding_data = []
        index_data = []
        for d in data:
            s,o = d[0].split("##")
            token_ids,segment_ids = tokenizer.encode(o+s)
            pair_cls = self.pair_MLM.predict([np.array([token_ids]),np.array([segment_ids])])[0,0,:]
            embedding_data.append(pair_cls)
            index_data.append("NC")
            for sent in d[1]:
                sent_token,sent_segment = tokenizer.encode(sent)
                sent_cls = self.sent_MLM.predict([np.array([sent_token]),np.array([sent_segment])])[0,0,:]
                embedding_data.append(sent_cls)
                index_data.append("Positive Para.")
            for sent in d[2]:
                sent_token,sent_segment = tokenizer.encode(sent)
                sent_cls = self.sent_MLM.predict([np.array([sent_token]),np.array([sent_segment])])[0,0,:]
                embedding_data.append(sent_cls)
                index_data.append("Negative Para.")
        embedding_data = np.array(embedding_data)
        # print(embedding_data.shape)
        u = umap.UMAP(random_state=42)
        umap_embs = u.fit_transform(embedding_data)
        # print(type(umap_embs))
        # print(type(umap_embs[:,0]))
        data = pd.DataFrame({"x":[float(v) for v in umap_embs[:,0]], "y": [float(v) for v in umap_embs[:,1]],"Node_Type":index_data})
        data.to_csv(out_fn)
     
def main():    
    train_fn = os.path.join(DATA_DIR,"PA/train.json")
    valid_fn = os.path.join(DATA_DIR,"PA/valid.json")
    
    train_data = load_data(train_fn)
    valid_data = load_data(valid_fn)
    model = CPModel(50,32)
    model.train(train_data,valid_data)

    out_fn = os.path.join(DATA_DIR,"PA/valid_res.json")
    # visualize_fn = os.path.join(DATA_DIR,"PA/visualize.csv")
    # pair_mfn = os.path.join(MODEL_DIR,"model_CP_pair.h5")
    # sent_mfn = os.path.join(MODEL_DIR,"model_CP_sent.h5")

    # model.visualize(valid_fn,visualize_fn,mfns=[pair_mfn,sent_mfn])
    # model.inference(valid_fn,out_fn,infer_type="PAIR",mfns=[pair_mfn,sent_mfn])
    model.evaluate(out_fn)

if __name__ == '__main__':
    main()