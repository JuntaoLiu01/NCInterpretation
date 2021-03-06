#_*_coding:utf-8_*_
import os
import re
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec

# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# conf = tf.ConfigProto()
# conf.gpu_options.allow_growth = True
# sess = tf.Session(config=conf)
# set_session(sess)

import keras
from keras_gcn import GraphConv
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from mlayer import *
from utils import plot_confusion, cal_metrics

max_text = 120
bag_num = 5

global_node_num = 68
global_node_count = 4785
global_emd_size = 50
global_gcn_size = 50

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR,"data")
MODEL_DIR = os.path.join(ROOT_DIR,"models")

def init_tokenizer(dict_path):
    def pre_tokenize(text):
        return re.split("(\[e1\]|\[/e1\]|\[e2\]|\[/e2\])",text)
    tokenizer = Tokenizer(dict_path,do_lower_case=True,pre_tokenize=pre_tokenize)
    new_words = ["[e1]","[/e1]","[e2]","[/e2]","[blank]"]
    for idx in range(1,len(new_words)+1):
        tokenizer._token_dict_inv[idx] = new_words[idx-1]
    tokenizer._token_dict = {v: k for k, v in tokenizer._token_dict_inv.items()}
    return tokenizer

def load_data(fn,shuffle=True):
    D = []
    with open(fn,"r",encoding="utf-8") as rf:
        for line in rf:
            line = json.loads(line)
            D.append([line["subject"],line["object"],line["texts"],line["predicate"]])
    if shuffle:
        random.shuffle(D)
    return D

class DG(DataGenerator):
    def __init__(self,data,batch_size,tokenizer,pair_info=True,global_info=False,ugc_info=True,gcn_type="gcn",atten_type=-1,**kwargs):
        self._global = global_info
        self.ugc_info = ugc_info
        self.pair_info = pair_info
        self.tokenizer = tokenizer
        self.gcn_type = gcn_type
        self.atten_type = atten_type

        if self._global:
            global_node_fn = os.path.join(DATA_DIR,"RC/{}_data/node2id.json".format(self.gcn_type))
            global_graph_fn = os.path.join(DATA_DIR,"RC/{}_data/global_graph.json".format(self.gcn_type))
            self.global_node2id = json.load(open(global_node_fn,"r",encoding="utf-8"))
            self.global_graph = json.load(open(global_graph_fn,"r",encoding="utf-8"))
        if self.pair_info:
            word_fn = os.path.join(DATA_DIR,"nodeVec/word2id.json")
            self.word2id = json.load(open(word_fn,"r",encoding="utf-8"))

        self.rel2id = json.load(open(os.path.join(DATA_DIR,"relation.json"),"r",encoding="utf-8"))
        super(DG,self).__init__(data,batch_size,**kwargs)

    def build_global_graph(self,s,o):
        X = np.zeros(shape=(global_node_num,),dtype="int32")
        M = np.eye(global_node_num,dtype="int32")
        node_id = dict()
        node_id[s] = 0
        node_id[o] = 1
        X[node_id[s]] = self.global_node2id.get(s,2)
        X[node_id[o]] = self.global_node2id.get(o,2)
        M[node_id[s]][node_id[o]] = 1
        M[node_id[o]][node_id[s]] = 1
        for so in self.global_graph[s]:
            if not so in node_id:
                node_id[so] = len(node_id)
            X[node_id[so]] = self.global_node2id.get(so,2)
            M[node_id[so]][node_id[s]] = 1
            M[node_id[s]][node_id[so]] = 1
        for oss in self.global_graph[o]:
            if not oss in node_id:
                node_id[oss] = len(node_id)
            X[node_id[oss]] = self.global_node2id.get(oss,2)
            M[node_id[oss]][node_id[o]] = 1
            M[node_id[o]][node_id[oss]] = 1
        return X,M

    def build_relational_graph(self,s,o):
        X = np.zeros(shape=(global_node_num,),dtype="int32")
        Mt = np.zeros(shape=(global_node_num,global_node_num),dtype="int32")
        Mr = np.zeros(shape=(global_node_num,global_node_num),dtype="int32")
        # Mrr = np.zeros(shape=(global_node_num,global_node_num),dtype="int32")
        node_id = dict()
        node_id[s] = 0
        node_id[o] = 1
        X[node_id[s]] = self.global_node2id.get(s,2)
        X[node_id[o]] = self.global_node2id.get(o,2)
        for p,so in self.global_graph[s]:
            if not so in node_id:
                node_id[so] = len(node_id)
            X[node_id[so]] = self.global_node2id.get(so,2)
            if p == "isA":
                Mt[node_id[so]][node_id[s]] = 1
                Mt[node_id[s]][node_id[so]] = 1
            else:
                Mr[node_id[so]][node_id[s]] = 1
                Mr[node_id[s]][node_id[so]] = 1
            # elif p.startswith("r_"):
            #     Mr[node_id[so]][node_id[s]] = 1
            #     Mrr[node_id[s]][node_id[so]] = 1
            # else:
            #     Mr[node_id[s]][node_id[so]] = 1
            #     Mrr[node_id[so]][node_id[s]] = 1

        for p,oss in self.global_graph[o]:
            if not oss in node_id:
                node_id[oss] = len(node_id)
            X[node_id[oss]] = self.global_node2id.get(oss,2)
            if p == "isA":
                Mt[node_id[oss]][node_id[o]] = 1
                Mt[node_id[o]][node_id[oss]] = 1
            else:
                Mr[node_id[oss]][node_id[o]] = 1
                Mr[node_id[o]][node_id[oss]] = 1
            # elif p.startswith("r_"):
            #     Mr[node_id[oss]][node_id[o]] = 1
            #     Mrr[node_id[o]][node_id[oss]] = 1
            # else:
            #     Mr[node_id[o]][node_id[oss]] = 1
            #     Mrr[node_id[oss]][node_id[o]] = 1
        # return X,Mt,Mr,Mrr
        return X,Mt,Mr
    
    def __iter__(self,random=False):
        X1,X2,X3,X4,X5,X6,X7,X8,X9,Y = [],[],[],[],[],[],[],[],[],[]
        ## X1 used for word_id; X2,X3,X4,X5 used for UGC; X6 used for UGC attention; X7,X8 used for global graph; X9 used for relational graph
        for is_end,(subj,obj,texts,label) in self.sample(random):
            y = [0 for _ in range(len(self.rel2id))]
            y[self.rel2id.get(label,0)] = 1
            Y.append(y)

            if self.pair_info:
                subj_id,obj_id = self.word2id.get(subj,1),self.word2id.get(obj,1)
                X1.append([subj_id,obj_id])

            if self.ugc_info:
                text_list = []
                segment_list = []
                head_index = []
                tail_index = []
                for text in texts[:bag_num]:
                    token_ids, segment_ids = self.tokenizer.encode(text,maxlen=max_text)
                    token_ids = token_ids + [0] * (max_text-len(token_ids))
                    segment_ids = segment_ids + [0] *  (max_text-len(segment_ids))
                    head_index.append(token_ids.index(1))
                    tail_index.append(token_ids.index(3))
                    text_list += token_ids
                    segment_list += segment_ids
                if len(texts) < bag_num:
                    for _ in range(bag_num-len(texts)):
                        text_list += [101,102] + [0] * (max_text-2)
                        segment_list += [0] * max_text
                        head_index.append(0)
                        tail_index.append(1)
                X2.append(text_list)
                X3.append(segment_list)
                X4.append(head_index)
                X5.append(tail_index)
                if self.atten_type == 2:
                    X6.append([self.rel2id.get(label,0)])

            if self._global:
                if self.gcn_type == "gcn":
                    global_node_ids,global_node_matrix = self.build_global_graph(subj,obj)
                    X7.append(global_node_ids)
                    X8.append(global_node_matrix)
                else:
                    global_node_ids,global_Mt,global_Mr = self.build_relational_graph(subj,obj)
                    X7.append(global_node_ids)
                    X8.append(global_Mt)
                    X9.append(global_Mr)

            if is_end or len(Y) == self.batch_size:
                Y = sequence_padding(Y)
                if self.pair_info:
                    X1 = sequence_padding(X1)
                if self.ugc_info:
                    X2 = sequence_padding(X2)
                    X3 = sequence_padding(X3)
                    X4 = sequence_padding(X4)
                    X5 = sequence_padding(X5)
                    if self.atten_type == 2:
                        X6 = sequence_padding(X6)
                if self._global:
                    X7 = np.array(X7)
                    X8 = np.array(X8)
                    if self.gcn_type == "rgcn":
                        X9 = np.array(X9)

                if self.ugc_info and self._global and self.pair_info:
                    if self.atten_type == 2:
                        if self.gcn_type == "gcn":
                            yield [X1,X2,X3,X4,X5,X6,X7,X8],Y
                        else:
                            yield [X1,X2,X3,X4,X5,X6,X7,X8],Y
                    else:
                        if self.gcn_type == "gcn":
                            yield [X1,X2,X3,X4,X5,X7,X8],Y
                        else:
                            yield [X1,X2,X3,X4,X5,X7,X8,X9],Y
                elif self.pair_info and self.ugc_info:
                    if self.atten_type == 2:
                        yield [X1,X2,X3,X4,X5,X6],Y
                    else:
                        yield [X1,X2,X3,X4,X5],Y
                elif self.pair_info and self._global:
                    if self.gcn_type == "gcn":
                        yield [X1,X7,X8],Y
                    else:
                        yield [X1,X7,X8,X9],Y
                elif self._global and self.ugc_info:
                    if self.atten_type == 2:
                        if self.gcn_type == "gcn":
                            yield [X2,X3,X4,X5,X6,X7,X8],Y
                        else:
                            yield [X2,X3,X4,X5,X6,X7,X8,X9],Y
                    else:
                        if self.gcn_type == "gcn":
                            yield [X2,X3,X4,X5,X7,X8],Y
                        else:
                            yield [X2,X3,X4,X5,X7,X8,X9],Y
                elif self.ugc_info:
                    if self.atten_type == 2:
                        yield [X2,X3,X4,X5,X6],Y
                    else:
                        yield [X2,X3,X4,X5],Y
                elif self._global:
                    if self.gcn_type == "gcn":
                        yield [X7,X8],Y
                    else:
                        yield [X7,X8,X9],Y
                
                X1,X2,X3,X4,X5,X6,X7,X8,X9,Y = [],[],[],[],[],[],[],[],[],[]

def build_embedding_matrix(model_name,level="word"):
    train_fn = os.path.join(DATA_DIR,"RC/train.json")
    valid_fn = os.path.join(DATA_DIR,"RC/valid.json")
    node2id = {}
    emd_matrix = []
    w2vmodel = Word2Vec.load(os.path.join(DATA_DIR,"nodeVec",model_name))
    word_size = w2vmodel.wv.vector_size

    node2id["PAD"] = len(node2id)
    node2id["UNK"] = len(node2id)
    emd_matrix.append(np.zeros(word_size,))
    emd_matrix.append(np.random.rand(word_size) * 2 - 1)
    m = 0
    for fn in [train_fn,valid_fn]:
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                s,o = d["subject"],d["object"]
                if level == "word":
                    for x in [s,o]:
                        if not x in node2id:
                            if x in w2vmodel.wv:
                                node2id[x] = len(node2id)
                                emd = w2vmodel.wv[x]
                                emd_matrix.append(emd)
                else:
                    m = max(m,len(s+o))
                    for ch in s+o:
                        if not ch in node2id:
                            if ch in w2vmodel.wv:
                                node2id[ch] = len(node2id)
                                emd = w2vmodel.wv[ch]
                                emd_matrix.append(emd)
    with open(os.path.join(DATA_DIR,"nodeVec/{}2id.json".format(level)),"w",encoding="utf-8") as wf:
        json.dump(node2id,wf,ensure_ascii=False)
    return len(node2id),word_size,np.array(emd_matrix)

class AllModel:
    def __init__(self,args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.ugc_info = args.ugc_info
        self.global_info = args.global_info
        self.pair_info = args.pair_info
        self.gate_mechanism = args.gate_mechanism
        self.ugc_attention_type = args.ugc_attention_type
        self.lock_layers = args.lock_layers
        self.gcn_type = args.gcn_type

        self.subj_info = args.subj_info
        self.obj_info = args.obj_info

        assert  self.subj_info or self.obj_info

        assert self.ugc_info != False or self.global_info != False

        if self.ugc_info and self.global_info:
            if self.ugc_attention_type == 0:
                if self.gate_mechanism:
                    self.mode = "pair_ugcGCNGate" if self.pair_info else "ugcGCNGate"
                else:
                    self.mode = "pair_ugcGCN" if self.pair_info else "ugcGCN"
            elif self.ugc_attention_type == 1:
                self.mode = "pair_SOAttenGCNGate" if self.gate_mechanism else "pair_SOAttenGCN"
            else:
                if self.gate_mechanism:
                    self.mode = "pair_ReAttenGCNGate" if self.pair_info else "ReAttenGCNGate"
                else:
                    self.mode = "pair_ReAttenGCN" if self.pair_info else "ReAttenGCN"         
            if self.gcn_type == "rgcn":
                if "GCN" in self.mode:
                    self.mode = self.mode.replace("GCN","RGCN")
                else:
                    self.mode = self.mode + "RGCN"
        elif self.ugc_info:
            if self.ugc_attention_type == 0:
                self.mode = "pair_UGC" if self.pair_info else "onlyUGC"
            elif self.ugc_attention_type == 1:
                 self.mode = "pair_SOAttenGate" if self.gate_mechanism else "pair_SOAtten"
            else:
                if self.gate_mechanism:
                    self.mode = "pair_ReAttenGate" if self.pair_info else "ReAttenGate"
                else:
                    self.mode = "pair_ReAtten" if self.pair_info else "ReAtten"
        else:
            if self.gate_mechanism:
                self.mode = "pair_GCNGate" if self.pair_info else "GCNGate"
            else:
                self.mode = "pair_GCN" if self.pair_info else "onlyGCN"
            if self.gcn_type == "rgcn":
                    self.mode = self.mode.replace("GCN","RGCN")
        
        if self.subj_info and not self.obj_info:
            self.mode += "Subj"
        elif self.obj_info and not self.subj_info:
            self.mode += "Obj"

        self.model = None
        self.rel2id = json.load(open(os.path.join(DATA_DIR,"relation.json"),"r",encoding="utf-8"))
        
        BERT_DIR = args.BERT_DIR
        self.config_path = os.path.join(BERT_DIR,"bert_config.json")
        self.checkpoint_path = os.path.join(BERT_DIR,"bert_model.ckpt")
        self.dict_path = os.path.join(BERT_DIR,"vocab.txt")
        self.tokenizer = init_tokenizer(self.dict_path)

    def build(self,compile=True,training=True):
        ## pair_info
        in_1 = keras.layers.Input(shape=(2,))
        ## ugc info
        in_2 = keras.layers.Input(shape=(None,))
        in_3 = keras.layers.Input(shape=(None,))
        in_4 = keras.layers.Input(shape=(None,))
        in_5 = keras.layers.Input(shape=(None,))
        in_6 = keras.layers.Input(shape=(None,))
        ## global gcn
        in_7 = keras.layers.Input(shape=(global_node_num,),dtype="int32")
        in_8 = keras.layers.Input(shape=(global_node_num,global_node_num),dtype="int32")
        in_9 = keras.layers.Input(shape=(global_node_num,global_node_num),dtype="int32")

        if self.pair_info:
            vocab_size,word_emd_size,emd_matrix = build_embedding_matrix("WVSG_word.model","word")
            WordEM = keras.layers.Embedding(input_dim=vocab_size,
                        output_dim=word_emd_size,
                        input_length=2,
                        weights=[emd_matrix],
                        trainable=False
            )
            WORD_RESHAPE = keras.layers.Reshape(target_shape=(200,))
            pair_x = WordEM(in_1)
            pair_x = WORD_RESHAPE(pair_x)
            
        if self.ugc_info:
            MLM = build_transformer_model(config_path=self.config_path,checkpoint_path=self.checkpoint_path)
            for layer in MLM.layers:
                for lock_layer in self.lock_layers:
                    lock_layer = "-"+lock_layer+"-"
                    if layer.name.find(lock_layer) != -1:
                        layer.trainable = False
                        break
            TEXT_SLICE = Slice(bag_num,max_text)
            IDX_SLICE = Slice(bag_num,1)
            MEAN = keras.layers.Lambda(lambda x: K.mean(x,axis=1))
            SUM = keras.layers.Lambda(lambda x: K.sum(x,axis=1))
            GATHER = keras.layers.Lambda(lambda x:seq_gather(x))
            GATHER_ONE = keras.layers.Lambda(lambda x:seq_gather_one(x))

            in_2_slice = TEXT_SLICE(in_2)
            in_3_slice = TEXT_SLICE(in_3)
            in_4_slice  = IDX_SLICE(in_4)
            in_5_slice  = IDX_SLICE(in_5)
            mlm_out_1 = []
            for i in range(bag_num):
                ## e1 feature and e2 feature
                out_1 = MLM([in_2_slice[i],in_3_slice[i]])
                if self.subj_info and self.obj_info:
                    out_1 = GATHER([out_1,in_4_slice[i],in_5_slice[i]])
                elif self.subj_info:
                    out_1 = GATHER_ONE([out_1,in_4_slice[i]])
                else:
                    out_1 = GATHER_ONE([out_1,in_5_slice[i]])
                mlm_out_1.append(out_1)
            ugc_x = keras.layers.Concatenate(axis=-1)(mlm_out_1)
            ugc_x = keras.layers.Reshape(target_shape=(bag_num,-1))(ugc_x)
            if self.ugc_attention_type == 2:
                if self.subj_info and self.obj_info:
                    RE_ATTEN = RelAttention(len(self.rel2id),training=training,bias=False)
                else:
                    RE_ATTEN = RelAttention(len(self.rel2id),training=training,bias=False)
                ugc_x = RE_ATTEN([ugc_x,in_6])
            elif self.ugc_attention_type == 1:
                SO_ATTEN = SOAttention()
                ugc_x = SO_ATTEN([ugc_x,pair_x])
                ugc_x = SUM(ugc_x)
            else:
                ugc_x = MEAN(ugc_x)
        
        if self.global_info:
            ## global gcn model
            EM = keras.layers.Embedding(input_dim=global_node_count,output_dim=global_emd_size,input_length=global_node_num)
            embed = EM(in_7)
            if self.gcn_type == "gcn":
                GCN = GraphConv(units=global_gcn_size,step_num=1,activation="relu",kernel_initializer="glorot_uniform")
                global_gcn_out = GCN([embed,in_8])
            else:
                RGCN = RGCNLayer(in_feat=global_emd_size,out_feat=global_gcn_size,num_rels=2,num_bases=2,bias=False,activation="relu",adj_node=global_node_num)
                global_gcn_out = RGCN([embed,in_8,in_9])
            global_s_hid = keras.layers.Lambda(lambda x:x[:,0])(global_gcn_out)
            global_o_hid = keras.layers.Lambda(lambda x:x[:,1])(global_gcn_out)
            if self.subj_info and self.obj_info:
                global_x = keras.layers.Concatenate(axis=-1)([global_s_hid,global_o_hid])
            elif self.subj_info:
                global_x = global_s_hid
            else:
                global_x = global_o_hid

        D2 = keras.layers.Dense(len(self.rel2id),activation="softmax")
        if self.ugc_info and self.global_info:
            D_UGC = keras.layers.Dense(100,activation="selu") 
            ugc_x = D_UGC(ugc_x)
            ## add dense for global
            D_Global = keras.layers.Dense(100,activation="selu")
            global_x = D_Global(global_x)
            if self.pair_info:
                D3 = keras.layers.Dense(100,activation="selu")
                pair_x = D3(pair_x)
                x = keras.layers.Concatenate(axis=-1)([pair_x,ugc_x,global_x])
            else:
                x = keras.layers.Concatenate(axis=-1)([ugc_x,global_x])
            # After Dense
            if self.gate_mechanism:
                Gate = GateMask(100+100+100)
                x = Gate(x)
            else:
                x = x
            outputs = D2(x)
            if self.pair_info:
                if self.ugc_attention_type == 2:
                    if self.gcn_type == "gcn":
                        model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_6,in_7,in_8], outputs=outputs)
                    else:
                        model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_6,in_7,in_8,in_9], outputs=outputs)
                else:
                    if self.gcn_type == "gcn":
                        model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_7,in_8], outputs=outputs)
                    else:
                        model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_7,in_8,in_9], outputs=outputs)
            else:
                if self.ugc_attention_type == 2:
                    if self.gcn_type == "gcn":
                        model = keras.models.Model(inputs=[in_2,in_3,in_4,in_5,in_6,in_7,in_8], outputs=outputs)
                    else:
                        model = keras.models.Model(inputs=[in_2,in_3,in_4,in_5,in_6,in_7,in_8,in_9], outputs=outputs)
                else:
                    if self.gcn_type == "gcn":
                        model = keras.models.Model(inputs=[in_2,in_3,in_4,in_5,in_7,in_8], outputs=outputs)
                    else:
                        model = keras.models.Model(inputs=[in_2,in_3,in_4,in_5,in_7,in_8,in_9], outputs=outputs)
        elif self.ugc_info:
            D1 = keras.layers.Dense(128,activation="selu")
            ugc_x = D1(ugc_x)
            if self.pair_info:
                D3 = keras.layers.Dense(128,activation="selu")
                pair_x = D3(pair_x)
                x = keras.layers.Concatenate(axis=-1)([pair_x,ugc_x])
            else:
                x = ugc_x
            ## gate after dense
            if self.gate_mechanism:
                if self.pair_info:
                    Gate = GateMask(128 + 128)
                else:
                    Gate = GateMask(128)
                x = Gate(x)
            outputs = D2(x)
            if self.pair_info:
                if self.ugc_attention_type == 2:
                    model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_6], outputs=outputs)
                else:
                    model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5], outputs=outputs)
            else:
                if self.ugc_attention_type == 2:
                    model = keras.models.Model(inputs=[in_2,in_3,in_4,in_5,in_6], outputs=outputs)
                else:
                    model = keras.models.Model(inputs=[in_2,in_3,in_4,in_5], outputs=outputs)
        elif self.global_info:
            D1 = keras.layers.Dense(64,activation="relu")
            global_x = D1(global_x)
            if self.pair_info:
                D3 = keras.layers.Dense(100,activation="selu")
                pair_x = D3(pair_x)
                x = keras.layers.Concatenate(axis=-1)([pair_x,global_x])
            else:
                x = global_x
            ## gate after dense
            if self.gate_mechanism:
                if self.pair_info:
                    Gate = GateMask(64 + 100)
                else:
                    Gate = GateMask(64)
                x = Gate(x)
            outputs = D2(x)
            if self.pair_info:
                if self.gcn_type == "gcn":
                    model = keras.models.Model(inputs=[in_1,in_7,in_8], outputs=outputs)
                else:
                    model = keras.models.Model(inputs=[in_1,in_7,in_8,in_9], outputs=outputs)
            else:
                if self.gcn_type == "gcn":
                    model = keras.models.Model(inputs=[in_7,in_8], outputs=outputs)
                else:
                    model = keras.models.Model(inputs=[in_7,in_8,in_9], outputs=outputs)

        self.model = model
        self.model.summary()
        if compile:
            self.model.compile(loss="categorical_crossentropy",optimizer=Adam(self.lr if self.lr else 5e-5),metrics=["accuracy"])

    def train(self,train_data,valid_data,plot=True,mfn=None):
        if not self.model:
            self.build(compile=True,training=False)
        if mfn:
            self.model.load_weights(mfn)
        save_best = keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR,"model_{}.h5".format(self.mode)),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            period=1
        )
        callbacks = [save_best]
        train_D = DG(train_data,self.batch_size,self.tokenizer,pair_info=self.pair_info,\
            global_info=self.global_info,ugc_info=self.ugc_info,gcn_type=self.gcn_type,atten_type=self.ugc_attention_type)
        valid_D = DG(valid_data,self.batch_size,self.tokenizer,pair_info=self.pair_info,\
            global_info=self.global_info,ugc_info=self.ugc_info,gcn_type=self.gcn_type,atten_type=self.ugc_attention_type)
        history = self.model.fit_generator(
            train_D.forfit(),
            steps_per_epoch=len(train_D),
            epochs=self.epochs,
            validation_data=valid_D.forfit(),
            validation_steps=len(valid_D),
            callbacks=callbacks,
        )
        if plot:
            self.plot_results(history)
    
    def plot_results(self,history,acc=True,loss=True):
        print(history.history.keys())
        dn = os.path.join(DATA_DIR,"imgs")
        if not os.path.exists(dn):
            os.makedirs(dn)
        if acc:
            plt.plot(history.history["accuracy"])
            plt.plot(history.history["val_accuracy"])
            plt.legend(["train","valid"],loc="upper left")
            plt.title("Model Accuracy");plt.xlabel("epoch");plt.ylabel("accuracy")
            plt.savefig(os.path.join(DATA_DIR,"imgs","{}_acc.pdf".format(self.mode)))
            plt.close()
        if loss:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.legend(["train","valid"],loc="upper left")
            plt.title("Model Loss");plt.xlabel("epoch");plt.ylabel("loss")
            plt.savefig(os.path.join(DATA_DIR,"imgs","{}_loss.pdf".format(self.mode)))
            plt.close()

    def predict(self,in_fn,out_fn=None,mfn=None):
        mfn = mfn if mfn else os.path.join(MODEL_DIR,"model_{}.h5".format(self.mode))
        out_dir = os.path.join(os.path.dirname(in_fn),"results")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = out_fn if out_fn else os.path.join(os.path.dirname(in_fn),\
            "results/valid_{}_res.json".format(self.mode))
        t = load_data(in_fn,shuffle=False)
        data_D = DG(t,self.batch_size,self.tokenizer,pair_info=self.pair_info,\
            global_info=self.global_info,ugc_info=self.ugc_info,gcn_type=self.gcn_type,atten_type=self.ugc_attention_type)
        ## rebuild model
        self.build(compile=False,training=False)
        self.model.load_weights(os.path.join(MODEL_DIR,mfn))
        pred_res = self.model.predict_generator(data_D.__iter__(random=False),steps=len(data_D))
        print(pred_res.shape)
        pred_res = np.argmax(pred_res,axis=-1)
        print(pred_res.shape)
        id2rel = {}
        for rel in self.rel2id:
            id2rel[self.rel2id[rel]] = rel

        with open(in_fn,"r",encoding="utf-8") as rf,\
            open(out_fn,"w",encoding="utf-8") as wf:
            for i,line in enumerate(rf):
                data = json.loads(line)
                data["pred_predicate"] = id2rel[pred_res[i]]
                wf.write(json.dumps(data,ensure_ascii=False) + "\n")

    def evaluate(self,in_fn,show_error=False):
        res = []
        y_true = []
        y_pred = []
        with open(in_fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                y_true.append(self.rel2id[d["predicate"]])
                y_pred.append(self.rel2id[d["pred_predicate"]])
                if d["predicate"] != d["pred_predicate"]:
                    res.append(d)
        plot_confusion(y_true,y_pred)
        print("Valid results, mode: {}".format(self.mode))
        cal_metrics(y_true,y_pred)

        if show_error:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(in_fn)),"errors")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_fn = os.path.join(os.path.dirname(os.path.dirname(in_fn)),"errors/valid_{}_error.json".format(self.mode))
            with open(out_fn,"w",encoding="utf-8") as wf:
                for x in res:
                    wf.write(json.dumps(x,ensure_ascii=False)+"\n")

def main():
    parser = argparse.ArgumentParser()
    ## required parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="training data for model."
    )
    parser.add_argument(
        "--valid_file",
        default=None,
        type=str,
        required=True,
        help="valid data for evaluation model."
    )

    ## other parameters
    parser.add_argument(
        "--result_file",
        default=None,
        type=str,
        help="result data predicted by model."
    )
    parser.add_argument(
        "--BERT_DIR",
        default="./models/bert/",
        type=str,
        help="Path to pretrained bert."
    )
    parser.add_argument(
        "--pretrained_weights",
        default=None,
        type=str,
        help="pretrained model weights."
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="training epochs."
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="training batch_size."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="learining rate."
    )
    parser.add_argument(
        "--cuda_device",
        default="0",
        type=str,
        help="Train model on which GPU device."
    )
    parser.add_argument(
        "--gcn_type",
        default="gcn",
        type=str,
        help="GCN or RGCN"
    )
    parser.add_argument(
        "--lock_layers",
        nargs="*",
        help="Lock layers for bert."
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="train model or just predict and evaluate."
    )
    parser.add_argument(
        "--pair_info",
        action="store_true",
        help="whether to use pair info for ugc attention."
    )
    parser.add_argument(
        "--ugc_info",
        action="store_true",
        help="whether to use ugc information."
    )
    parser.add_argument(
        "--global_info",
        action="store_true",
        help="whether to use global GCN."
    )
    parser.add_argument(
        "--gate_mechanism",
        action="store_true",
        help="whether to use gate mechanism."
    )
    parser.add_argument(
        "--ugc_attention_type",
        default=0,
        type=int,
        help="UGC attention type: -1,no attention(average); 0: base attention; 1: SO attention"
    )
    parser.add_argument(
        "--subj_info",
        action="store_true",
        help="whether to use head information." 
    )
    parser.add_argument(
        "--obj_info",
        action="store_true",
        help="whether to use modifier information."
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    REmodel = AllModel(args)
    train_data = load_data(args.train_file)
    valid_data = load_data(args.valid_file)
    res_fn = args.result_file if args.result_file else os.path.join(os.path.dirname(args.valid_file),"results/valid_{}_res.json".format(REmodel.mode))
    if args.training:
        mfn = args.pretrained_weights
        REmodel.train(train_data,valid_data,plot=True,mfn=mfn)
    else:
        REmodel.build(compile=False,training=False)
        REmodel.model.load_weights(os.path.join(MODEL_DIR,"model_{}.h5".format(REmodel.mode)))
    REmodel.predict(args.valid_file,res_fn)
    REmodel.evaluate(res_fn,show_error=False)

if __name__ == '__main__':
    main()