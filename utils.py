#_*_coding:utf-8_*_
import os
import re
import json
import pymongo
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from ltp import LTP
from time import sleep
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

DATA_DIR = os.path.join(os.getcwd(),"data")

def get_yearlypv(fn=None):
    """pv value of query every year
    """
    fn = fn if fn else os.path.join(DATA_DIR,"meituan_info/concept_info.txt")
    res = dict()
    with open(fn,"r",encoding="utf-8") as rf:
        for i,line in enumerate(rf):
            if i == 0:continue
            content = line.strip().split("\t")
            phrase,pv = content[:2]
            pv = float(pv)
            res[phrase] = pv
    return  res

def get_ent_type(fn=None):
    fn = fn if fn else os.path.join(DATA_DIR,"meituan_info/taxonomy.data")
    type_dict = {}
    with open(fn,"r",encoding="utf-8") as rf:
        for line in rf:
            t,c,_,score = line.strip().split("\t")
            if not t in type_dict:
                type_dict[t] = (c,score)
            else:
                if score > type_dict[t][1]:
                    type_dict[t] = (c,score)
    for k in type_dict:
        type_dict[k] = type_dict[k][0]
    return type_dict

def get_multi_type(fn=None):
    fn = fn if fn else os.path.join(DATA_DIR,"meituan_info/taxonomy.data")
    type_dict = {}
    with open(fn,"r",encoding="utf-8") as rf:
        for line in rf:
            t,c,_,_ = line.strip().split("\t")
            if not t in type_dict:
                type_dict[t] = [c]
            else:
                type_dict[t].append(c)
    return type_dict

def find_chinese(str):
    """remove no chinese char in predicate
    """
    pattern = re.compile(r"[^\u4e00-\u9fa5]")
    chinese = re.sub(pattern,"",str)
    return chinese

def init_mongo(conf_fn=None):
    """init DBpedia Mongo DB for searching predicates
    """
    conf_fn = conf_fn if conf_fn else os.path.join(DATA_DIR,"config/mongo_conf.json")
    conf = json.load(open(conf_fn,"r",encoding="utf-8"))
    client = pymongo.MongoClient(conf.get("host","localhost"),conf.get("port",27017))
    client.admin.authenticate(conf.get("user",""),conf.get("passwd",""),mechanism="SCRAM-SHA-1")
    db = client.get_database(conf.get("db",""))
    col = db[conf.get("col","")]
    # col.create_index([("o",1)])
    return col

def get_char_distance(s,o,text):
    """char distance
    """
    start = 0
    s_index = []
    while True:
        try:
            idx = text.index(s,start)
            s_index.append(idx)
            start = idx + len(s)
        except:
            break
    start = 0
    o_index = []
    while True:
        try:
            idx = text.index(o,start)
            o_index.append(idx)
            start = idx + len(o)
        except:
            break
    min_dis = len(text)
    for sidx in s_index:
        for oidx in o_index:
            if sidx < oidx:
                le = len(s)
            else:
                le = len(o)
            l = abs(sidx-oidx) - le
            if l < min_dis:
                min_dis = l
    return min_dis

def get_dep_distance(s,o,text,nlp_model=None):
    """dependency distance
    """
    if not nlp_model:
        nlp_model = LTP("tiny")
        nlp_model.add_words(words=[s,o])
    seg,hidden = nlp_model.seg([text])

    s_index = []
    o_index = []
    for i,w in enumerate(seg[0]):
        if w == s:
            s_index.append(i+1)
        if w == o:
            o_index.append(i+1)
    if s_index == [] or o_index == []:
        return None

    dep = nlp_model.dep(hidden)
    edges = []
    for x in dep[0]:
        edges.append(x[:2])
    graph = nx.Graph(edges)

    min_dis = len(seg[0])
    for sidx in s_index:
        for oidx in o_index:
            l = nx.shortest_path_length(graph,source=sidx,target=oidx)
            min_dis = min(min_dis,l)
    return min_dis

def get_dep_path(s,o,text,nlp_model=None):
    """dependency path
    """
    if not nlp_model:
        nlp_model = LTP("tiny")
        nlp_model.add_words(words=[s,o])
    seg,hidden = nlp_model.seg([text])
    s_index = []
    o_index = []
    for i,w in enumerate(seg[0]):
        if w == s:
            s_index.append(i+1)
        if w == o:
            o_index.append(i+1)
    if s_index == [] or o_index == []:
        return None
    
    dep = nlp_model.dep(hidden)
    edges = []
    for x in dep[0]:
        edges.append(x[:2])
    graph = nx.Graph(edges)
    min_dis = len(seg[0])
    path = []
    for sidx in s_index:
        for oidx in o_index:
            p = nx.shortest_path(graph,source=sidx,target=oidx)
            if len(p) - 1 < min_dis:
                min_dis = len(p) - 1
                path = p
            # min_dis = min(min_dis,l-1)

    res = []
    for i,p in enumerate(path):
        if i < len(path)-1 and dep[0][p-1][1] == path[i+1]:
            xx = seg[0][p-1],dep[0][p-1][-1],seg[0][path[i+1]-1]
            for x in xx:
                if res == [] or x != res[-1]:
                    res.append(x)
        if i > 0 and dep[0][p-1][1] == path[i-1]:
            xx = seg[0][path[i-1]-1],dep[0][p-1][-1],seg[0][p-1]
            for x in xx:
                if res == [] or x != res[-1]:
                    res.append(x)
    return res
       
def add_special_tokens(s,o,text,mask=False):
    start = 0
    s_index = []
    while True:
        try:
            idx = text.index(s,start)
            s_index.append(idx)
            start = idx + len(s)
        except:
            break
    start = 0
    o_index = []
    while True:
        try:
            idx = text.index(o,start)
            o_index.append(idx)
            start = idx + len(o)
        except:
            break
    min_dis = len(text)
    min_s_ind = -1
    min_o_ind = -1
    for sidx in s_index:
        for oidx in o_index:
            if sidx < oidx:
                le = len(s)
            else:
                le = len(o)
            l = abs(sidx-oidx) - le
            if l < min_dis:
                min_dis = l
                min_s_ind = sidx
                min_o_ind = oidx
    if min_s_ind < min_o_ind:
        if not mask:
            text = text[:min_s_ind] + "[e1]" + text[min_s_ind:min_s_ind+len(s)] + "[/e1]" + \
                text[min_s_ind+len(s):min_o_ind]+"[e2]"+text[min_o_ind:min_o_ind+len(o)] + "[/e2]" + text[min_o_ind+len(o):]
        else:
            text = text[:min_s_ind] + "[e1][blank][/e1]" + \
                text[min_s_ind+len(s):min_o_ind]+"[e2][blank][/e2]" + text[min_o_ind+len(o):]
    else:
        if not mask:
            text = text[:min_o_ind] + "[e2]" + text[min_o_ind:min_o_ind+len(o)] + "[/e2]" + \
                text[min_o_ind+len(o):min_s_ind] + "[e1]" + text[min_s_ind:min_s_ind+len(s)] + "[/e1]" + text[min_s_ind+len(s):]
        else:
            text = text[:min_o_ind] + "[e2][blank][/e2]" + \
                text[min_o_ind+len(o):min_s_ind] + "[e1][blank][/e1]" + text[min_s_ind+len(s):]
    return text

def get_dep_info(s,o,text,nlp_model=None):
    """get base dep info
    """
    if not nlp_model:
        nlp_model = LTP("tiny")
        nlp_model.add_words([s,o])
    seg,hidden = nlp_model.seg([text])
    dep = nlp_model.dep(hidden)
    return seg[0],dep[0]

def plot_confusion(y_true,y_pred,index=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    index = index if index else [x for x in range(21)]
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=index, columns=index)
    fig, ax = plt.subplots(figsize = (14.5,12.5))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
    plt.ylabel("True label", fontsize=18)
    plt.xlabel("Predicted label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig("confusion.pdf", bbox_inches="tight")

def cal_metrics(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("------Accuracy------")
    print("Accuracy:", round(accuracy_score(y_true, y_pred),4))
    print("------Weighted------")
    print("Precision:", round(precision_score(y_true, y_pred, average="weighted"),4))
    print("Recall:", round(recall_score(y_true, y_pred, average="weighted"),4))
    print("F1-score:", round(f1_score(y_true, y_pred, average="weighted"),4))
    print("------Macro------")
    print("Precision:", round(precision_score(y_true, y_pred, average="macro"),4))
    print("Recall:", round(recall_score(y_true, y_pred, average="macro"),4))
    print("F1-score:", round(f1_score(y_true, y_pred, average="macro"),4))
    print("------Micro------")
    print("Precision:", round(precision_score(y_true, y_pred, average="micro"),4))
    print("Recall:", round(recall_score(y_true, y_pred, average="micro"),4))
    print("F1-score:", round(f1_score(y_true, y_pred, average="micro"),4))

def clean_zh_meaning(s):
    r = re.sub("\[.*\]","",s)
    r = re.sub("（.*）","",r)
    r = re.sub("\(.*\)","",r)
    r = re.sub("[ ]{2,}"," ",r).strip()
    return r

def kappa(data):
    data = np.array(data)
    total = np.sum(data)
    P0 = np.trace(data) / total
    xsum = np.sum(data,axis=1)
    ysum = np.sum(data,axis=0)
    Pe = np.dot(xsum,ysum)/ total ** 2
    res = (P0-Pe)/(1-Pe)
    return res

def fleiss_kappa(data,n):
    data = np.array(data)
    N = data.shape[0]
    k = data.shape[1]

    dataSquare = np.power(data,2)
    xsum = np.sum(dataSquare,axis=1)
    xsum = (xsum - n)/(n*(n-1))
    P0 = np.sum(xsum)/N

    total = np.sum(data)
    ysum = np.sum(data,axis=0)
    ysum = ysum / total
    Pe = np.sum(np.power(ysum,2))
    res = (P0-Pe)/(1-Pe)
    return res

def rouge_score(references,candidate):
    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    rouge = Rouge()
    for ref in references:
        score = rouge.get_scores(hyps=candidate,refs=ref)
        rouge_1.append(score[0]["rouge-1"]["f"])
        rouge_2.append(score[0]["rouge-2"]["f"])
        rouge_l.append(score[0]["rouge-l"]["f"])
    return max(rouge_1),max(rouge_2),max(rouge_l)

def bleu_score(references,candidate):
    scores = []
    for ref in references:
        score  = sentence_bleu([ref.split(" ")],candidate.split(" "),smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return max(scores)

