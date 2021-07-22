#_*_coding:utf-8_*_
import os
import json
import random

from ltp import LTP
from tqdm import tqdm
from utils import *

DATA_DIR = os.path.join(os.getcwd(),"data")

def build_train_test_data(fn=None):
    per = 0.3
    ugc_text_num = 10
    # triple_num = 5
    fn = fn if fn else os.path.join(DATA_DIR,"human_label.json")
    with open(fn,"r",encoding="utf-8") as rf:
        data = json.load(rf)
    count_all = 0
    counts = []
    for r in data:
        print(r,len(data[r]))
        count_all += len(data[r])
        counts.append(len(data[r]))
    ugc_dict = json.load(open(os.path.join(DATA_DIR,"ugc/ugc_pair_dep.json"),"r",encoding="utf-8"))
    def get_info(subj,obj):
        k = subj+"##"+obj
        if not k in ugc_dict:
            return None
        texts = ugc_dict[k]
        random.shuffle(texts)
        texts = texts[:ugc_text_num]
        x = {
            "subject":subj,
            "object":obj,
            "texts":texts,
        }
        return x

    if not os.path.exists(os.path.join(DATA_DIR,"training")):
        os.makedirs(os.path.join(DATA_DIR,"training"))
    with open(os.path.join(DATA_DIR,"training/train.json"),"w",encoding="utf-8") as wf_train,\
        open(os.path.join(DATA_DIR,"training/valid.json"),"w",encoding="utf-8") as wf_valid:
        for r in data:
            samples = data[r]
            random.shuffle(samples)
            for sample in samples[:int(len(samples) * per)]:
                s,o = sample
                info = get_info(s,o)
                if info:
                    info["predicate"] = r
                    wf_valid.write(json.dumps(info,ensure_ascii=False)+"\n")
            for sample in samples[int(len(samples) * per):]:
                s,o = sample
                info = get_info(s,o)
                if info:
                    info["predicate"] = r
                    wf_train.write(json.dumps(info,ensure_ascii=False)+"\n") 

def mask_strategy(lab_dir=None):
    lab_dir = lab_dir if lab_dir else os.path.join(DATA_DIR,"labeling")
    count = {}
    for f in os.listdir(lab_dir):
        if not f.endswith(".txt"):
            continue
        fn = os.path.join(lab_dir,f)
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                s,_,o,r = line.strip().split("\t")
                if r == "-1":
                    continue
                if not o in count:
                    count[o] = {r:1}
                else:
                    count[o][r] = count[o].get(r,0)+1
    res = {}
    for k in count:
        if len(count[k]) > 1:
            res[k] = 1.-random.uniform(0.,0.2)
        else:
            res[k] = 0.+random.uniform(0.,0.01)
    return res

def shape_data(version,mode="train",mask=False,mask_rate=None,add_head=False,use_row=False):
    """
    version: format, mask: False, mask_rate: None, add_head: False, use_row: False,\n
    version: mask, mask_True, mask_rate: None or prob, add_head: False, use_row: False\n
    version: head, mask: False, mask_rate: None, add_head: True, use_row: False,\n
    version: raw, mask: False, mask_rate: None, add_head: False, use_row: True
    """
    version = str(version)
    dn = os.path.join(DATA_DIR,"training",version)
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = os.path.join(DATA_DIR,"training/{}_info.json".format(mode))
    max_text_len = 0
    if mask:
        if mask_rate:
            mode = mode+"_{}".format(str(mask_rate))
            mask_dict = None
        else:
            mode = mode+"_gate"
            mask_dict = mask_strategy()
    with open(fn,"r",encoding="utf-8") as rf,\
        open(os.path.join(DATA_DIR,"training/{}/{}_{}.json".format(version,mode,version)),"w",encoding="utf-8") as wf:
        for line in rf:
            d = json.loads(line)
            s = d["subject"]
            o = d["object"]
            new_texts = []
            if mask:
                if not mask_rate:
                    mask_rate = mask_dict[o]
                if random.random() < mask_rate:
                    mask_flag = True
                else:
                    mask_flag = False
            if not mask:
                mask_flag = False

            for text in d["texts"]:
                if use_row:
                    t = text
                else:
                    t = add_special_tokens(s,o,text,mask_flag)
                if add_head:
                    text = o+s + "：" + t
                else:
                    text = t
                new_texts.append(text)
                max_text_len = max(max_text_len,len(text))
            d["texts"] = new_texts
            wf.write(json.dumps(d,ensure_ascii=False)+"\n") 
        print("Max ugc text length: %d" % max_text_len)

def build_global_graph(with_type=True,with_rel=True,neighbor_num=30):
    if with_type:
        ent_type = get_multi_type()
        type_fn = os.path.join(DATA_DIR,"meituan_info/type2id.json")
        type_dict = json.load(open(type_fn,"r",encoding="utf-8"))
    node2id = {}
    node2id["PAD"] = len(node2id)
    node2id["NoneType"] = len(node2id)
    node2id["NoneNode"] = len(node2id)
    if with_type:
        for t in type_dict:
            node2id[t] = len(node2id)
    tmp = dict()
    count = 0
    valid_set = set()
    for mode in ["train","valid"]:
        with open("./data/training/format/{}_format.json".format(mode),"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                s,o,p = d["subject"],d["object"],d["predicate"]
                if mode == "valid":
                    valid_set.add(s)
                    valid_set.add(o)
                    if not s in node2id:
                        count += 1
                    if not o in node2id:
                        count += 1
                if not s in node2id:
                    node2id[s] = len(node2id)
                if not o in node2id:
                    node2id[o] = len(node2id)
                if with_rel:
                    if mode == "train" or mode == "valid":
                        if not s in tmp:
                            tmp[s] = {}
                        if not p in tmp[s]:
                            tmp[s][p] = [o]
                        else:
                            tmp[s][p].append(o)
                        
                        if not o in tmp:
                            tmp[o] = {}
                        rp = "r_" + p
                        if not rp in tmp[o]:
                            tmp[o][rp] = [s]
                        else:
                            tmp[o][rp].append(s)
                    else:
                        if not s in tmp:
                            tmp[s] = {}
                        if not o in tmp:
                            tmp[o] = {}
                else:
                    if not s in tmp:
                        tmp[s] = [o]
                    else:
                        tmp[s].append(o)
                    if not o in tmp:
                        tmp[o] = [s]
                    else:
                        tmp[o].append(s)           
    print(count)
    print(len(valid_set))
    if not with_rel:
        dn = os.path.join(DATA_DIR,"training/gcn_data")
        if not os.path.exists(dn):
            os.makedirs(dn)
        res = {}
        max_node_num = 0
        for k in tmp:
            if len(tmp[k]) > neighbor_num:
                random.shuffle(tmp[k])
            res[k] = list(ent_type.get(k,["NoneType"])) + tmp[k][:neighbor_num]
            max_node_num = max(max_node_num,len(res[k]))
        print(len(node2id))
        print(max_node_num)
        with open(os.path.join(dn,"node2id.json"),"w",encoding="utf-8") as wf:
            json.dump(node2id,wf,ensure_ascii=False)
        with open(os.path.join(dn,"global_graph.json"),"w",encoding="utf-8") as wf:
            json.dump(res,wf,ensure_ascii=False)

        # f_count = [len(tmp[x]) for x in tmp]
        # print(len(f_count))
        # print(len([x for x in f_count if x <= 30]))
        # plt.hist([x for x in f_count if x <= 30],bins="auto")
        # plt.savefig(dn + "/node_neighbors.png")

    else:
        dn = os.path.join(DATA_DIR,"training/rgcn_data")
        if not os.path.exists(dn):
            os.makedirs(dn)
        res = {}
        max_node_num = 0
        for k in tmp:
            cur_data = []
            for p in tmp[k]:
                cur_data += [(p,v) for v in tmp[k][p]]
            
            cur_types = ent_type.get(k,["NoneType"])
            cur_types = [("isA",t) for t in cur_types]
            random.shuffle(cur_data)
            res[k] = cur_data[:neighbor_num] + cur_types
            max_node_num = max(max_node_num,len(res[k]))
        print(len(node2id))
        print(max_node_num)
        with open(os.path.join(dn,"node2id.json"),"w",encoding="utf-8") as wf:
            json.dump(node2id,wf,ensure_ascii=False)
        with open(os.path.join(dn,"global_graph.json"),"w",encoding="utf-8") as wf:
            json.dump(res,wf,ensure_ascii=False)

        # r_count = {}
        # for t in tmp:
        #     for p in tmp[t]:
        #         if not p in r_count:
        #             r_count[p] = [len(tmp[t][p])]
        #         else:
        #             r_count[p].append(len(tmp[t][p]))
        # if not os.path.exists(dn+"/imgs"):
        #     os.makedirs(dn+"/imgs")
        # for p in r_count:
        #     plt.hist([x for x in r_count[p]],bins="auto")
        #     plt.savefig(dn + "/imgs/{}_node_neighbors.png".format(p))
        #     plt.close()

def shape_para_data():
    ugc_text_num = 10
    ugc_dict = json.load(open(os.path.join(DATA_DIR,"ugc/ugc_pair_dep.json"),"r",encoding="utf-8"))
    def get_info(subj,obj):
        k = subj+"##"+obj
        if not k in ugc_dict:
            return None
        texts = ugc_dict[k]
        random.shuffle(texts)
        texts = texts[:ugc_text_num]
        return texts
    
    for m in ["train","valid"]:
        fn = "./data/explanation/{}.json".format(m)
        wfn = "./data/explanation/{}_format.json".format(m)
        with open(fn,"r",encoding="utf-8") as rf,\
            open(wfn,"w",encoding="utf-8") as wf:
            for line in rf:
                d = json.loads(line)
                s,o = d["pair"].split("##")
                texts = get_info(s,o)
                if not texts:
                    continue
                new_texts = []
                for t in texts:
                    r = add_special_tokens(s,o,t,False)
                    new_texts.append(r)
                d["texts"] = new_texts
                wf.write(json.dumps(d,ensure_ascii=False)+"\n")
    
if __name__ == '__main__':
    build_train_test_data()
    shape_data(version="format",mode="train",mask=False,mask_rate=None,add_head=False,use_row=False)
    shape_data(version="format",mode="valid",mask=False,mask_rate=None,add_head=False,use_row=False)

    build_global_graph(with_rel=False)
    build_global_graph(with_rel=True)

    shape_para_data()

