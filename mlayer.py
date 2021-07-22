#_*_coding:utf-8_*_
import keras
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from bert4keras.layers import Loss

def seq_gather(inputs):
    x,head,tail = inputs
    head = K.cast(head,"int32")
    tail = K.cast(tail,"int32")
    batch_idx = K.arange(0,K.shape(x)[0])
    batch_idx = K.expand_dims(batch_idx,1)
    head_new = K.concatenate([batch_idx,head],axis=-1)
    tail_new = K.concatenate([batch_idx,tail],axis=-1)
    head_f = tf.gather_nd(x,head_new)
    tail_f = tf.gather_nd(x,tail_new)
    outputs = K.concatenate([head_f,tail_f],axis=-1)
    return outputs

def seq_gather_one(inputs):
    x,idx = inputs
    idx = K.cast(idx,"int32")
    batch_idx = K.arange(0,K.shape(x)[0])
    batch_idx = K.expand_dims(batch_idx,1)
    res_new = K.concatenate([batch_idx,idx],axis=-1)
    res_f = tf.gather_nd(x,res_new)
    return res_f

def info_attention(inputs,f_cnt,dim):
    a = keras.layers.Dense(f_cnt*dim,activation="softmax")(inputs)
    a = keras.layers.Reshape((f_cnt,dim))(a)
    a = keras.layers.Lambda(lambda x: K.sum(x,axis=2))(a)
    a = keras.layers.RepeatVector(dim)(a)
    a_probs = keras.layers.Permute((2,1),name="info_atten_vec")(a)
    a_probs = keras.layers.Flatten()(a_probs)
    outputs = keras.layers.Multiply()([inputs,a_probs])
    return outputs

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

class SOAttention(keras.layers.Layer):
    def __init__(self,bias=False,**kwargs):
        super(SOAttention,self).__init__(**kwargs)
        self.bias = bias

    def build(self,input_shape):
        super(SOAttention,self).build(input_shape)
        self.W = self.add_weight("{}_W".format(self.name),
                                shape=(input_shape[0][-1],input_shape[1][-1]),
                                initializer="glorot_uniform",
                                trainable=True
                            )
        if self.bias:
            self.b = self.add_weight("{}_b".format(self.name),
                                shape=(input_shape[0][1],),
                                initializer="glorot_uniform",
                                trainable=True
                            )
                
    def call(self,inputs):
        q,k = inputs
        x = K.dot(q,self.W)
        x = K.batch_dot(x,k)
        if self.bias:
            x += self.b
        x = K.softmax(K.cast(x,"float32"),axis=-1)
        # zeros = K.zeros_like(x)
        # x = tf.where(x > 0.2, x, zeros)
        probs = K.permute_dimensions(K.repeat(x,K.shape(q)[-1]),(0,2,1))
        outputs = tf.multiply(q,probs)
        outputs = K.sum(outputs,axis=1)
        return outputs
        
    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[0][2]])

class RelAttention(keras.layers.Layer):
    def __init__(self,input_dim,rel_tot,training=True,bias=False,**kwargs):
        # self.dim = input_dim
        self.rel_tot = rel_tot
        self.training = training
        self.bias = bias
        super(RelAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        super(RelAttention,self).build(input_shape)
        self.W = self.add_weight("{}_W".format(self.name),
                                shape=(input_shape[0][-1],),
                                initializer="glorot_uniform",
                                trainable=True
                            )

        self.relMat = self.add_weight("{}_relMat".format(self.name),
                                shape=(self.rel_tot,input_shape[0][-1]),
                                initializer="glorot_uniform",
                                trainable=True
                            )
        if self.bias:
            self.b = self.add_weight("{}_b".format(self.name),
                                shape=(input_shape[0][-1],),
                                initializer="glorot_uniform",
                                trainable=True
                            )
                
    def call(self,inputs):
        query,key = inputs
        W = tf.diag(self.W)
        if self.training:
            key = K.cast(key,"int32")
            key = tf.nn.embedding_lookup(self.relMat,key)
            key = K.squeeze(key,axis=1)
            x = K.dot(query,W)
            x = K.batch_dot(x,key)
            if self.bias:
                x += self.b
            x = K.softmax(K.cast(x,"float32"),axis=-1)
            probs = K.permute_dimensions(K.repeat(x,K.shape(query)[-1]),(0,2,1))
            outputs = tf.multiply(query,probs)
            outputs = K.sum(outputs,axis=1)
        else:
            x = K.dot(query,W)
            x = K.dot(x,K.transpose(self.relMat)) # B * n * r
            attention_score = K.softmax(K.cast(K.permute_dimensions(x,(0,2,1)),K.floatx()),axis=-1) # B * r * n
            attention_rep = K.batch_dot(attention_score,query) # B * r * d
            rel_logit = K.softmax(K.dot(attention_rep,K.transpose(self.relMat)),axis=-1) # B * r * r
            rel_logit = K.map_fn(tf.diag_part,rel_logit) # B * r
            rel_tar = K.reshape(K.cast(K.argmax(rel_logit),"int32"),(-1,1)) # B * 1
            batch_idx = K.arange(0,K.shape(rel_tar)[0])
            batch_idx = K.reshape(batch_idx,(-1,1))
            idx = K.concatenate([batch_idx,rel_tar],axis=-1) # B * 2
            outputs = tf.gather_nd(attention_rep,idx) # B * d
        return outputs
        
    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[0][2]])

class Slice(keras.layers.Layer):
    def __init__(self,bag_num,dim,**kwargs):
        self.dim = dim
        self.bag_num = bag_num
        super(Slice,self).__init__(**kwargs)

    def slice(self,x,index):
        return K.slice(x,[0,index * self.dim],[-1,self.dim])

    def call(self,inputs):
        outputs = []
        for i in range(self.bag_num):
            output = self.slice(inputs,i)
            outputs.append(output)
        return outputs

    def compute_output_shape(self,input_shape):
        return [tuple([None,self.dim])] * self.bag_num

class GateMask(keras.layers.Layer):
    def __init__(self,dim,**kwargs):
        self.dim = dim
        # self.activation = keras.activations.get(activation,None)
        super(GateMask,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W = self.add_weight(name="{}_W".format(self.name),
                                shape=(self.dim,self.dim),
                                dtype="float32",
                                initializer="glorot_uniform",
                                trainable=True
                            )
        super(GateMask,self).build(input_shape)

    def call(self,inputs):
        x = K.dot(inputs,self.W)
        g = K.sigmoid(x)
        outputs = tf.multiply(inputs,g)
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[1]])

class RGCNLayer(keras.layers.Layer):
    def __init__(self,in_feat,out_feat,num_rels,num_bases=-1,bias=None,activation=None,adj_node=None,**kwargs):
        super(RGCNLayer,self).__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.activation = keras.activations.get(activation)
        self.bias = bias
        self.adj_node = adj_node
        # self.is_input_layer = is_input_layer

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        

    def build(self,input_shape):
        self.weight = self.add_weight("{}_W".format(self.name),
                                    shape=(self.num_bases,self.in_feat,self.out_feat),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        self.Wo = self.add_weight("{}_W0".format(self.name),
                                    shape=(self.in_feat,self.out_feat),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        if self.num_bases < self.num_rels:
            self.w_cmp = self.add_weight("{}_cmp".format(self.name),
                                    shape=(self.num_rels,self.num_bases),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        if self.bias:
            self.b = self.add_weight("{}_bias".format(self.name),
                                    shape=(self.out_feat,),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        super(RGCNLayer,self).build(input_shape)

    def call(self,inputs):
        if self.num_bases < self.num_rels:
            self.weight = K.reshape(self.weight,(self.num_bases,self.in_feat,self.out_feat))
            self.weight = K.permute_dimensions(self.weight,(1,0,2))
            weight = K.dot(self.w_cmp,self.weight)
            weight = K.reshape(weight,(self.num_rels,self.in_feat,self.out_feat))
        else:
            weight = self.weight
        
        features = inputs[0]
        A = inputs[1:]
        xs = []
        for i in range(self.num_rels):
            edges = A[i]
            edges = K.cast(edges,K.floatx())
            x = K.dot(features,weight[i])
            x = K.batch_dot(K.permute_dimensions(edges,(0,2,1)),x) / (K.sum(edges,axis=2,keepdims=True) + K.epsilon())
            xs.append(x)
        # eyes_matrix = K.eye(self.adj_node)
        # print(K.int_shape(inputs[0]),type(K.int_shape(inputs[0])))
        # eyes_matrix = K.repeat_elements(eyes_matrix,K.int_shape(inputs[0])[0],axis=0)
        # xs.append(K.batch_dot(eyes_matrix,K.dot(features,self.Wo)))
        xs.append(K.dot(features,self.Wo))
        outputs = K.concatenate(xs,axis=-1)
        outputs = K.reshape(outputs,(-1,self.adj_node,self.out_feat,self.num_rels+1))
        outputs = K.sum(outputs,axis=-1)
        if self.bias:
            outputs += self.b
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[0][0],self.out_feat])
     