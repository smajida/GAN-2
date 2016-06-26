import tensorflow as tf

"""
Simple fully connected encoder/decoder
"""

class SingleFFLayer():
    def __init__(self,shape,act_fn,dropout,keep_prob):
        self.shape = shape
        self.act_fn = act_fn
     
        self.weights = tf.Variable(tf.truncated_normal(self.shape,mean = 0,stddev=0.001))
        self.biases = tf.Variable(tf.zeros([shape[-1]]))
        
    def __call__(self,x):
        return self.act_fn(tf.matmul(x,self.weights) + self.biases)
