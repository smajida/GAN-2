import tensorflow as tf
def SquaredError(x,y):
    with tf.name_scope('SquaredError'):
        error = tf.reduce_mean(tf.reduce_sum(x-y,reduction_indices=[1]))
        tf.scalar_summary('SquaredError',error)
        return error
        
def CrossEntropy(output,label):
    with tf.name_scope('CrossEntropy'):
        loss = -tf.reduce_mean(tf.reduce_sum(label * tf.log(output),reduction_indices = [1]))
        return loss
    
if __name__ == '__main__':
    import numpy as np
    x = np.random.random((50,10))
    y = np.random.random((50,10))
    
    with tf.Session() as sess:
        print(sess.run(SquaredError(x,y)))
        print(sess.run(CrossEntropy(x,y)))