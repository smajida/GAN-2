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