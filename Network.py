import tensorflow as tf
from Tools import SquaredError,CrossEntropy

class Net(object):
    """
    A parent class for my generative and discriminative networks.
    Simply to save some lines of code. All the difinitions are a little tiresome
    and apply to both.
    """
    def __init__(self,shape = None, act_fn = tf.nn.relu, loss_fn = None,
                 batch_size = 50, dropout = False, keep_prob = 1.0,
                 learning_rate = 0.0005, decay = 1.0, momentum = 0.0, opt = tf.train.MomentumOptimizer,
                 summaries = False, log_dir = 'tmp/AE'):
        
        #architecture params
        self.shape = shape #first tuple is the shape of the input
        self.act_fn = act_fn
        
        #more params
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.dropout = dropout
        self.keep_prob = keep_prob
          
        #training params
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.opt = opt(learning_rate,momentum=momentum)
        
        #tensorboard params
        self.summaries = summaries
        self.log_dir = log_dir
        
class Discriminative(Net):
    """
    A discriminative network has;
    - a decoder.
    - a loss that is the error between their output and the data labels.
    """
    def __init__(self,decoder,shape, 
                 act_fn = tf.nn.relu,loss_fn = CrossEntropy,
                 batch_size = 50, dropout = False, keep_prob = 1.0,
                 learning_rate = 0.0005, decay = 1.0, momentum = 0.0, opt = tf.train.MomentumOptimizer,
                 summaries = False, log_dir = 'tmp/AE'):
        Net.__init__(self, shape, act_fn, loss_fn, batch_size, dropout, keep_prob, learning_rate, decay, momentum, opt, summaries, log_dir)
        self.decoder = decoder
    
        #Define all the variables and placeholders
        self.inputs = tf.placeholder(tf.float32,shape=self.shape[0])
        self.labels = tf.placeholder(tf.int64,shape=(None,1))
        self.onehot_labels = tf.squeeze(tf.one_hot(self.labels,10,1.0,0.0))
        self.decoder = self.decoder(self.shape[1],self.act_fn,self.dropout,self.keep_prob)
        
        #Build the computational graph
        with tf.name_scope('Discriminative'):
            self.out = self.decoder(self.inputs)
            self.error = self.loss_fn(self.out,self.onehot_labels)
            #self.accuracy = 
            self.train_step = self.opt.minimize(self.error)

class Generative(Net):
    """
    A generative network has;
    - an encoder and a decoder.
    - a loss that is the error between their reconstruction and the input.
    """
    def __init__(self,encoder,decoder,
                 shape = [784,10], act_fn = tf.nn.relu,
                 batch_size = 50, dropout = False,
                 learning_rate = 0.0005, decay = 1.0, momentum = 0.0, opt = tf.train.MomentumOptimizer,
                 summaries = False, log_dir = 'tmp/AE'):
        
        Net.__init__(self, shape, act_fn, batch_size, dropout, learning_rate, decay, momentum, opt, summaries, log_dir)
        
    def build(self):
        #Define all the variables and placeholders
        self.inputs = tf.placeholder(tf.float32,shape=self.shape)
        self.encoder = encoder(shape,dropout)
        self.decoder = decoder(shape,dropout)
        
        #Build the computational graph
        with tf.name_scope('Generative'):
            self.bottle = self.encoder(self.inputs)
            self.out = self.decoder(self.bottle)
            self.error = SquaredError(data,self.out)
            self.train_step = self.opt.minimise(self.error)
            
class Aversarial(Net):
    """
    A generative adversarial network has;
    - a generative net
    - a discriminative net
    - a loss that is ???
    """
    def __init__(self,encoder,decoder,
                 shape = [784,10], act_fn = tf.nn.relu,
                 batch_size = 50, dropout = False,
                 learning_rate = 0.0005, decay = 1.0, momentum = 0.0, opt = tf.train.MomentumOptimizer,
                 summaries = False, log_dir = 'tmp/AE'):
        
        Net.__init__(self, shape, act_fn, batch_size, dropout, learning_rate, decay, momentum, opt, summaries, log_dir)
        
        #Define all the variables and placeholders
        self.inputs = tf.placeholder(tf.float32,shape=self.shape)
        self.Gen = Generative()
        self.Dis = Discriminative()
        
        #Build the computational graph
        with tf.name_scope('Aversarial'):
            self.bottle = self.encoder(self.inputs)
            self.out = self.decoder(self.bottle)
            self.error = SquaredError(data,self.out)
            self.train_step = self.opt.minimise(self.error)