import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar
import cPickle as pkl

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ==
            """
            
            
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32,name='all_embeddings', trainable=True)

            feature_len = 18 + 18 + 12
            numTrans = parsing_system.numTransitions()

            self.train_inputs = tf.placeholder("int32",[Config.batch_size,feature_len], name= 'train_input')
            self.train_labels = tf.placeholder("int32",[Config.batch_size,numTrans], name= 'train_labels')
            self.train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs, name = 'train_embed')

            
            # ###### for best configuration ######
            # weights_input = tf.Variable(tf.random.truncated_normal([18*Config.embedding_size, Config.hidden_size],stddev=0.08), name='w_in')
            # biases_input = tf.Variable(tf.random.truncated_normal([Config.hidden_size],stddev=0.08), name='b_in')
            
            # weights_h2 = tf.Variable(tf.random.truncated_normal([18*Config.embedding_size, Config.hidden_size], stddev=0.08), name='w_h2')
            # biases_h2 = tf.Variable(tf.random.truncated_normal([Config.hidden_size], stddev=0.08), name='b_h2')
            
            
            # weights_h3 = tf.Variable(tf.random.truncated_normal([12*Config.embedding_size, Config.hidden_size], stddev=0.008), name='w_h3')
            # biases_h3 = tf.Variable(tf.random.truncated_normal([Config.hidden_size], stddev=0.08), name='b_h3')

            # weights_output = tf.Variable(tf.random.truncated_normal([3*Config.hidden_size, numTrans], stddev=0.08), name='w_out')

            # self.prediction = self.forward_pass(self.train_embed, weights_input, biases_input, weights_output,    
            #                                     weights_h2, biases_h2, weights_h3, biases_h3)
            
            # ### masked cross entropy loss
            # mask = self.train_labels > -1
            # mask = tf.dtypes.cast(mask,'int32')
            # self.labels = tf.math.multiply(self.train_labels,mask)
            # self.prediction = tf.math.multiply(self.prediction,tf.dtypes.cast(mask,'float'))
            
            # self.labels = tf.math.argmax(self.train_labels,axis=1)
            # self.cce = tf.losses.sparse_softmax_cross_entropy(self.labels,self.prediction)

            
            # self.regularization = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_output)\
            #                     + tf.nn.l2_loss(self.train_embed)\
            #                     + tf.nn.l2_loss(weights_h2) + tf.nn.l2_loss(biases_h2)\
            #                     + tf.nn.l2_loss(weights_h3) + tf.nn.l2_loss(biases_h3)
            
            # self.loss = self.cce + Config.lam * self.regularization
            
            # optimizer = tf.train.AdamOptimizer(Config.learning_rate)

            
            # grads = optimizer.compute_gradients(self.loss)
            # clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            # self.app = optimizer.apply_gradients(clipped_grads)            
            
            # ###### for best configuration ######
            
            # ## multiple hidden layer experiments ###
            # weights_h2 = tf.Variable(tf.random.truncated_normal([Config.hidden_size, Config.hidden_size], stddev=0.08), name='w_h2')
            # biases_h2 = tf.Variable(tf.random.truncated_normal([Config.hidden_size], stddev=0.08), name='b_h2')
            
            
            # weights_h3 = tf.Variable(tf.random.truncated_normal([Config.hidden_size, Config.hidden_size], stddev=0.08), name='w_h3')
            # biases_h3 = tf.Variable(tf.random.truncated_normal([Config.hidden_size], stddev=0.08), name='b_h3')
            # ### multiple hidden layers experiments ###
    
    
            ####### default configuration #######
            weights_input = tf.Variable(tf.random.truncated_normal([feature_len*Config.embedding_size, Config.hidden_size],stddev=0.08), name='w_in')
            biases_input = tf.Variable(tf.random.truncated_normal([Config.hidden_size],stddev=0.08), name='b_in')

            weights_output = tf.Variable(tf.random.truncated_normal([Config.hidden_size, numTrans], stddev=0.08), name='w_out')

            self.prediction = self.forward_pass(self.train_embed, weights_input, biases_input, weights_output)    
            
            self.labels = tf.math.argmax(self.train_labels,axis=1)
            self.cce = tf.losses.sparse_softmax_cross_entropy(self.labels,self.prediction)

            
            self.regularization = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_output)\
                                + tf.nn.l2_loss(self.train_embed)
            
            self.loss = self.cce + Config.lam * self.regularization
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)


            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            

            ####### default configuration #######
            

            
            # For test data, we only need to get its prediction
            self.test_inputs = tf.placeholder("int32",[feature_len], name='test_input')
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1], name='test_embed')
            
            # ###### for best configuration ######
            # self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output,
            #                                    weights_h2, biases_h2, weights_h3, biases_h3)

            # ###### for best configuration ######
            
            ####### default configuration #######
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)
            ####### default configuration #######
            
            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"
        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val, cce, regu = sess.run([self.app, self.loss, self.cce, self.regularization], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)

    ## for best config ## 
    # def forward_pass(self, embed, weights_input, biases_input, weights_output, weights_h2, biases_h2, weights_h3, biases_h3):
    ## for best config ##

    ## for default config ##
    def forward_pass(self, embed, weights_input, biases_input, weights_output):
    ## for default config ##
        """
        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        embed = tf.reshape(embed,[embed.shape[0],-1])
        
        # ######## For best configuration ##########
        # word_embed, pos_embed, deb_embed = tf.split(embed,
        #                                             [18*Config.embedding_size, 18*Config.embedding_size, 12*Config.embedding_size],
        #                                             axis=1)        
        
        # word_h = tf.add(tf.matmul(word_embed,weights_input), biases_input)
        # pos_h = tf.add(tf.matmul(pos_embed,weights_h2), biases_h2)
        # deb_h = tf.add(tf.matmul(deb_embed,weights_h3), biases_h3)
        
        # # cube activation function for experiment 2.b
        # # word_act = tf.pow(word_h,3)
        # # pos_act = tf.pow(pos_h,3)
        # # deb_act = tf.pow(deb_h,3)
        

        # word_act = tf.nn.relu(word_h)
        # pos_act = tf.nn.relu(pos_h)
        # deb_act = tf.nn.relu(deb_h)
    
        
        # all_hidden = tf.concat([word_act,pos_act,deb_act],axis=1)
        # # all_hidden = tf.math.multiply(tf.math.multiply(word_act,pos_act),deb_act )
                
        # output = tf.matmul(all_hidden,weights_output)
        # ####### For best configuration ##########
        
        ###### For default model + commented out parts for accomodating multiple layers ######

        hidden_vals = tf.add(tf.matmul(embed,weights_input), biases_input)


        ## options of different activation functions ###
        activation = tf.pow(hidden_vals,3)
        # activation = tf.nn.relu(hidden_vals)
        # activation = tf.nn.sigmoid(hidden_vals)
        # activation = tf.nn.tanh(hidden_vals)
        
        # # second hidden layer ##
        # h2 = tf.add(tf.matmul(activation,weights_h2), biases_h2)
        # h2_activation = tf.pow(h2,3)
        # h2_activation = tf.nn.relu(h2)
        # h2_activation = tf.nn.sigmoid(h2)
        # h2_activation = tf.nn.tanh(h2)
        # # second hidden layer ##
    
        # third hidden layer ##
        # h3 = tf.add(tf.matmul(h2_activation,weights_h3), biases_h3)
        # h3_activation = tf.pow(h3,3)
        # h3_activation = tf.nn.relu(h3)
        # h3_activation = tf.nn.sigmoid(h3)
        # # third hidden layer ##

        output = tf.matmul(activation,weights_output)
        ###### For default model + commented out parts for accomodating multiple layers ######
        
        
        return output



def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    features = []

    for i in range(3):
        word_ind = c.getStack(i)
        word = c.getWord(word_ind)
        word_id = getWordID(word)

        features.append(word_id)
    
    for i in range(3):
        word_ind = c.getBuffer(i)
        word = c.getWord(word_ind)
        word_id = getWordID(word)

        features.append(word_id)

    for i in range(2):
        for j in range(1,3):
            word_ind = c.getLeftChild(c.getStack(i),j)
            word = c.getWord(word_ind)
            word_id = getWordID(word)

            features.append(word_id)

            word_ind = c.getRightChild(c.getStack(i),j)
            word = c.getWord(word_ind)
            word_id = getWordID(word)

            features.append(word_id)

    for i in range(2):
        word_ind = c.getLeftChild(c.getLeftChild(c.getStack(i),1),1)

        word = c.getWord(word_ind)
        word_id = getWordID(word)

        features.append(word_id)

        word_ind = c.getRightChild(c.getRightChild(c.getStack(i),1),1)

        word = c.getWord(word_ind)
        word_id = getWordID(word)

        features.append(word_id)

    pos_features = []
    for ind in features:
        pos = c.getPOS(ind)
        pos_id = getPosID(pos)

        pos_features.append(pos_id)

    label_features = []
    for ind in features[6:]: # excluding the 3 words from the top of stack and top of buffer
        label = c.getLabel(ind)
        if label == Config.UNKNOWN:
            label = Config.NULL
        label_id = getLabelID(label)

        label_features.append(label_id)
    
    features.extend(pos_features)
    features.extend(label_features)


    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):

        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
            
#             print(trees[i].equal(c.tree))
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    pos_ind = 0
    dep_ind = 0
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        #### for default behavior ####
        else:
            # for unknown words, pos and labels
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01            
          
        #### for default behavior ####

        ###### for fixed embeddings ########
        # elif i < len(wordDict):
        #     #for unknown words
        #     embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01            
        
        # elif pos_ind < len(posDict):
        #     dummy_emb = [0 for _ in range(Config.embedding_size)]
        #     dummy_emb[pos_ind] = 1
        #     embedding_array[i] = dummy_emb[:]
        #     pos_ind+=1
        # else:
        #     dummy_emb = [0 for _ in range(Config.embedding_size)]
        #     dummy_emb[dep_ind] = 1
        #     embedding_array[i] = dummy_emb[:]
        #     dep_ind+=1
        ###### for fixed embeddings ########
            
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Training Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

