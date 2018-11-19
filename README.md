# neural-dependency-parser
Implementation of the Neural Network based Transition Parsing system described in "A fast and accurate dependency parser using neural networks" by Chen and Manning, EMNLP 2014


### Implementation details

Python Version: 2.7.15
Embeddings used: Default embeddings provided by the TA for Python 2.x

1. Implement the arc-standard algorithm in ParsingSystem.py
Implemented as required. <br>
2. Implement feature extraction in DependencyParser.py: getFeatures(...)
Implemented as required. <br>
3. Implement neural network architecture including activation function: forward_pass(...)
There were a lot of variations of 'forward pass' implemented as a part of the experimentation process.
The default version is currently setup in the file, with the changes for the 'best configuration', marked as comments. <br>

4. Implement loss function and calculate loss value: in DependencyParserModel.build_graph(...)

For the calculation of the loss function, we avoid the train_labels with label = -1 by making a mask of the input with -1 elements,
multiply all such elements in  the train label and prediction with 0. Doing this elements those elements from contributing to the loss.

For the actuall calculation of the loss, we take the argmax over train labels and use the tensorflow function: tf.losses.sparse_softmax_cross_entropy. For regularization we add the 'tf.nn.l2_loss' of all the elements

5. and 6. Experimentation details added in the report

7. Embeddings are formed by using a onehot vector to represent POS and Dep labels, unknwon word embeddings are still randomly allocated