==========================================================================================

         Introduction to Natural Laguage Processing Assignment 3
 
==========================================================================================

1. In this assignment, you will be asked to:

  From Incrementality in Deterministic Dependency Parsing(2004, Nivre)
  - implement the arc-standard algorithm

  From A Fast and Accurate Dependency Parser using Neural Networks(2014, Danqi and Manning) 
  - implement feature extraction
  - implement the neural network architecture including activation function
  - implement loss function

  * You will need to copy your embeddings to this folder, and name it as "word2vec.model"
  Also, change "embedding_size" in Config.py to match your model.

2. This package contains several files:

  - DependencyParser.py: 
    This file is the main script for training your dependency parser.

  - DependencyTree.py
    The dependency tree class file.

  - ParsingSystem.py
    This file contains the class for a transition-based parsing framework for dependency parsing.

  - Configuration.py
    The configuration class file.

  - Config.py
    This file contains all hyper parameters.

  - Util.py
    This file contains functions for reading and writing CONLL file. 

  - data/
    train.conll - train set, labeld
    dev.conll - dev set, labeld
    test.conll - test set, *unlabeled*


3. What you should do:
  1. Implement the arc-standard algorithm in ParsingSystem.py
  2. Implement feature extraction in DependencyParser.py: getFeatures(...)
  3. Implement neural network architecture including activation function: forward_pass(...)
  4. Implement loss function and calculate loss value: in DependencyParserModel.build_graph(...)
  5. Try different number of hidden layers
  6. Try different non-linear activation functions
  7. Train a parser with Fixed embeddings -- by setting trainable=False in tf.Variable


4. What you should submit:

  Creating a single zip file contains the following files:
      1. DependencyParser.py
      2. ParsingSystem.py
      3. Configuration.py
      4. results_test.conll  // your prediction on test data.
      5. Report as detailed in the assignment pdf
      6. README file with explanation of your implementation details

