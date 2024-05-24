from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W # (N x C)
    scores -= np.max(scores) # multiply constant !!! -> numeric stability

    for i in range(num_train):
      # scores[i] -= np.max(scores[i]) # multiply constant !!! -> numeric stability
      p = np.exp(scores[i]) / np.sum(np.exp(scores[i]))
      loss -= np.log(p[y[i]])

      scoress = scores[i]
      dW[:,y[i]] -= X[i]
      dW += X[i].reshape(-1, 1) @ p.reshape(1,-1)

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_class = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W # (N x C)
    scores -= np.max(scores) # multiply constant !!! -> numeric stability
    p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
    loss = -np.sum(np.log(p[range(num_train), y])) / num_train # cross-entropy
    loss += reg * np.sum(W * W)
    
    # compute gradient
    coef = p.copy()
    coef[range(num_train), y] = - (1 - p[range(num_train), y])
    dW = X.T @ coef / num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
