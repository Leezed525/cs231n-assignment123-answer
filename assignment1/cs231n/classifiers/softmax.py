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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 训练集的数量
    num_train = X.shape[0]
    # 分类的数量
    num_classes = W.shape[1]
    for i in range(num_train):
      scores = X[i] @ W
      # 对其求e的幂函数
      scores = np.exp(scores)
      # 求对于每一个分类的概率
      p = scores / np.sum(scores)
      # 求loss函数
      loss += -np.log(p[y[i]])

      # 求梯度
      for k in range(num_classes):
        # 获取当前分类的概率
        p_k = p[k]
        # 判断当前分类是否是正确分类
        if k == y[i]:
          dW[:,k] += (p_k - 1) * X[i] 
        else:
          dW[:,k] += (p_k) * X[i]


    # 处理正则项
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 训练集的数量
    num_train = X.shape[0]
    # 分类的数量
    num_classes = W.shape[1]

    # 先计算得分
    scores = X @ W
    # 再取e的幂函数
    scores = np.exp(scores)
    # 计算所有的概率
    p = scores / np.sum(scores,axis = 1,keepdims = True)
    # 计算loss函数
    loss += np.sum(-np.log(p[range(num_train),y]))

    # 计算梯度 根据上面的公式可以知道只要给正确分类的P - 1就可以得到dW
    p[range(num_train),y] -= 1
    dW = X.T @ p


    # 计算正则项
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
