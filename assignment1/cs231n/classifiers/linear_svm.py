from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # 正确分类的梯度减上X[i]
                dW[:,y[i]] -= X[i].T
                # 错误分类的梯度加去X[i]
                dW[:,j] += X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train


    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
 


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 梯度同样处理
    dW /= num_train
    # 正则项的梯度
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X @ W
    # 获取对于每个x而言正确分类的分数
    scores_correct = scores[range(num_train),y].reshape((scores.shape[0],1))
    # 对每个元素做max(0,scores_error - scores_correct + 1)操作，包括正确分类的元素
    # 统一操作后减少代码编写难度，只需要最后处理一下正确分类的分数，把他们变成0就行了
    margins = np.maximum(0,scores - scores_correct + 1)
    # 将正确分类的margins置为0
    margins[range(num_train),y] = 0
    loss += np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 先把所有margins > 0 的标记为1 ，因为我们算梯度的时候并不需要用到具体的元素的loss是多少
    # 我们只想知道这个元素有没有被算进loss里面
    margins[margins > 0] = 1
    # 并且，对于每一个分类错误的元素而言，他对错误分类的W的梯度影响是 +X[i]
    # 他对正确分类的W的梯度影响是-X[i]
    # 并且正确分类的分数位置我们是知道的
    # 因此我们只需要计算对于X[i]而言，有多少个分类 > 0,就代表错误分类的个数
    # 这个数量就是影响了梯度的数量，并且我们已经把错误分类的位置记为了1
    # 接下来我们只要做到在正确分类的位置 - 错误分类的个数
    # 接下来是举例，一直我们一共有10个分类，对于X[i]而言，我们有3个分类正确，加上一个本来就是正确分类的分数
    # 那么剩下6个分类错误的，也就是错误分类的预估值> 正确分类的预估值 - 1 的数量
    # 那么对于梯度而言，我们只需要对正确分类的梯度 减去六个X[i]就行，错误分类的个数各自 加上一个X[i]，具体的结合矩阵的shape
    # 思考一下，如果实在理解不了可以给我留言，我画个图
    row_sum = np.sum(margins,axis = 1)
    margins[range(num_train),y] = -row_sum
    dW += np.dot(X.T, margins)/num_train + reg * W 


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
