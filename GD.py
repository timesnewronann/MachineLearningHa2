
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar


def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)  # number of training examples
    arrCost = []  # record all cost calculation for each iteration
    # transpose X into a vector  -> XColCount X m matrix
    transposedX = np.transpose(X)
    for iteration in range(numIterations):
        ################ PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate.
        # Replace the following variables if needed
        # G=
        # theta = np.subtract(theta, alpha*G)  # or theta = theta - alpha * gradient

        predictedValue = np.dot(X, theta)
        loss = predictedValue - y
        gradient = np.dot(transposedX, loss) / m
        theta = theta - alpha * gradient
        ################ PLACEHOLDER3 #end##########################

        ################ PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta;
        # atmp =
        # print(atmp)
        # arrCost.append(atmp)

        cost = (np.sum((X.dot(theta) - y) ** 2)) / (2 * m)

        arrCost.append(cost)
        ################ PLACEHOLDER4 #start##########################
    return theta, arrCost
