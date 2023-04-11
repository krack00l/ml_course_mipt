import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return : float
            single number with MSE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.
        
        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        return np.mean((X.dot(w) - Y)**2)

    @staticmethod
    def mae(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
                
        Return: float
            single number with MAE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.

        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        
        return np.mean(np.abs(X.dot(w) - Y))

    @staticmethod
    def l2_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return: float
            single number with sum of squared elements of the weight matrix ( \sum_{ij} w_{ij}^2 )

        Computes the L2 regularization term for the weight matrix w.
        """
        
        
        return np.sum(w**2)

    @staticmethod
    def l1_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : float
            single number with sum of the absolute values of the weight matrix ( \sum_{ij} |w_{ij}| )
        
        Computes the L1 regularization term for the weight matrix w.
        """

        # YOUR CODE HERE
        return np.sum(np.abs(w))

    @staticmethod
    def no_reg(w):
        """
        Simply ignores the regularization
        """
        return 0.
    
    @staticmethod
    def mse_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        
        Return : numpy array of same shape as `w`

        Computes the MSE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """
        '''
        output = np.zeros(w.shape)
        for j in range(w.shape[1]): # j = 4
            mse_derivative_buff_1 = 0
            mse_derivative_buff_2 = 0
            for k in range(X.shape[0]): # k = 406 I = 2
                mse_derivative_buff_1 += 2*(w[0,j]*(X[k,0]**2) + w[1,j]*X[k,1]*X[k,0] - Y[k,j]*X[k,0])
                mse_derivative_buff_2 += 2*(w[1,j]*(X[k,1]**2) + w[0,j]*X[k,1]*X[k,0] - Y[k,j]*X[k,1])
            output[0,j] = mse_derivative_buff_1/406/4
            output[1,j] = mse_derivative_buff_2/406/4
            #на 406 понятно зачем делить но нафиг делить на 4 не понял, чисто по данным увидел что нужно разделить на 4                
        '''
        return (2 * X.transpose().dot(X.dot(w) - Y)) / w.shape[1] / X.shape[0]

    @staticmethod
    def mae_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        
        Return : numpy array of same shape as `w`

        Computes the MAE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """
        
        '''
        print(output.shape)
        for j in range(w.shape[1]):
            mae_derivative_1 = sum(abs(X))[0]/406
            mae_derivative_2 = sum(abs(X))[1]/406
            output[0, j] = mae_derivative_1/4
            output[1, j] = mae_derivative_2/4

        for j in range(w.shape[1]): # j = 4
            mse_derivative_buff_1 = 0
            mse_derivative_buff_2 = 0
            for k in range(X.shape[0]): # k = 406 I = 2
                mse_derivative_buff_1 += X[k,0]
                mse_derivative_buff_2 += X[k,1]
            output[0,j] = mse_derivative_buff_1/406/4
            output[1,j] = mse_derivative_buff_2/406/4
        '''
        output = np.zeros(w.shape)

        for i in range(w.shape[0]): #2
            for j in range(w.shape[1]): #4
                for k in range(X.shape[0]): #406
                    Sum = 0
                    for l in range(X.shape[1]): #2
                        Sum = Sum + X[k,l] * w[l,j]
                    output[i,j] = output[i,j] + X[k,i] * np.sign(Sum - Y[k,j])
        return output / X.shape[0] / w.shape[1]



    @staticmethod
    def l2_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return : numpy array of same shape as `w`

        Computes the L2 regularization term derivative w.r.t. the weight matrix w.
        """ 
        
        return 2*w

    @staticmethod
    def l1_reg_derivative(w):
        """
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)

        Return : numpy array of same shape as `w`

        Computes the L1 regularization term derivative w.r.t. the weight matrix w.
        """
        '''
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i,j] = 1
        '''        
        return np.sign(w)

    @staticmethod
    def no_reg_derivative(w):
        """
        Simply ignores the derivative
        """
        return np.zeros_like(w)
