import numpy as np
from logistic_regression_functions 
import predict_proba, predict, cost_function, gradient

class GradientDescent(object):


    def __init__(self, cost, gradient, predict_func, 
                 alpha=0.01,
                 num_iterations=10000):
        
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.alpha = alpha
        self.num_iterations = num_iterations

    def fit(self, X, y, coeffs):

        m=len(y)
        J_history =[]

        for i in range(self.num_iterations):
            
            cost, grad = cost_function(X, y, coeffs)
            coeffs = coeffs - (self.alpha * grad)
            J_history.append(cost)
            
        self.coeffs = coeffs
            
        return J_history
    
    def get_coeffs(self):
            
        return self.coeffs

    def predict(self, X):

        pass
