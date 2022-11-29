import numpy as np

def predict_proba(X, coeffs):

    return 1/ (1 + np.exp(-(X@coeffs)))


def predict(X, coeffs, thresh=0.5):
   
    results = predict_proba(X, coeffs)
    return (results > thresh)*1

def cost_function(X, y, coeffs):
   
    m=len(y)
    y=y[:,np.newaxis]
    
    predictions = predict_proba(X, coeffs)
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
    cost = 1/m * sum(error)
    
    # compute gradient
    grad = 1/m * (X.transpose() @ (predictions - y))#[0]
    #j_1 = 1/m * (X.transpose() @ (predictions - y))[1:]
    
    #grad= np.vstack((j_0[:,np.newaxis],j_1))
    
    return cost[0], grad

def gradient(X, y, coeffs):
   
    pass
