import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return A.dot(B) - C

def problem_1c (A, B, C):
    return A*B + C.T

def problem_1d (x, y):
    return x.T.dot(y)

def problem_1e (A, i):
    return np.sum(A[i,::2])

def problem_1f (A, c, d):
    return np.mean(np.nonzero((A>=c) & (A<=d)))

def problem_1g (A, k):
    return np.linalg.eig(A)[1][np.argsort(np.linalg.eig(A)[0])[-k:]]

def problem_1h (x, k, m, s):
    return np.random.multivariate_normal(x.flatten() + m * np.ones(x.shape[0]) , s*np.eye(x.shape[0]), size=k)

def problem_1i (A):
    return A[:,np.random.permutation(A.shape[1])]

def problem_1j (x):
    return (x - np.mean(x))/(np.std(x))

def problem_1k (x, k):
    return np.repeat(x[:,np.newaxis],k,axis=1)

def problem_1l (X, Y):
    
    # X was broadcasted in m dimension -- kxnx1
    # Y was broadcasted in n dimension --- kx1xm
    
    # Subtracting resulted in kxnxm 3D array 
    # after summation -- becomes nxm array
    return np.sqrt(np.sum(((X[:,:,np.newaxis] - Y[:,np.newaxis,:]) ** 2),axis=0))
