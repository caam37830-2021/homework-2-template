"""
matlib.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""

import numpy as np
import numpy.linalg as la
import time
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from numpy import random
import scipy.linalg as sl
from numba import njit
from scipy.linalg import eigh
from scipy.spatial import distance

# Problem 0

# Part A...

def solve_chol(A, b):
    L = cholesky(A, lower=True) #A = L * L.T 
    y = la.solve(L, b) #L*L.T * x = b
    return la.solve(L.T, y)  #L.T * x = y

# Part B

def my_solve(A, b):
    """
    solve A1 * x = b for x

    use LU decomposition
    """
    P, L, U = sl.lu(A)
    x = sl.solve_triangular(U, sl.solve_triangular(L, P.T @ b, lower=True), lower=False)
    return x

n = np.logspace(np.log10(10), np.log10(4000), num=10)
n = np.int64(np.round(n))
tlu = list()
tchol = list()
valdiff = list()

for i in n:
    A = np.random.randn(i, i)
    A = A @ A.T 
    x = np.random.rand(i)
    b = A @ x

    t0lu = time.time()
    x1 = my_solve(A, b)
    t1lu = time.time()
    tludiff = t1lu - t0lu #operation time
    tlu.append(tludiff)

    t0chol = time.time()
    x2 = solve_chol(A, b)
    t1chol = time.time()
    tcholdiff = t1chol - t0chol
    tchol.append(tcholdiff)

    
# problem c
def matrix_pow(A, n):
    L, X = la.eig(A) #L is eigenvalues of A, X is eigenvectors of A 
    L = np.diag(np.power(L, n)) #let L be diagonal matrix with elements L**n
    return X @ L @ X.T 


# problem d
def abs_det(A):
    P, L, U = sl.lu(A) 
    U = np.diag(U)
    ans=1
    for i in range(U.shape[1]):
        ans = ans*U[i][i] #product of diagonal elements of U
    return abs(ans)




# Problem 1
# part a
@njit
def matmul_ijk(B,C):
    p = B.shape[0] #p is the number of rows of B
    q = C.shape[1] #q is the number of columns of C
    r = B.shape[1] #r is the number of columns of B
    A = np.zeros((p,q))
    for i in range(p):
        for j in range(q):
            for k in range(r):
                A[i, j] = A[i, j] + B[i, k] * C[k, j]

    return A

@njit
def matmul_ikj(B,C):
    p = B.shape[0]
    q = C.shape[1]
    r = B.shape[1]
    A = np.zeros((p,q))

    for i in range(p):
        for k in range(r):
            for j in range(q):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]

    return A

@njit
def matmul_jik(B,C):
    p = B.shape[0]
    q = C.shape[1]
    r = B.shape[1]
    A = np.zeros((p,q))

    for j in range(q):
        for i in range(p):
            for k in range(r):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]

    return A

@njit
def matmul_kji(B,C):
    p = B.shape[0]
    q = C.shape[1]
    r = B.shape[1]
    A = np.zeros((p,q))

    for k in range(r):
        for j in range(q):
            for i in range(p):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]

    return A

@njit
def matmul_jki(B,C):
    p = B.shape[0]
    q = C.shape[1]
    r = B.shape[1]
    A = np.zeros((p,q))

    for j in range(q):
        for k in range(r):
            for i in range(p):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]

    return A

@njit
def matmul_kij(B,C):
    p = B.shape[0]
    q = C.shape[1]
    r = B.shape[1]
    A = np.zeros((p,q))

    for k in range(r):
        for i in range(p):
            for j in range(q):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]

    return A



#part b

def matmul_blocked(B,C):
    n = B.shape(1) #n is the number of columns of B
    A = np.zeros(n, n)
    if n > 64:
        slices = (slice(0, n // 2), slice(n // 2, n)) #when n>64, continue block the matrix
        for I in slices:
            for J in slices:
                for K in slices:
                    A[I, J] = A[I, J] + matmul_blocked(B[I, K],C[K, J])
    else:
        A = matmul_blocked(B,C)

    return A



@njit
def matmul_strassen(B,C):
    # compute A = B @ C
    n = B.shape(1)
    if n < 64:
       return matmul_ikj(B,C) #when n<64, use matmul_ikj to compute

    s1 = slice(0, n//2)
    s2 = slice(n//2, n)
    B11, B12, B21, B22 = B[s1,s1], B[s1,s2], B[s2, s1], B[s2, s2]
    C11, C12, C21, C22 = C[s1,s1], C[s1,s2], C[s2, s1], C[s2, s2]

    M1 = matmul_strassen((B11 + B22),(C11 + C22))
    M2 = matmul_strassen((B21 + B22),C11)
    M3 = matmul_strassen((B11 + B22),(C11 + C22))
    M4 = matmul_strassen(B22, (C21 - C11))
    M5 = matmul_strassen((B11 + B12), C22)
    M6 = matmul_strassen((B21 - B11), (C11 + C12))
    M7 = matmul_strassen((B12 - B22), (C21 + C22))

    A = np.zeros(n, n)
    A[s1, s1] = M1 + M4 - M5 + M7
    A[s1, s2] = M3 + M5
    A[s2, s1] = M2 + M4
    A[s2, s2] = M1 - M2 + M3 + M6

    return A



# Problem 2
#part a
def markov_matrix(n):
    A = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            A[i][i], A[i + 1][i] = 0.5, 0.5 #when i=0,row i will either be i or i+1 with both prob=0.5
        elif i == n - 1:
            A[i][i], A[i - 1][i] = 0.5, 0.5 #when i=n-1,row i will either be i or i-1 with both prob=0.5
        else:
            A[i - 1][i], A[i + 1][i] = 0.5, 0.5 ##when i=others,row i will either be i-1 or i+1 with both prob=0.5
    return A



#part C
w1, v1 = la.eig(markov_matrix(n4)) #w1 are the eigenvalues,v1 are the eigenvectors
w, v = eigh(markov_matrix(n4),subset_by_index=[0, 0]) #w is the largest eigenvalue, v is the corresponding eigenvector
v = v / sum(v1) #normalize v

dst1 = distance.euclidean(v, p2) #the euclidean distance between v and p2
dst2 = distance.euclidean(v, p3) #the euclidean distance between v and p3


