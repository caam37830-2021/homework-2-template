# Homework 2 - Dense Linear Algebra

Please put any written answers in [`answers.md`](answers.md)

Reminder: you can embed images in markdown.  If you are asked to make a plot, save it as a `png` file, commit it to git, and embed it in this file.

Please put functions and class definitions in [`matlib.py`](matlib.py).  This file should contain a Python *module*.

Please put your code to generate plots, run experiments etc. in [`script.py`](script.py).  This file is a Python *script*.

You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

## Important Information

### Due Date
This assignment is due Friday, October 23 at 12pm (noon) Chicago time.

### Grading Rubric

The following rubric will be used for grading.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Problem 0 |  |   |  | /25 |
| Part A | /2 | /2 | /1 | /5 |
| Part B |  | /3 | /2 | /5 |
| Part C | /3 | /5 | /2 | /10 |
| Part D | /2 | /5 | /2 | /5 |
| Problem 1 |  |   |  | /55 |
| Part A | /12 | /15 | /3 | /30 |
| Part B | /3 | /10 | /2 | /15 |
| Part C | /2 | /6  | /2 | /10 |
| Problem 2 | /2 | /15  | /3 | /20 |

Correctness will be based on code (i.e. did you provide was was aksed for) and the content of [`answers.md`](answers.md).

To get full points on style you should use comments to explain what you are doing in your code and write docstrings for your functions.  In other words, make your code readable and understandable.

### Autograder

You can run tests for the functions you will write in problems 1-3 using either `unittest` or `pytest` (you may need to `conda install pytest`).  From the command line:
```
python -m unittest test.py
```
or
```
pytest test.py
```
The tests are in [`test.py`](test.py).  You do not need to modify (or understand) this code.

You need to pass all tests to receive full points.

Please enable GitHub Actions on your repository (if it isn't already) - this will cause the autograder to run automatically every time you push a commit to GitHub, and you can get quick feedback.


## Problem 0 - Matrix Factorizations (25 points)

In this problem, you'll practice applying some matrix factorizations in `scipy.linalg`, and do some performance comparisons.  

In this problem, you'll deal with the Cholesky decomposition. First, some definitions:

A matrix is symmetric positive-definite (SPD) if
1. A is symmetric (`A = A.T`)
2. Eigenvalues of A are all non-negative (strictly greater than 0)

An easy way to generate a random SPD matrix is:
```python
A = np.random.randn(m,n) # n >= m
A = A * A.T # use * not + for SPD!
```

The Cholesky factorization of a SPD matrix `A` is `A = L * L.T` where `L` is lower triangular.  Alternatively, we might say `A = U.T * U` where `U` is `L.T`.  This is a variant of the LU decomposition, where we can use symmetry in `A`.


### Part A (5 points)

Write a function `solve_chol` which solves a linear system using the Cholesky decomposition.  Explicitly, `x = solve_chol(A, b)` should (numerically) satisfy `A * x = b`.  You can assume that `A` is SPD.

Use [`cholesky`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html#scipy.linalg.cholesky) in `scipy.linalg` to compute the decomposition.

### Part B (5 points)

Create a log-log plot of the time it takes to compute the Cholesky decomposition vs. the LU decomposition of random n x n SPD matrices.  Compute times for 10 values of `n` logarithmically spaced between `n=10` and `n=4000`. (use `np.logspace` and `np.round`).  Make sure to give your plot axes labels, a legend, and a title.

Both factorizations take O(n^3) time to compute - which is faster in practice?

### Part C - Basic Matrix Functions (10 points)

One example of a matrix function is the matrix power `A**n`.  In Homework 0, you computed this using a version of the Egyptian algorithm.

If `A` is symmetric, recall `A` has an eigenvalue decomposition `A = Q @ L @ Q.T`, where `Q` is orthogonal, which can be computed with `eigh`.  Then we can write the power

`A**n = A @ A @ ... @ A`

as

`A**n = (Q @ L @ Q.T) @ (Q @ L @ Q.T) @ ...`

Because `Q.T @ Q = I` (the identity), this becomes

`A**n = Q * (L**n) * Q.T`

Recall that `L` is diagonal, so `L**n` can be computed element-wise.

Write a function `matrix_pow` where `matrix_pow(A, n)` computed via the Eigenvalue decomposition, as above.  You can assume that `A` is symmetric.  You can also just use the NumPy vectorized implementation of power for `L`.

Do you expect this function to be asymptotically faster or slower than the approach you used in Homework 0?  Considering that the Fibonacci numbers are integers, what issues might you need to consider if using this function to compute Fibonacci numbers?


### Part D - Determinant (5 points)

Calculating the determinant of a matrix is very easy using the LU decomposition.  Recall that if `A = B @ C`, that `det(A) = det(B) * det(C)`.

A useful property of lower and upper-triangular matrices `T` are that `det(T)` is just the product of the diagonal elements (you don't have to look at the off-diagonal elements).

Write a function `abs_det(A)` which computes the absolute value of the determinant of a square matrix `A` by using its LU decomposition.  One useful property of LU decompositions is that `L` by convention just has 1 on its diagonal.  Note that the determinant of a permutation matrix is either +1 or -1 depending on the parity of the permutation.

## Problem 1 - Matrix-Matrix Multiplication (55 points)

Recall that if we perform matrix-matrix multiplication `A = B @ C`, `A[i,j]` is the sum over `k` of `B[i,k] * C[k,j]`.  We'll use the conventions that `B.shape = p, r`, `C.shape = r, q`, and so `A.shape = p, q`.

In this problem, you should use the `njit` decorator from numba.  E.g.
```python
from numba import njit

@njit
def myfn():
	pass
```
This will make sure you get an error if the function isn't able to compile.

### Part A - Basic Matrix Multiplication (30 points)

There are three indices used in matrix multiplication, and we can write matrix-matrix multiplication using three nested loops:
```python
A = np.zeros((p, q))

for i in range(p):
	for j in range(q):
		for k in range(r):
			A[i,j] = A[i,j] + B[i,k] * C[k,j]
```

There are 6 different orders (outer loop to inner loop) in which you might loop over these indices: `ijk`, `ikj`, `jik`, `jki`, `kij`, `kji`.

Write 6 functions: `matmul_***` where `***` is replaced by each option above.  For example, `matmul_ijk` would use the nested loops demonstrated above.  Use Numba to make these functions compile just-in time.  For each function `A = matmul_***(B, C)` should be equivalent to `A = B @ C`

Compare the time it takes to run each of the 6 versions of `matmul_***` above with BLAS dgemm (called through SciPy) and NumPy `matmul`.  (Remember to precompile your JIT functions before timing).
Use random `n x n` matrices for `B` and `C`, i.e. `p = q = r = n`.  Use row-major `ndarray`s in NumPy.

Make a log-log plot of the runtimes of the 9 functions, for 10 values of `n` logarithmically spaced between 100 and 4000.  Include a legend, axis labels, and a plot title.

All of these implementations have an O(n**3) asymptotic run time, and perform an identical number of floating point operations.  Give an explanation for why some loop orders are faster than others - why is the fastest version fastest?  Why is the slowest version slowest?


### Part B - Blocked Matrix Multiplication (15 points)

In this problem, we'll assume that all matrices involved are `n x n`, and that `n` is a power of 2.

Another way to compute matrix-matrix multiplication is not to use three nested for-loops, but to use blocked multiplication.  This can have the advantage of (potentially) reducing the number of cache misses that occur when performing the algorithm, and be extended to parallel implementations.

In Python, this might look like
```python
A[I, J] = A[I, J] + B[I, K] @ C[K, J]
```
I, J, and K are all slices.

Write a function `matmul_blocked`, where `A = matmul_blocked(B, C)` is equivalent to `A = B @ C`.  The slices should be either `:n//2` or `n//2:`, so each recursive call reduces the number of rows and columns of the matrices involved by a factor of 2.

This function should be defined recursively, where the block multiplication also uses `matmul_blocked`, unless `n <= 64`, in which case, you can just use the best version of `matmul_***` from part A.  Use Numba to make this function JIT compile.

Compare the run time of this algorithm to the best version of `matmul_***` you wrote in part A.  Make a plot like you did in part A comparing the run times of these two functions for values of `n` in `[2**i for i in range(6,13)] `

Hint: one possible way to loop over all slices is to use a `slice` object in Python.  E.g.
```python
slices = (slice(0, n//2), slice(n//2, n))
for I in slices:
	for J in slices:
		for K in slices:
			pass
```
This is just an option - you could do something else.




### Part C - Strassen's Algorithm (10 points)

Again, we'll assume square `n x n` matrices, where `n` is a power of 2.

So far, all the algorithms we have seen for matrix-matrix multiplication are O(n**3).  [Strassen's algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) improves the asymptotic run time.  This is also a version of blocked matrix-matrix multiplication defined recursively, but the number of multiplications done on blocks is 7 instead of 8, at the cost of extra additions.  The Master Theorem for recursion can be used to give an asymptotic run time of `n**log2(7) = n**2.81`.

Write a function `matmul_strassen` which implements Strassen's algorithm using recursion.  Again, you can use regular matrix-matrix multiplication (choose the best `matmul_***` in part A) if the block size is `<= 64`.  Again, use Numba to produce a JIT function.

Here is pseudo-code which gets you pretty close to what you need:

```python
# compute A = B @ C
s1 = slice(0, n//2)
s2 = slice(n//2, n)

B11, B12, B21, B22 = B[s1,s1], B[s1,s2], B[s2, s1], B[s2, s2]
C11, C12, C21, C22 = C[s1,s1], C[s1,s2], C[s2, s1], C[s2, s2]

M1 = (B11 + B22) @ (C11 + C22)
M2 = (B21 + B22) @ C11
M3 = B11 @ (C12 - C22)
M4 = B22 @ (C21 - C11)
M5 = (B11 + B12) @ C22
M6 = (B21 - B11) @ (C11 + C12)
M7 = (B12 - B22) @ (C21 + C22)

A[s1, s1] = M1 + M4 - M5 + M7
A[s1, s2] = M3 + M5
A[s2, s1] = M2 + M4
A[s2, s2] = M1 - M2 + M3 + M6
```

Yes, this is a bit of an exercise to verify (you can if you want to, but don't need anything in `answers.md`).  You can essentially just replace `@` with the appropriate function call in the above to write the algorithm (and implement the rest of the function).

Create a loglog plot of run times for `n` in `[2**i for i in range(6,13)]`.  Compare `matmul_strassen` to `matmul_blocked`. Add a legend, axis labels, and title.

Does Strassen's algorithm actually beat blocked matrix multiplication for the values of `n` that you tested?

Note: Strassen's algorithm is not as stable (numerically speaking) as standard matrix-matrix multiplication, because the additions and subtractions create opportunities for (catastrophic) numerical error.

Also note that the optimal exponent for matrix-matrix multiplication is still unknown.  The best value so far is due to Coppersmith and Winograd, and has a value of about 2.37.  This algorithm would be impractical to implement.


## Problem 3 - Markov Chains (20 points)

A Markov Chain models a random walk through a sequence of states. We'll consider a simple random walk on a 1-dimensional grid.  A "classic" description of the problem is that a drunk (or very confused) person wanders down the sidewalk.  However, because they are easily confused, they choose whether to take a step forward or backward with equal probability.

We'll consider discrete positions on the sidewalk.  We'll say there are `n` positions total, and that position `i` is adjacent to positions `i+1` and `i-1`, except for position `0` which is only adjacent to position `1`, and position `n-1` which is only adjacent to position `n-2`.  At these endpoints, we'll say the person either stays in the same location or moves to the neighboring position with equal probability (they either stand confused or turn around).

The drunkard's position at time `t` is only dependent on their position at time `t-1`.  Because they are making random movements, their position is a random variable `p`.  `p` can be expressed as a vector, where `p[i|t]` is the probability that the person is at position `i` at time `t`.  To get to time `t` from time `t-1`, we can compute `p[i|t] = 0.5 * p[i-1|t-1] + 0.5 * p[i+1|t-1]` (with necessary modifications for endpoints).

Another way to state this is that we can get the vector `p` at time `t` from the vector `p` at time `t-1` by calculating the product `A * p[:|t-1]`, where `A` is an `n x n` matrix, with the `i`th column of `A` has `0.5` in row `i+1`, `0.5` in row `i-1`, and `0` in all other entries (this encodes the probabilities of where the person can end up from state `i`).

1. Write a function `markov_matrix(n)` which returns the matrix `A` above for the random walk on the sidewalk of length `n`.  Make the necessary modifications for the endpoints of the sidewalk.
2. Run a simulation where the person starts at `i=0` at time `t=0` on a sidewalk of length `n=50` (i.e. `p` at time 0 is a vector of length 50 with `p[0] = 1` and 0 everywhere else).  Plot the vector `p` for `t in (10, 100, 1000)`.  Give your plot labels and a title.
3. Calculate the eigenvector with largest eigenvalue of the matrix `A`.  Normalize the vector so its entries sum to 1. (e.g. `v = v / np.sum(v)`). How close is this to `p` at `t=1000` in part 2 (give the euclidean distance between the two vectors?)  How close is this to `p` at `t=2000`?


A few notes on the above problem (you don't need to use this information in your answers):
* In this problem, `A` is what is called a *doubly stochastic* matrix (rows and columns sum to 1)
* `A` is also symmetric and tri-diagonal.  You can use `la.eigh_tridiagonal` for maximum performance.
* You may notice a strong similarity between running the simulation and running power method (see the OOP lecture)
* No matter how many times you apply the matrix `A`, you should have `np.sum(p) = 1` (it is a probability distribution after all...) If you're theoretically inclined, try proving this from the properties of the matrix `A`.
* over time, `p` "spreads out" - this demonstrates our uncertainty about where the random walker will be.




## Feedback

If you'd like share how long it took you to complete this assignment, it will help adjust the difficulty for future assignments.  You're welcome to share additional feedback as well.
