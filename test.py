import unittest

from matlib import *
import numpy as np

tol = 1e-8

class TestChol(unittest.TestCase):

	def setUp(self):
		pass

	def test_solve_chol(self):
		m = 10
		n = 20
		np.random.seed(0)
		for i in range(5):
			A = np.random.randn(m,n)
			A = A @ A.T
			x = np.random.rand(m)
			b = A @ x
			x2 = solve_chol(A, b)
			self.assertTrue(np.all(np.abs(x - x2) < tol))


class TestPow(unittest.TestCase):

	def setUp(self):
		pass

	def test_matrix_pow(self):
		m = 10
		n = 10
		np.random.seed(0)
		for i in range(5):
			A = np.random.randn(m,n)
			A = A + A.T

			An = matrix_pow(A, i)
			An2 = np.linalg.matrix_power(A, i)

			self.assertTrue(np.all(np.abs(An - An2) < tol))


class TestDet(unittest.TestCase):

	def setUp(self):
		pass

	def test_det(self):
		m = 10
		n = 10
		np.random.seed(0)
		for i in range(5):
			A = np.random.randn(m,n)

			d = abs_det(A)
			d2 = la.det(A)

			self.assertAlmostEqual(d, abs(d2))


class TestMatmul(unittest.TestCase):

	def setUp(self):
		pass


	def test_matmuls(self):

		matmuls = (
			matmul_ijk,
			matmul_ikj,
			matmul_jik,
			matmul_jki,
			matmul_kij,
			matmul_kji
		)

		p, q, r = 5,6,7

		np.random.seed(0)
		for i in range(5):
			B = np.random.randn(p, r)
			C = np.random.randn(r, q)

			A_true = B @ C

			for mm in matmuls:
				A = mm(B, C)
				self.assertTrue(np.all(np.abs(A - A_true) < tol))


	def test_matmuls2(self):

		matmuls = (
			matmul_ijk,
			matmul_ikj,
			matmul_jik,
			matmul_jki,
			matmul_kij,
			matmul_kij
		)

		p, q, r = 100, 101, 102

		np.random.seed(0)
		for i in range(5):
			B = np.random.randn(p, r)
			C = np.random.randn(r, q)

			A_true = B @ C

			for mm in matmuls:
				A = mm(B, C)
				self.assertTrue(np.all(np.abs(A - A_true) < tol))


class TestBlocked(unittest.TestCase):

	def setUp(self):
		pass


	def test_matmul_blocked(self):
		n = 2**9

		np.random.seed(0)
		for i in range(5):
			B = np.random.randn(n, n)
			C = np.random.randn(n, n)

			A_true = B @ C

			A = matmul_blocked(B, C)

			self.assertTrue(np.all(np.abs(A - A_true) < tol))


class TestStrassen(unittest.TestCase):

	def setUp(self):
		pass


	def test_matmul_strassen(self):
		n = 2**9

		np.random.seed(0)
		for i in range(5):
			B = np.random.randn(n, n)
			C = np.random.randn(n, n)

			A_true = B @ C

			A = matmul_strassen(B, C)

			self.assertTrue(np.all(np.abs(A - A_true) < tol))


class TestMarkov(unittest.TestCase):

	def setUp(self):
		pass


	def test_markov_matrix(self):
		B = np.array(
			[[0.5, 0.5, 0. , 0. ],
			[0.5, 0. , 0.5, 0. ],
			[0. , 0.5, 0. , 0.5],
			[0. , 0. , 0.5, 0.5]])

		A = markov_matrix(4)

		self.assertTrue(np.all(A == B))
