import unittest
import numpy as np
import faktoryzacja


class Test_lu(unittest.TestCase):
    def test1(self):
        matrix = np.array([[4, 4, 4], [1, 1, 18], [2, 10, 4]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0.], [0.25, 1., 0.], [0.5, -0.11111111, 1.]])
        uu = np.array([[4., 4., 4.], [0., 9., 3.], [0., 0., 16.33333333]])
        pp = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())

    def test2(self):
        matrix = np.array([[5, 3, 2], [1, 2, 0], [3, 0, 4]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0.], [0.2, 1., 0.], [0.6, -1.28571429, 1.]])
        uu = np.array([[5., 3., 2.], [0., 1.4, -0.4], [0., 0., 2.28571429]])
        pp = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())

    def test3(self):
        matrix = np.array(
            [[6, 1, 0, 0, 0, 0], [1, 6, 1, 0, 0, 0], [0, 1, 6, 1, 0, 0], [0, 0, 1, 6, 1, 0], [0, 0, 0, 1, 6, 1],
             [0, 0, 0, 0, 1, 6]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0., 0., 0., 0.], [0.16666667, 1., 0., 0., 0., 0.], [0., 0.17142857, 1., 0., 0., 0.],
                       [0., 0., 0.17156863, 1., 0., 0.], [0., 0., 0., 0.17157275, 1., 0.],
                       [0., 0., 0., 0., 0.17157287, 1.]])
        uu = np.array([[6., 1., 0., 0., 0., 0.], [0., 5.83333333, 1., 0., 0., 0.], [0., 0., 5.82857143, 1., 0., 0.],
                       [0., 0., 0., 5.82843137, 1., 0.], [0., 0., 0., 0., 5.82842725, 1.],
                       [0., 0., 0., 0., 0., 5.82842713]])
        pp = np.array(
            [[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())

    def test4(self):
        matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        uu = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        pp = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())

    def test5(self):
        matrix = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0.], [2., 1., 0.], [-1., -1., 1.]])
        uu = np.array([[2., 1., 1.], [0., -8., -2.], [0., 0., 1.]])
        pp = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())

    def test6(self):
        matrix = np.array([[0.2425, 0, 0.9701], [0, 0.2425, 0.9701], [0.2357, 0.2357, 0.9428]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0.], [0., 1., 0.], [0.97195876, 0.97195876, 1.]])
        uu = np.array([[0.2425, 0., 0.9701], [0., 0.2425, 0.9701], [0., 0., -0.94299439]])
        pp = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())

    def test7(self):
        matrix = np.array([[0.2425, 0, 0.9701], [0, 0.2425, 0.9701], [0.2357, 0.2357, 0.9428]])
        lu = faktoryzacja.facto_lu(matrix)
        L, U, P = lu
        ll = np.array([[1., 0., 0.], [0., 1., 0.], [0.97195876, 0.97195876, 1.]])
        uu = np.array([[0.2425, 0., 0.9701], [0., 0.2425, 0.9701], [0., 0., -0.94299439]])
        pp = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.assertEqual(L.all(), ll.all())
        self.assertEqual(U.all(), uu.all())
        self.assertEqual(P.all(), pp.all())
        self.assertEqual((np.dot(np.dot(P, L), U)).all(), matrix.all())


if __name__ == '__main__':
    unittest.main()
