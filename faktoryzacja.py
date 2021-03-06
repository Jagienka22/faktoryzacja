import numpy as np
import sys


def pivot(matrix):
    n = np.shape(matrix)
    ID = np.identity(n[0])
    for i in range(n[0]):
        maxm = abs(matrix[i][i])
        row = i
        for j in range(i, n[0]):
            if abs(matrix[j][i]) > maxm:
                maxm = abs(matrix[j][i])
                row = j
        if i != row:
            tmp = np.copy(ID[i])
            ID[i] = ID[row]
            ID[row] = tmp
    return ID


def facto_lu(matrix):
    size_of_matrix = np.shape(matrix)
    n = size_of_matrix[0]
    U = np.zeros((n, n))
    L = np.identity(n)
    P = np.identity(n)
    for i in range(n):
        j = i
        while j < n:
            sum = 0
            for k in range(i):
                sum += (L[i][k] * U[k][j])
            U[i][j] = matrix[i][j] - sum
            if i == j and U[i][j] == 0.0:
                tmp_P = pivot(matrix)
                P = np.dot(P, tmp_P)
                matrix = np.dot(tmp_P, matrix)
            else:
                j += 1
        for j in range(i + 1, n):
            if i != j:
                sum = 0
                for k in range(i):
                    sum += (L[j][k] * U[k][i])
                L[j][i] = (matrix[j][i] - sum) / U[i][i]
    return L, U, P


def check_if(element, method):
    try:
        method(element)
    except ValueError:
        return False
    return True


def read_int():
    while True:
        size = input("Prosze podac rozmiar macierzy\n")
        if check_if(size, int):
            if int(size) > 0:
                return size
            else:
                print("Podana liczba jest mniejsza lub rowna zero")
        else:
            print("Podana wartosc nie jest liczba calkowita.")


def read_matrix(n):
    M = np.zeros((n, n))
    x = 0
    while x < n:
        print("Element w wierszu", x)
        m = input()
        if len(m.split()) != n:
            print("Zla liczba elementow w wierszu")
            x -= 1
        else:
            i = 0
            for y in m.split():
                if check_if(y, float):
                    M[x][i] = y
                    i += 1
                else:
                    print("Ktorys z elementow z wiersza nie jest liczba")
                    x -= 1
        x += 1
    if np.linalg.det(M) != 0:
        return M
    else:
        print("Nie da sie dokonac faktoryzacji dla tej macierzy, bo wyznacznik jest rowny zero (jest osobliwa)")
        sys.exit()


def start():
    while True:
        print("Wybierz sposob wprowadzania danych:")
        print("1: Wczytaj z pliku (Podaj sciezke)")
        print("2: Wpisz do terminala")
        print("0: Wyjdź")
        n = int(input())
        if n == 0:
            break
        if n == 1:
            path = input("Podaj nazwę pliku:  ")
            f = open(path, 'r')
            l = [[int(num) for num in line.split()] for line in f]
            print("Macierz z pliku")
            print(l)
            L, U, P = facto_lu(l)
            print("L:\n", L, "\nU:\n", U, "\nP:\n", P, "\n")
            print("Sprawdzenie PxLxU: \n", np.dot(np.dot(P, L), U))
            print("Poczatkowa macierz:\n", l, "\n")
        if n == 2:
            n = int(read_int())
            print("Prosze podac kolejno elementy macierzy odzielajac je spacja (np 0 1 2)")
            print("(w przypadku liczb dziesietnych uzywac kropki)")
            M = read_matrix(n)
            print("Wpisana macierz:\n", M, "\n")
            lu = facto_lu(M)
            L, U, P = lu
            print("L:\n", L, "\nU:\n", U, "\nP:\n", P, "\n")
            print("Sprawdzenie PxLxU: \n", np.dot(np.dot(P, L), U))
            print("Poczatkowa macierz:\n", M, "\n")


if __name__ == "__main__":
    start()
