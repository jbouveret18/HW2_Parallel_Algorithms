#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

class Matrix {
public:
    Matrix(int r, int c) : rows(r), columns(c) {
        matrix = new int*[rows];

        for (int i = 0; i < rows; ++i) {
            matrix[i] = new int[columns];
        }
    }

    ~Matrix() {
        for (int i = 0; i < rows; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
    void populateRandom() {
        srand(static_cast<unsigned int>(time(nullptr)));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                matrix[i][j] = rand() % 100;
            }
        }
    }
    void display() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    int rows;
    int columns;
    int **matrix;
};

void multiplyMatricesParallel(const Matrix &matrixA, const Matrix &matrixB, Matrix &matrixC, int numThreads, int caseNum) {
    if (matrixA.columns != matrixB.rows) {
        return;
    }

    omp_set_num_threads(numThreads);

    auto start_time = std::chrono::high_resolution_clock::now();

    if (caseNum == 1) {
#pragma omp parallel for
        for (int i = 0; i < matrixA.rows; ++i) {
            for (int j = 0; j < matrixB.columns; ++j) {
                for (int k = 0; k < matrixA.columns; ++k) {
                    matrixC.matrix[i][j] += matrixA.matrix[i][k] * matrixB.matrix[k][j];
                }
            }
        }
    } else if (caseNum == 2) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < matrixA.rows; ++i) {
            for (int j = 0; j < matrixB.columns; ++j) {
                for (int k = 0; k < matrixA.columns; ++k) {
                    matrixC.matrix[i][j] += matrixA.matrix[i][k] * matrixB.matrix[k][j];
                }
            }
        }
    } else if (caseNum == 3) {
#pragma omp parallel for collapse(3)
        for (int i = 0; i < matrixA.rows; ++i) {
            for (int j = 0; j < matrixB.columns; ++j) {
                for (int k = 0; k < matrixA.columns; ++k) {
                    matrixC.matrix[i][j] += matrixA.matrix[i][k] * matrixB.matrix[k][j];
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Time taken with " << numThreads << " threads for case " << caseNum << ": " << duration.count() << " seconds." << std::endl;
}

int main() {
    int numRows = 1000;
    int numCols = 1000;

    Matrix matrixA(numRows, numCols);
    Matrix matrixB(numCols, numRows);

    matrixA.populateRandom();
    matrixB.populateRandom();


    for (int numThreads = 1; numThreads <= 8; numThreads *= 2) {
        for (int caseNum = 1; caseNum <= 3; ++caseNum) {
            Matrix matrixC(numRows, numRows);
            multiplyMatricesParallel(matrixA, matrixB, matrixC, numThreads, caseNum);
        }
    }

    return 0;
}
