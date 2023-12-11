#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

//Part 1
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
//End of Part 1
//Part2
class SparseMatrix {
public:
    SparseMatrix(int r, int c, double *vals, int *cols, int *row_ptrs, int num_nonzeros)
            : rows(r), columns(c), values(vals), column_indices(cols), row_pointers(row_ptrs), nonzeros(num_nonzeros) {}

    void display() {
        int k = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                if (k < nonzeros && column_indices[k] == j && row_pointers[i] <= k && k < row_pointers[i + 1]) {
                    std::cout << values[k] << " ";
                    ++k;
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << std::endl;
        }
    }
    // Function to multiply the matrix with a vector
    std::vector<double> multiplyWithVector(const std::vector<double>& vector) {
        std::vector<double> result(rows, 0.0);
#pragma omp parallel
        for (int i = 0; i < rows; ++i) {
            for (int k = row_pointers[i]; k < row_pointers[i + 1]; ++k) {
                result[i] += values[k] * vector[column_indices[k]];
            }
        }

        return result;
    }

private:
    int rows;
    int columns;
    double *values;
    int *column_indices;
    int *row_pointers;
    int nonzeros;
};
void read_CSR_matrix(const char *filename, double **values, int **column_indices, int **row_pointers, int *num_rows, int *num_nonzeros) {
    FILE *file;
    char line[256];

    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    int M = 0, N = 0, nonzeros = 0;
    int header_read = 0;

    // Read and parse the header for matrix dimensions
    while (fgets(line, sizeof(line), file)) {
        if (header_read == 0) {
            if (sscanf(line, "%d %d %d", &M, &N, &nonzeros) == 3) {
                printf("Matrix Dimensions: M = %d, N = %d, Nonzeros = %d\n", M, N, nonzeros);
                header_read = 1;
                break; // Stop reading after finding the header
            }
        }
    }
    *values = (double *)malloc(nonzeros * sizeof(double));
    *column_indices = (int *)malloc(nonzeros * sizeof(int));
    *row_pointers = (int *)malloc((M + 1) * sizeof(int));

    int row = 0;
    int current_row = 0;
    int idx = 0;
    (*row_pointers)[row] = idx;

    while (fgets(line, sizeof(line), file) && idx < nonzeros) {
        int i, j;
        double value;
        if (sscanf(line, "%d %d %lf", &i, &j, &value) == 3) {
            if (i - 1 != current_row) {
                while (current_row < i - 1) {
                    (*row_pointers)[++current_row] = idx;
                }
                row++;
            }
            (*values)[idx] = value;
            (*column_indices)[idx] = j - 1;
            idx++;
        }
    }
    (*row_pointers)[M] = nonzeros;

    *num_rows = M;
    *num_nonzeros = nonzeros;

    fclose(file);
}
//End Of Part 2

int main() {
    printf("PART 1:\n\n");
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
    printf("\n\nPART 1 END\n\n");
    printf("PART 2:\n\n");
    const char *filename = "C:\\Users\\jbouv\\OneDrive\\Desktop\\Ecole\\ISEP\\Cycle_ingenieur\\A3\\ITU\\Classes\\Parallel Algorithms\\Homework\\HW2\\e40r5000.mtx"; // Replace with your file name
    double *values;
    int *column_indices;
    int *row_pointers;
    int num_rows, num_nonzeros;

    read_CSR_matrix(filename, &values, &column_indices, &row_pointers, &num_rows, &num_nonzeros);
    SparseMatrix sparse_matrix(num_rows, num_rows, values, column_indices, row_pointers, num_nonzeros);

    // Display the sparse matrix
    //sparse_matrix.display();

    int columns = 10000;

    // Generate a random vector
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::vector<double> randomVector(columns);
    for (int i = 0; i < columns; ++i) {
        randomVector[i] = static_cast<double>(std::rand() % 101); // Random values between 0 and 100
    }

    for (int num_threads = 1; num_threads <= 8; ++num_threads) {
        omp_set_num_threads(num_threads);
        double start_time = omp_get_wtime();
        std::vector<double> result = sparse_matrix.multiplyWithVector(randomVector);
        double end_time = omp_get_wtime();

        std::cout << "Time taken with " << num_threads << " thread(s): " << (end_time - start_time) << " seconds"
                  << std::endl;
    }
    free(values);
    free(column_indices);
    free(row_pointers);
    return 0;
}
