#include <iostream>
#include "diskpp/common/eigen.hpp"
#include "diskpp/solvers/direct_solvers.hpp"

int main(void)
{
    static const int size = 5;
    using T = double;
    disk::sparse_matrix<T> A = disk::sparse_matrix<T>(size,size);
    disk::dynamic_vector<T> b = disk::dynamic_vector<T>::Zero(size);

    std::vector<disk::triplet<T>> trips;

    for (int i = 0; i < size; i++) {
        trips.push_back({i, i, 10.0*(i+1)});
        b(i) = i+1;
    } 
    trips.push_back({2, 0, 1.0});
    //trips.push_back({0, 2, 1.0});
    A.setFromTriplets(trips.begin(), trips.end());


    int     N       = A.rows();
    T *     data    = A.valuePtr();
    int *   ia      = A.outerIndexPtr();
    int     js      = A.nonZeros();
    int *   ja      = A.innerIndexPtr();
    int     is      = A.innerSize();

    std::vector<int> Airn( A.nonZeros() );
    std::vector<int> Ajcn( A.nonZeros() );

    /* Convert CSC to COO */
    for (int i = 0; i < is; i++) {
        int begin = ia[i];
        int end = ia[i+1];

        for (size_t j = begin; j < end; j++) {
            Airn[j] = i+1;
        }
    }

    //state->Aja.resize( A.nonZeros() );
    for (size_t i = 0; i < js; i++) {
        Ajcn[i] = ja[i] + 1;
    }

    for (size_t i = 0; i < A.nonZeros(); i++) {
        std::cout << Airn[i] << " " << Ajcn[i];
        std::cout << " " << data[i] << "\n";
    }

    disk::dynamic_vector<T> x;
    disk::solvers::sparse_lu(A, b, x);
    std::cout << x.transpose() << std::endl;
    disk::dynamic_vector<T> r = b - A*x;
    std::cout << r.transpose() << std::endl;

    if (r.norm() > 1e-15)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}