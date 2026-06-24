/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *  
 * Matteo Cicuttin (C) 2020-2022
 * matteo.cicuttin@uliege.be
 *
 * University of Liège - Montefiore Institute
 * Applied and Computational Electromagnetics group  
 */
/*
 * Copyright (C) 2013-2016, Matteo Cicuttin - matteo.cicuttin@uniud.it
 * Department of Electrical Engineering, University of Udine
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Udine nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(s) ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR(s) BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifdef HAVE_MUMPS
#include <complex>

#ifdef HAVE_MUMPSS
#include <smumps_c.h>
#endif

#ifdef HAVE_MUMPSD
#include <dmumps_c.h>
#endif

#ifdef HAVE_MUMPSC
#include <cmumps_c.h>
#endif

#ifdef HAVE_MUMPSZ
#include <zmumps_c.h>
#endif

#define MUMPS_JOB_INIT              -1
#define MUMPS_JOB_TERMINATE         -2
#define MUMPS_JOB_ANALYZE            1
#define MUMPS_JOB_FACTORIZE          2
#define MUMPS_JOB_SOLVE              3
#define MUMPS_JOB_ANALYZE_FACTORIZE  4
#define MUMPS_JOB_FACTORIZE_SOLVE    5
#define MUMPS_JOB_EVERYTHING         6

#define MUMPS_OUTPUT_ERROR           1
#define MUMPS_OUTPUT_DIAG            2
#define MUMPS_OUTPUT_GLOBAL          4

/* This is typical Fortran madness: the
 * magic number -987654 is really required
 * to inform the mumps code to use the
 * MPI communicator MPI_COMM_WORLD
 */
#define FORTRAN_MADNESS_MAGIC       -987654

namespace disk::solvers {

namespace mumps_priv {

template<typename T>
struct mumps_types;

#ifdef HAVE_MUMPSS
template<>
struct mumps_types<float> {
    typedef SMUMPS_STRUC_C          MUMPS_STRUC_C;
    typedef float                   mumps_elem_t;
};

void
call_mumps(typename mumps_types<float>::MUMPS_STRUC_C *id)
{
    smumps_c(id);
}
#endif

#ifdef HAVE_MUMPSD
template<>
struct mumps_types<double> {
    typedef DMUMPS_STRUC_C          MUMPS_STRUC_C;
    typedef double                  mumps_elem_t;
};

void
call_mumps(typename mumps_types<double>::MUMPS_STRUC_C *id)
{
    dmumps_c(id);
}
#endif

#ifdef HAVE_MUMPSC
template<>
struct mumps_types<std::complex<float>> {
    typedef CMUMPS_STRUC_C          MUMPS_STRUC_C;
    typedef mumps_complex           mumps_elem_t;
};

void
call_mumps(typename mumps_types<std::complex<float>>::MUMPS_STRUC_C *id)
{
    cmumps_c(id);
}
#endif

#ifdef HAVE_MUMPSZ
template<>
struct mumps_types<std::complex<double>> {
    typedef ZMUMPS_STRUC_C          MUMPS_STRUC_C;
    typedef mumps_double_complex    mumps_elem_t;
};

void
call_mumps(typename mumps_types<std::complex<double>>::MUMPS_STRUC_C *id)
{
    zmumps_c(id);
}
#endif

template<typename From>
typename mumps_types<From>::mumps_elem_t *
mumps_cast_from(From *from)
{
    return reinterpret_cast<typename mumps_types<From>::mumps_elem_t *>(from);
}

} // namespace mumps_priv

template<typename T>
struct mumps_state
{
    using MUMPS_STRUC = typename mumps_priv::mumps_types<T>::MUMPS_STRUC_C;
    MUMPS_STRUC             id;
    size_t                  mflops;
    /* From the MUMPS documentation it is not clear which
     * should be the lifetime of Airn and Ajcn */
    std::vector<int>        Airn;
    std::vector<int>        Ajcn;

    std::vector<int>        irhs_sparse;
    std::vector<int>        irhs_ptr;
    std::vector<T>          rhs_sparse;

    bool fail = false;
};

template<typename T>
class mumps_solver
{
    using MUMPS_STRUC = typename mumps_priv::mumps_types<T>::MUMPS_STRUC_C;
    using mstate = mumps_state<T>;
    using mstate_uptr = std::unique_ptr<mstate>;

    mstate_uptr     state;
    int             symmetric_flag;
    int             parallel_flag;

public:
    mumps_solver()
    {
        state = std::make_unique<mstate>();
        state->id.job = MUMPS_JOB_INIT;
        state->id.par = 1;
        state->id.sym = 0;
        state->id.comm_fortran = FORTRAN_MADNESS_MAGIC;
        
        mumps_priv::call_mumps(&state->id);
        
        state->id.icntl[0]= -1;//6;    // Suppress error output
        state->id.icntl[1]= -1;//6;    // Suppress diagnostic output
        state->id.icntl[2]= -1;//6;    // Suppress global output
        state->id.icntl[3]= 2;         // Loglevel
    
        if (state->id.infog[0] != 0) {
            state->fail = true;
        }
    }
    
    ~mumps_solver()
    {
        state->id.job = MUMPS_JOB_TERMINATE;
        mumps_priv::call_mumps(&state->id);
    }
    
    template<int _Options, typename _Index>
    bool
    factorize(Eigen::SparseMatrix<T, _Options, _Index>& A) const
    {
        static_assert( !(_Options & Eigen::RowMajor), "CSR not supported yet.");
        if ( A.rows() != A.cols() )
            throw std::invalid_argument("Only square matrices");

        A.makeCompressed();

        int     N       = A.rows();
        T *     data    = A.valuePtr();
        int *   ia      = A.outerIndexPtr();
        int     js      = A.nonZeros();
        int *   ja      = A.innerIndexPtr();
        int     is      = A.innerSize();

        state->Airn.resize( A.nonZeros() );
        state->Ajcn.resize( A.nonZeros() );

        /* WARNING: here we deal with CSC and we want to convert
         * to COO. Therefore here we extract COLUMN indices. */
        for (int i = 0; i < is; i++) {
            int begin = ia[i];
            int end = ia[i+1];

            for (size_t j = begin; j < end; j++) {
                state->Ajcn[j] = i+1;
            }
        }

        /* And here we extract ROW indices */
        for (size_t i = 0; i < js; i++) {
            state->Airn[i] = ja[i] + 1;
        }
        
        state->id.a = mumps_priv::mumps_cast_from<T>(data);
        state->id.irn = state->Airn.data();
        state->id.jcn = state->Ajcn.data();
        state->id.n = A.rows();
        state->id.nz = A.nonZeros();

        state->id.job = MUMPS_JOB_ANALYZE_FACTORIZE;
        mumps_priv::call_mumps(&state->id);
        state->mflops = (size_t)((state->id.rinfog[0] + state->id.rinfog[1])/1e6);
        
        if (state->id.infog[0] != 0) {
            state->fail = true;
            return false;
        }
        
        return true;
    }

    /* Pseudo-compatibility with Eigen solver interface */
    template<int _Options, typename _Index>
    bool
    compute(Eigen::SparseMatrix<T, _Options, _Index>& A) const
    {
        return factorize(A);
    }

    template<int nrhs>
    Eigen::Matrix<T, Eigen::Dynamic, nrhs>
    solve(const Eigen::Matrix<T, Eigen::Dynamic, nrhs>& b) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, nrhs> ret = b;
        T* x = ret.data();

        state->id.nrhs = (nrhs > 0) ? nrhs : b.cols();
        state->id.rhs = mumps_priv::mumps_cast_from<T>(ret.data());
        /* -----> */ state->id.lrhs = state->id.n;
        
        state->id.job = MUMPS_JOB_SOLVE;
        mumps_priv::call_mumps(&state->id);

        if (state->id.infog[0] != 0) {
            state->fail = true;
        }

        return ret;
    }

    template<int _Options, typename _Index>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    solve(Eigen::SparseMatrix<T, _Options, _Index>& B) const
    {
        static_assert(!(_Options & Eigen::RowMajor), "RowMajor RHS not supported");

        B.makeCompressed();

        const int n    = state->id.n;
        const int nrhs = B.cols();

        if (B.rows() != n)
            throw std::invalid_argument("RHS dimension mismatch");

        // Enable sparse RHS mode
        state->id.icntl[19] = 1; // ICNTL(20) in Fortran indexing

        state->id.nrhs   = nrhs;
        state->id.lrhs   = n;
        state->id.nz_rhs = B.nonZeros();

        state->irhs_sparse.resize(B.nonZeros());
        state->irhs_ptr.resize(nrhs + 1);
        state->rhs_sparse.resize(B.nonZeros());

        int nnz = 0;

        for (int j = 0; j < nrhs; ++j)
        {
            state->irhs_ptr[j] = nnz + 1;

            for (typename Eigen::SparseMatrix<T, _Options, _Index>::InnerIterator it(B, j); it; ++it)
            {
                state->irhs_sparse[nnz] = it.row() + 1;
                state->rhs_sparse[nnz]  = it.value();
                nnz++;
            }
        }

        state->irhs_ptr[nrhs] = nnz + 1;

        // Attach to MUMPS
        state->id.irhs_sparse = state->irhs_sparse.data();
        state->id.irhs_ptr    = state->irhs_ptr.data();
        state->id.rhs_sparse  = mumps_priv::mumps_cast_from<T>(state->rhs_sparse.data());

        // Output will be dense
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X(n, nrhs);
        state->id.rhs = mumps_priv::mumps_cast_from<T>(X.data());

        state->id.job = MUMPS_JOB_SOLVE;
        mumps_priv::call_mumps(&state->id);

        // IMPORTANT: reset mode (avoid breaking next dense solve)
        state->id.icntl[19] = 0;
        state->id.irhs_sparse = nullptr;
        state->id.irhs_ptr    = nullptr;
        state->id.rhs_sparse  = nullptr;

        if (state->id.infog[0] != 0) {
            state->fail = true;
        }

        return X;
    }
    
    void set_output(int oflags)
    {
        state->id.icntl[0]= -1;//6     // Suppress error output
        state->id.icntl[1]= -1;//6     // Suppress diagnostic output
        state->id.icntl[2]= -1;//6;    // Suppress global output
        
        if (oflags & MUMPS_OUTPUT_ERROR)
            state->id.icntl[0] = 6;
        
        if (oflags & MUMPS_OUTPUT_DIAG)
            state->id.icntl[1] = 6;
        
        if (oflags & MUMPS_OUTPUT_GLOBAL)
            state->id.icntl[2] = 6;
    }
    
    int symmetric() const
    {
        return state->id.sym;
    }
    
    void symmetric(int flag)
    {
        state->id.sym = flag;
    }
    
    int parallel() const
    {
        return state->id.par;
    }
    
    void parallel(int flag)
    {
        state->id.par = flag;
    }

    size_t get_Mflops() const
    {
        return state->mflops;
    }

    bool failure() const {
        return state->fail;
    }
};

/*
template<typename T, int _Options, typename _Index, _Index nrhs>
Eigen::Matrix<T, Eigen::Dynamic, nrhs>
mumps_ldlt(Eigen::SparseMatrix<T, _Options, _Index>& A, Eigen::Matrix<T, Eigen::Dynamic, nrhs>& b)
{
    mumps_solver<T> solver;
    solver.symmetric(true);
    solver.factorize(A);
    return solver.solve(b);
}

template<typename T, int _Options, typename _Index, _Index nrhs>
Eigen::Matrix<T, Eigen::Dynamic, nrhs>
mumps_lu(Eigen::SparseMatrix<T, _Options, _Index>& A, Eigen::Matrix<T, Eigen::Dynamic, nrhs>& b)
{
    mumps_solver<T> solver;
    solver.factorize(A);
    return solver.solve(b);
}
*/

} // namespace disk::solvers

#endif /* HAVE_MUMPS */
