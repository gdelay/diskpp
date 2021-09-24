/*
 *       /\        Matteo Cicuttin (C) 2016-2021
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
 *
 * This file is copyright of the following authors:
 * Guillaume Delay  (C) 2021         guillaume.delay@sorbonne-universite.fr
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

/* this file deals with the unique continuation problem subject to the heat equation */
#include <iostream>
#include <regex>
#include <sstream>
#include <iomanip>

#include <unistd.h>

#include "loaders/loader.hpp"
#include "methods/hho"
#include "solvers/solver.hpp"
#include "output/silo.hpp"
#include "timecounter.h"
#include "colormanip.h"
#include "bases/bases.hpp"

using namespace disk;
using namespace Eigen;
using namespace std;



//////////////////   test case   ///////////////////
/* RHS definition */
template<typename Mesh>
struct rhs_functor;


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct rhs_functor< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        return 0.0;
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct rhs_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        return 0.0;
    }
};

template<typename Mesh>
auto make_rhs_function(const Mesh& msh)
{
    return rhs_functor<Mesh>();
}

/***************************************************************************/
/* Expected solution definition */
template<typename Mesh>
struct solution_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct solution_functor< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    T omega;

    solution_functor< Mesh<T, 1, Storage> >(const T omega_)
        : omega(omega_)
        {}

    scalar_type operator()(T t, const point_type& pt) const
    {
        int K = 1;
        // assert( N > omega );

        // return std::sin(M_PI*K*pt.x()) * std::sin(M_PI*K*pt.y());
        return std::exp(-M_PI*M_PI*K*K*t) * std::sin(M_PI*K*pt.x());
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct solution_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    T omega;

    solution_functor< Mesh<T, 2, Storage> >(const T omega_)
        : omega(omega_)
        {}

    scalar_type operator()(T t, const point_type& pt) const
    {
        int K = 1;
        // assert( N > omega );

        // return std::sin(M_PI*K*pt.x()) * std::sin(M_PI*K*pt.y());
        return std::exp(-2*M_PI*M_PI*K*K*t) * std::sin(M_PI*K*pt.x()) * std::sin(M_PI*K*pt.y());
    }
};

template<typename Mesh>
auto make_solution_function(const Mesh& msh, const typename Mesh::coordinate_type omega_)
{
    return solution_functor<Mesh>(omega_);
}

/***************************************************************************/
/* gradients of the expected solution */
template<typename Mesh>
struct grad_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct grad_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    grad_functor< Mesh<T, 2, Storage> >(const T omega_)
        : omega(omega_)
        {}

    T omega;

    auto operator()(const point_type& pt) const
    {
        int N = 3;
        Matrix<T, 1, 2> ret;
        // T coeff = omega / std::sqrt(2);
        // auto sin_cx = std::sin(coeff * pt.x());
        // auto sin_cy = std::sin(coeff * pt.y());
        // auto cos_cx = std::cos(coeff * pt.x());
        // auto cos_cy = std::cos(coeff * pt.y());

        T coeff = std::sqrt( N*N - omega*omega );
        T sinh = 0.5 * ( std::exp(coeff * pt.y()) - std::exp(-coeff * pt.y()) );
        T cosh = 0.5 * ( std::exp(coeff * pt.y()) + std::exp(-coeff * pt.y()) );

        // ret(0) = - coeff * sin_cx * cos_cy;
        // ret(1) = - coeff * cos_cx * sin_cy;

        ret(0) = N * std::cos(N*pt.x()) * sinh / coeff;
        ret(1) = std::sin(N*pt.x()) * cosh;
        return ret;
    }
};

template<typename Mesh>
auto make_grad_function(const Mesh& msh, const typename Mesh::coordinate_type omega_)
{
    return grad_functor<Mesh>(omega_);
}

/***************************************************************************/
/* domain varpi function */

template<typename Mesh>
struct varpi_functor;


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct varpi_functor< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        scalar_type ret;

        bool Ndom1 = (pt.x() >= 0.25) && (pt.x() <= 0.75);
        bool Ndom4 = !( (pt.x() >= 0.25) && (pt.x() <= 0.75) );
        if( Ndom1 )
            ret = 0.0;
        else
            ret = 1.0;

        return ret;
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct varpi_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        scalar_type ret;

        bool Ndom1 = (pt.x() >= 0.0) && (pt.x() <= 0.875) && (pt.y() >= 0.125) && (pt.y() <= 0.875);
        // bool Ndom2 = (pt.y() >= 0.125) && (pt.y() <= 0.875);
        // bool Ndom3 = (pt.x() >= 0.0) && (pt.x() <= 0.875) && (pt.y() >= 0.125);
        bool Ndom4 = !( (pt.x() >= 0.25) && (pt.x() <= 0.75) && (pt.y() <= 0.5) );
        if( Ndom1 )
            ret = 0.0;
        else
            ret = 1.0;

        return ret;
    }
};

template<typename Mesh>
auto make_varpi_function(const Mesh& msh)
{
    return varpi_functor<Mesh>();
}

/***************************************************************************/
/* domain B function */

template<typename Mesh>
struct B_functor;


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct B_functor< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        scalar_type ret;

        bool Ndom4 = false;
        if( Ndom4 )
            ret = 0.0;
        else
            ret = 1.0;

        return ret;
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct B_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        scalar_type ret;

        // bool Ndom1 = (pt.x() >= 0.0) && (pt.x() <= 0.125) && (pt.y() >= 0.125) && (pt.y() <= 0.875);
        // bool Ndom2 = (pt.y() >= 0.125) && (pt.y() <= 0.875) &&
        //     ( ((pt.x() >= 0.0) && (pt.x() <= 0.25)) || ((pt.x() >= 0.75) && (pt.x() <= 1.0)) );
        // bool Ndom3 = ((pt.x() >= 0.0) && (pt.x() <= 0.125) && (pt.y() >= 0.125)) ||
        //     ((pt.y() >= 0.875) && (pt.x() <= 0.875));
        bool Ndom4 = !( (pt.x() >= 0.125) && (pt.x() <= 0.875) && (pt.y() <= 0.875) );
        if( Ndom4 )
            ret = 0.0;
        else
            ret = 1.0;

        return ret;
    }
};


template<typename Mesh>
auto make_B_function(const Mesh& msh)
{
    return B_functor<Mesh>();
}

/////// test_info -> for error output
template<typename T>
class test_info {
public:
    test_info()
        {
            H1_Om = 0.0;
            L2_Om = 0.0;
            H1_B = 0.0;
            L2_B = 0.0;
            H1_z = 0.0;
        }
    T H1_Om; // H1-error in Omega
    T L2_Om; // L2-error in Omega
    T H1_B; // H1-error in B
    T L2_B; // L2-error in B
    T H1_z; // H1-error for the dual variable
};

////////////////////////////////////////////////////////////////////////////
////////////////////////   ASSEMBLERS  /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

template<typename Mesh>
class heat_UC_assembler
{
    using T = typename Mesh::coordinate_type;

    std::vector<size_t>     compress_table;
    std::vector<size_t>     expand_table;
    hho_degree_info         di;
    size_t                  time_degree;
    std::vector<Triplet<T>> triplets; //, triplets_MAT_RHS;

    size_t num_all_faces, num_dirichlet_faces, num_other_faces, num_cells, system_size, time_steps;

    class assembly_index
    {
        size_t idx;
        bool   assem;

      public:
        assembly_index(size_t i, bool as) : idx(i), assem(as) {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool
        assemble() const
        {
            return assem;
        }

        friend std::ostream&
        operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

  public:
    typedef dynamic_matrix<T> matrix_type;
    typedef dynamic_vector<T> vector_type;

    SparseMatrix<T> LHS; //, MAT_RHS;
    vector_type     RHS;

    heat_UC_assembler(const Mesh& msh, hho_degree_info hdi, size_t t_degree, size_t t_steps)
	: di(hdi), time_degree(t_degree), time_steps(t_steps)
    {
	auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool { return msh.is_boundary(fc); };

        num_all_faces       = msh.faces_size();
        num_dirichlet_faces = std::count_if(msh.faces_begin(), msh.faces_end(), is_dirichlet);
        num_other_faces     = num_all_faces - num_dirichlet_faces;
        num_cells = msh.cells_size();

        compress_table.resize(num_all_faces);
        expand_table.resize(num_other_faces);

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            const auto fc = *std::next(msh.faces_begin(), i);
            if (!is_dirichlet(fc))
            {
                compress_table.at(i)               = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
        size_t space_system_size = fbs * (num_other_faces + num_all_faces) + 2 * cbs * msh.cells_size();
        system_size = space_system_size * (time_degree + 1) * time_steps + cbs * (time_steps+1) * msh.cells_size();

        LHS = SparseMatrix<T>(system_size, system_size);
        // MAT_RHS = SparseMatrix<T>(system_size, system_size);
        RHS = vector_type::Zero(system_size);
        // RHS_F = vector_type::Zero(system_size);
    }

    // here the Dirichlet data are not taken into account
    // the rhs function is not time-dependent

    void
    assemble(const Mesh&                     msh,
             const typename Mesh::cell_type& cl,
             const size_t                    n_step,
             const matrix_type&              lhs,
             const vector_type&              rhs)
    {
	auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool { return msh.is_boundary(fc); };

        const auto fbs    = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs    = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fcs    = faces(msh, cl);
        const auto fcs_id = faces_id(msh, cl);
        // const auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        size_t loc_size = 2 * ( fcs.size() * fbs + cbs ) * (time_degree + 1) + 2*cbs;
        asm_map.reserve(loc_size);

        auto cell_offset = offset(msh, cl);
        // first degrees of freedom are the cell components of the primal variable
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step + i, true));

        // then face components of the primal variable (no Dirichlet BC)
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto face_offset     = fcs_id[face_i]; // offset(msh, fc);
            const auto face_LHS_offset = num_cells * cbs * (time_degree+1) * time_steps
                + face_offset * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step; // no Dirichlet BC so no compress table

            for (size_t i = 0; i < fbs*(time_degree+1); i++)
                asm_map.push_back(assembly_index(face_LHS_offset + i, true)); // no test on Dirichlet
        }

        // then cell components of the dual variable
        size_t dual_offset = num_cells * cbs * (time_degree+1) * time_steps + num_all_faces * fbs * (time_degree+1) * time_steps;
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(dual_offset + cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step + i, true));

        // then face components of the dual variable (with Dirichlet BC)
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = fcs_id[face_i]; // offset(msh, fc);
            const auto face_LHS_offset = dual_offset + num_cells * cbs * (time_degree+1) * time_steps
                + compress_table.at(face_offset) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step;

            const bool dirichlet = is_dirichlet(fc);

            for (size_t i = 0; i < fbs*(time_degree+1); i++)
                asm_map.push_back(assembly_index(face_LHS_offset + i, !dirichlet));
        }

        // then time-interface variable
        size_t tertial_offset = 2 * num_cells * cbs * (time_degree+1) * time_steps + (num_all_faces + num_other_faces) * fbs * (time_degree+1) * time_steps;
        for(int i=0; i<2*cbs; i++)
            asm_map.push_back(assembly_index(tertial_offset + cell_offset * cbs * (time_steps+1) + cbs * n_step + i, true));


        // no initial data for the moment !!
	// compute initial datum contribution to RHS
        // vector_type u0 = vector_type::Zero(loc_size);
        // u0.block(0,0,cbs,1) = project_function(msh, cl, di.cell_degree(), init_fun, di.cell_degree());
        // auto rhs_modif = mat_rhs * u0 + rhs;

        // complete from here ...
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if (asm_map[j].assemble())
                {
                    triplets.push_back(Triplet<T>(asm_map[i], asm_map[j], lhs(i, j)));
                    // triplets_MAT_RHS.push_back(Triplet<T>(asm_map[i], asm_map[j], mat_rhs(i, j)));
                    // do we need this ??
		}
                // Dirichlet not taken into account
                // else
                //     RHS(asm_map[i]) -= lhs(i, j) * dirichlet_data(j);
            }

            RHS(asm_map[i]) += rhs(i);
            // RHS_F(asm_map[i]) += rhs(i);
            // do we need this ??
        }
    } // assemble()

    template<typename Function>
    vector_type
    take_local_solution(const Mesh&                     msh,
                        const typename Mesh::cell_type& cl,
                        const size_t                    n_step,
                        const vector_type&              solution,
                        const Function&                 dirichlet_bf)
    {
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fcs = faces(msh, cl);
        const auto num_faces = fcs.size();

        vector_type ret = vector_type::Zero( (num_faces * fbs + cbs) * (time_degree + 1) );

        auto cell_offset = offset(msh, cl);

        // primal variable : cell components
        ret.block(0, 0, (time_degree+1)*cbs, 1)
            = solution.block(cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step, 0, (time_degree+1)*cbs, 1);


        // primal variable : face components (no BC)
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            const auto fc = fcs[face_i];

            // auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool { return msh.is_boundary(fc); };

            // const bool dirichlet = is_dirichlet(fc);

            // Dirichlet data not taken into account
            // if (dirichlet)
            // {
            // 	for(int l=0; l <= time_degree; l++)
            // 	    ret.block(cbs * (time_degree+1) + (face_i * time_degree + l) * fbs, 0, fbs, 1) =
            // 		project_function(msh, fc, di.face_degree(), dirichlet_bf, di.face_degree());
            // }

            const auto face_offset     = offset(msh, fc);
            const auto face_LHS_offset = num_cells * cbs * (time_degree+1) * time_steps
                + face_offset * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step; // no Dirichlet BC so no compress table

            ret.block(cbs * (time_degree+1) + face_i * (time_degree+1) * fbs, 0, (time_degree+1) * fbs, 1)
                = solution.block(face_LHS_offset, 0, fbs * (time_degree+1), 1);
        }

        // dual variable : cell components
        size_t dual_offset = num_cells * cbs * (time_degree+1) * time_steps + num_all_faces * fbs * (time_degree+1) * time_steps;
        ret.block((cbs + num_faces*fbs)*(time_degree+1), 0, (time_degree+1)*cbs, 1)
            = solution.block(dual_offset + cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step, 0, (time_degree+1)*cbs, 1);


        // dual variable : face components (BC)
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            const auto fc = fcs[face_i];

            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool { return msh.is_boundary(fc); };

            const bool dirichlet = is_dirichlet(fc);

            // Dirichlet data not taken into account
            // if (dirichlet)
            // {
            // 	for(int l=0; l <= time_degree; l++)
            // 	    ret.block(cbs * (time_degree+1) + (face_i * time_degree + l) * fbs, 0, fbs, 1) =
            // 		project_function(msh, fc, di.face_degree(), dirichlet_bf, di.face_degree());
            // }

            if(!dirichlet)
            {
                const auto face_offset     = offset(msh, fc);
                const auto face_SOL_offset = dual_offset + num_cells * cbs * (time_degree+1) * time_steps
                + compress_table.at(face_offset) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step;
		
                ret.block((2*cbs + num_faces*fbs)*(time_degree+1) + face_i * (time_degree+1) * fbs, 0, (time_degree+1) * fbs, 1)
                    = solution.block(face_SOL_offset, 0, fbs * (time_degree+1), 1);
            }
        }

        // tertial variable
        size_t tertial_offset = 2 * num_cells * cbs * (time_degree+1) * time_steps + (num_all_faces + num_other_faces) * fbs * (time_degree+1) * time_steps;
        ret.block(2*(cbs + num_faces*fbs)*(time_degree+1), 0, 2*cbs, 1)
            = solution.block(tertial_offset + cell_offset * cbs * (time_steps+1) + cbs * n_step, 0, 2*cbs, 1);

        return ret;
    }

    void
    finalize(void)
    {
        LHS.setFromTriplets(triplets.begin(), triplets.end());
        // MAT_RHS.setFromTriplets(triplets_MAT_RHS.begin(), triplets_MAT_RHS.end());
        triplets.clear();
        // triplets_MAT_RHS.clear();
    }

    size_t
    num_assembled_faces() const
    {
        return num_other_faces;
    }

};


template<typename Mesh>
auto
make_heat_UC_assembler(const Mesh& msh, const hho_degree_info& hdi, size_t time_degree, size_t time_steps)
{
    return heat_UC_assembler<Mesh>(msh, hdi, time_degree, time_steps);
}

//////////////////////////////////////////////////////


template<template<typename, size_t, typename> class Mesh,
         typename T, typename Storage>
void
export_to_silo(const Mesh<T, 2, Storage>& msh,
               const Matrix<T, Dynamic, 1>& data, int cycle = -1)
{
    disk::silo_database silo;

    if (cycle == -1)
        silo.create("heat.silo");
    else
    {
        std::stringstream ss;
        ss << "out_" << cycle << ".silo";
        silo.create(ss.str());
    }

    silo.add_mesh(msh, "mesh");

    silo.add_variable("mesh", "sol", data, disk::zonal_variable_t );
    silo.close();
}

///////////////////////////////////////

/////////////////////////////////////////////////////////////////
//////////////////  ASSEMBLY  ROUTINES //////////////////////////
/////////////////////////////////////////////////////////////////


/****************************************************************/
/************          Laplacian  terms          ****************/

template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_DGH_laplacian(const Mesh&                     msh,
                   const typename Mesh::cell_type& cl,
                   const hho_degree_info&          di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    // const auto graddeg = di.grad_degree();

    const auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
    // const auto gb = make_vector_monomial_basis(msh, cl, graddeg);

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);
    // const auto gbs = gb.size();

    const auto num_faces = howmany_faces(msh, cl);

    // const matrix_type gr_lhs = make_mass_matrix(msh, cl, gb);
    // matrix_type       gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    size_t loc_size = cbs + num_faces * fbs;
    matrix_type ret = matrix_type::Zero(loc_size, loc_size);

    int intdeg = 2*celdeg - 2;
    auto int_deg = std::max(intdeg , 0);
    const auto qps = integrate(msh, cl, int_deg);
    for (auto& qp : qps)
    {
        const auto grad_phi = cb.eval_gradients(qp.point());

        ret.block(0, 0, cbs, cbs) += qp.weight() * grad_phi * grad_phi.transpose();
    }

    const auto fcs = faces(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = normal(msh, cl, fc);
        const auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

        int intdeg2 = celdeg + facdeg - 1;
        auto int_deg2 = std::max(intdeg,0);
        const auto qps_f = integrate(msh, fc, int_deg2);
        for (auto& qp : qps_f)
        {
            const vector_type f_phi  = fb.eval_functions(qp.point());
            const auto c_phi         = cb.eval_functions(qp.point());
            const auto grad_phi      = cb.eval_gradients(qp.point());
            const vector_type qp_g_phi_n = grad_phi * (qp.weight() * n);

            ret.block(0, cbs + i * fbs, cbs, fbs) += qp_g_phi_n *  f_phi.transpose();
            ret.block(0, 0, cbs, cbs) -= qp_g_phi_n * c_phi.transpose();

            ret.block(cbs + i * fbs, 0, fbs, cbs) += f_phi * qp_g_phi_n.transpose();
            ret.block(0, 0, cbs, cbs) -= c_phi * qp_g_phi_n.transpose();
        }
    }

    return ret;
}

/****************************************************************/
/**********         Stabilization  terms          ***************/

// we compute hT^{-1} (uT - uF , vT - vF)_F
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_no_proj_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl,
                           const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);

    const auto num_faces = howmany_faces(msh, cl);
    const auto total_dofs = cbs + num_faces * fbs;

    matrix_type       ret = matrix_type::Zero(total_dofs, total_dofs);

    auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
    const auto fcs = faces(msh, cl);


    for (size_t i = 0; i < num_faces; i++)
    {
        const auto fc = fcs[i];
        auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

        // compute (uT - uF , vT - vF)_F
        const auto qps = integrate(msh, fc, 2*std::max(facdeg, celdeg));
        for (auto& qp : qps)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto f_phi = fb.eval_functions(qp.point());

            ret.block(0, 0, cbs, cbs) += qp.weight() * c_phi * c_phi.transpose();
            ret.block(0, cbs + i * fbs, cbs, fbs) -= qp.weight() * c_phi * f_phi.transpose();
            ret.block(cbs + i * fbs, 0, fbs, cbs) -= qp.weight() * f_phi * c_phi.transpose();
            ret.block(cbs + i * fbs, cbs + i * fbs, fbs, fbs)
                += qp.weight() * f_phi * f_phi.transpose();
        }
    }

    // scale with hT^{-1}
    const auto hT  = diameter(msh, cl);
    ret = (1.0/hT) * ret;

    return ret;
}

///////////////////////////////////////////////
/*
 * TODO : add gnuplot outputs for 1D and 2D
 */

template<typename Mesh>
bool
UC_heat_solver(const Mesh& msh, size_t degree, size_t time_steps, size_t time_degree)
{
    typedef typename Mesh::coordinate_type  scalar_type;
    typedef typename Mesh::point_type       point_type;
    using T = scalar_type;

    hho_degree_info hdi(degree, degree, degree+1);

    auto num_cells = msh.cells_size();
    auto num_faces = msh.faces_size();

    // auto assembler = make_heat_dG_assembler(msh, hdi, time_degree);
    auto assembler = make_heat_UC_assembler(msh, hdi, time_degree, time_steps);

    auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
    auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension-1);

    timecounter tc;

    int K = 1; // index for the exact solution
    scalar_type final_time = 2.;

    T omega = 1;

    auto rhs_fun = make_rhs_function(msh);
    auto sol_fun = make_solution_function(msh,omega);
    // auto sol_grad = make_grad_function(msh,omega);

    auto varpi_fun = make_varpi_function(msh);
    auto B_fun = make_B_function(msh);

    cout << "time_steps = " << time_steps << endl;
    scalar_type dt = final_time/time_steps;
    cout << "dt = " << dt << endl;
    disk::generic_mesh<T, 1>  time_mesh;
    disk::uniform_mesh_loader<T, 1> time_loader(0, final_time, time_steps);
    time_loader.populate_mesh(time_mesh);

    auto time_cell = *time_mesh.cells_begin();
    auto next_time_cell = *(++time_mesh.cells_begin());

    auto time_cb = make_scalar_monomial_basis(time_mesh, time_cell, time_degree);
    auto time_cb_next = make_scalar_monomial_basis(time_mesh, next_time_cell, time_degree);
    auto time_mass = make_mass_matrix(time_mesh, time_cell, time_cb);

    ////// time derivative term
    Matrix<T, Dynamic, Dynamic> time_deriv = Matrix<T, Dynamic, Dynamic>::Zero(time_degree+1, time_degree+1);
    auto qps_t = integrate(time_mesh, time_cell, 2*time_degree);
    for (auto& qp : qps_t)
    {
        auto phi   = time_cb.eval_functions( qp.point() );
        auto phi_t = time_cb.eval_gradients( qp.point() );
        time_deriv += qp.weight() * phi * phi_t.transpose();
    }
    
    ////// 1st jump term
    Matrix<T, Dynamic, Dynamic> time_loc = Matrix<T, Dynamic, Dynamic>::Zero(time_degree+1, time_degree+1);
    auto t_fcs = faces(time_mesh, time_cell);
    auto qps_f_t = integrate(time_mesh, t_fcs[0], 2*time_degree);
    for (auto& qp : qps_f_t)
    {
        auto phi   = time_cb.eval_functions( qp.point() );
        time_loc += qp.weight() * phi * phi.transpose();
    }

    ////// 2nd jump term
    Matrix<T, Dynamic, Dynamic> time_loc_bis = Matrix<T, Dynamic, Dynamic>::Zero(time_degree+1, time_degree+1);
    auto qps_f_t_bis = integrate(time_mesh, t_fcs[1], 2*time_degree);
    for (auto& qp : qps_f_t_bis)
    {
        auto phi_prev   = time_cb.eval_functions( qp.point() );
        auto phi_next   = time_cb_next.eval_functions( qp.point() );
        time_loc_bis += qp.weight() * phi_next * phi_next.transpose();
    }

    // min and max h_T
    T h_min = 1000, h_max = 0;
    for (auto& cl : msh)
    {
        const auto hT = diameter(msh, cl);
        if(hT < h_min) h_min = hT;
        if(hT > h_max) h_max = hT;
    }

    T gamma = 1.0;  // Tikhonov coefficient
    T dtl = 1., hk = 1.;
    for(int l = 0; l< time_degree; l++) dtl *= dt;
    for(int k=0; k < hdi.cell_degree(); k++) hk *= h_max;

    T Tikhonov_coeff = dtl * dt / sqrt(h_min) + hk + dtl * sqrt(dt) + hk * h_max / sqrt(dt);

    tc.tic();

    cout << "start assembly" << endl;
    for (auto& cl : msh)
    {
        auto fcs    = faces(msh, cl);
        auto num_faces = fcs.size();
        auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        // auto gr     = make_scalar_hho_laplacian(msh, cl, hdi);
        auto ah     = make_DGH_laplacian(msh, cl, hdi);
        // auto stab   = make_scalar_hho_stabilization(msh, cl, gr.first, hdi);
        auto stab   = make_no_proj_stabilization(msh, cl, hdi);
        auto mass   = make_mass_matrix(msh, cl, cb);
        // Matrix<scalar_type, Dynamic, Dynamic> ah = gr.second + stab;

        size_t loc_size = 2*ah.cols()*(time_degree+1) + 2*cbs;
        Matrix<scalar_type, Dynamic, Dynamic> lhs = Matrix<scalar_type, Dynamic, Dynamic>::Zero(loc_size, loc_size);

        /* Primal - Primal */
        Matrix<T, Dynamic, Dynamic> PP = stab;
        PP.block(0,0,cbs,cbs) += gamma * Tikhonov_coeff * mass.block(0,0,cbs,cbs);
        // add data assimilation term
        if( varpi_fun(barycenter(msh,cl)) > 0.5 )
        {
            // cout << "enter cell with data" << "..." << endl;
            PP.block(0,0,cbs,cbs) += mass;

            // TODO : add rhs
        }

        // fill lhs with the tensor product between time and space
        // cell - cell
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, l2*cbs, cbs, cbs) = PP.block(0, 0, cbs, cbs) * time_mass(l1,l2);
	// cell - face
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_j = 0; face_j < num_faces; face_j++)
                    lhs.block(l1*cbs, cbs*(time_degree+1) + face_j*(time_degree+1)*fbs + l2*fbs,
                              cbs, fbs) = PP.block(0, cbs+face_j*fbs, cbs, fbs) * time_mass(l1,l2);
        // face - cell
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_i = 0; face_i < num_faces; face_i++)
                    lhs.block(cbs*(time_degree+1) + face_i*(time_degree+1)*fbs + l1*fbs, l2*cbs,
                              fbs, cbs) = PP.block(cbs + face_i * fbs, 0, fbs, cbs) * time_mass(l1,l2);
        // face - face
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_i = 0; face_i < num_faces; face_i++)
		    for(size_t face_j = 0; face_j < num_faces; face_j++)
                        lhs.block(cbs*(time_degree+1) + face_i*(time_degree+1)*fbs + l1*fbs,
                                  cbs*(time_degree+1) + face_j*(time_degree+1)*fbs + l2*fbs,
                                  fbs, fbs) = PP.block(cbs + face_i*fbs, cbs + face_j*fbs, fbs, fbs*num_faces) * time_mass(l1,l2);

        /* Primal - Dual */
        size_t dual_offset = (cbs + num_faces * fbs) * (time_degree+1);
        // diffusion term
        // cell - cell
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, dual_offset + l2*cbs, cbs, cbs)
                    = ah.block(0, 0, cbs, cbs) * time_mass(l1,l2);

	// cell - face
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_j = 0; face_j < num_faces; face_j++)
                    lhs.block(l1*cbs, dual_offset + cbs*(time_degree+1) + face_j*(time_degree+1)*fbs + l2*fbs, cbs, fbs) = ah.block(0, cbs+face_j*fbs, cbs, fbs) * time_mass(l1,l2);

        // face - cell
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_i = 0; face_i < num_faces; face_i++)
                    lhs.block(cbs*(time_degree+1) + face_i*(time_degree+1)*fbs + l1*fbs, dual_offset + l2*cbs, fbs, cbs) = ah.block(cbs + face_i * fbs, 0, fbs, cbs) * time_mass(l1,l2);

        // face - face
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_i = 0; face_i < num_faces; face_i++)
		    for(size_t face_j = 0; face_j < num_faces; face_j++)
                        lhs.block(cbs*(time_degree+1) + face_i*(time_degree+1)*fbs + l1*fbs, dual_offset + cbs*(time_degree+1) + face_j*(time_degree+1)*fbs + l2*fbs, fbs, fbs)
                            = ah.block(cbs + face_i*fbs, cbs + face_j*fbs, fbs, fbs*num_faces) * time_mass(l1,l2);


        // derivative terms
        size_t tertial_offset = 2 * dual_offset;

        // (xi_T , d_t w_T)_{I_n x T}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, dual_offset + l2*cbs, cbs, cbs) += mass.block(0, 0, cbs, cbs) * time_deriv(l2,l1); // reversed l2 / l1 since derivative is on the test function

        // (w_T(t_{n-1}^+) , xi_T(t_{n-1}^+) )_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, dual_offset + l2*cbs, cbs, cbs) += mass.block(0, 0, cbs, cbs) * time_loc(l1,l2);

        // - (z^{n-1} , xi_T(t_{n-1}^+))_{Omega}
        for(size_t l2 = 0; l2 <= time_degree; l2++)
            lhs.block(tertial_offset, dual_offset + l2*cbs, cbs, cbs) -= mass.block(0, 0, cbs, cbs) * time_loc(0,l2);


        /* Dual - Primal */
        // obtained by symmetry
        lhs.block(dual_offset, 0, cbs*(time_degree+1), cbs*(time_degree+1))
            = lhs.block(0, dual_offset, cbs*(time_degree+1), cbs*(time_degree+1)).transpose();

        lhs.block(dual_offset, tertial_offset, cbs*(time_degree+1), cbs)
            = lhs.block(tertial_offset, dual_offset, cbs, cbs*(time_degree+1)).transpose();


        /* Dual - Dual */
        Matrix<T, Dynamic, Dynamic> sigma = stab;
        sigma.block(0,0,cbs,cbs) += mass;

        auto qps = integrate( msh, cl, 2*hdi.cell_degree() );
        for (auto& qp : qps)
        {
            // auto phi   = cb.eval_functions( qp.point() );
            auto g_phi = cb.eval_gradients( qp.point() );
            sigma.block(0,0,cbs,cbs) += qp.weight() * g_phi * g_phi.transpose();
        }

        // cell-cell
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(dual_offset + l1*cbs, dual_offset + l2*cbs, cbs, cbs)
                    -= sigma.block(0, 0, cbs, cbs) * time_mass(l1,l2);

        // cell - face
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_j = 0; face_j < num_faces; face_j++)
                    lhs.block(dual_offset + l1*cbs, dual_offset + cbs*(time_degree+1) + face_j*(time_degree+1)*fbs + l2*fbs, cbs, fbs) -= sigma.block(0, cbs+face_j*fbs, cbs, fbs) * time_mass(l1,l2);

        // face - cell
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_i = 0; face_i < num_faces; face_i++)
                    lhs.block(dual_offset + cbs*(time_degree+1) + face_i*(time_degree+1)*fbs + l1*fbs, dual_offset + l2*cbs, fbs, cbs) -= sigma.block(cbs + face_i * fbs, 0, fbs, cbs) * time_mass(l1,l2);

        // face - face
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                for(size_t face_i = 0; face_i < num_faces; face_i++)
		    for(size_t face_j = 0; face_j < num_faces; face_j++)
                        lhs.block(dual_offset + cbs*(time_degree+1) + face_i*(time_degree+1)*fbs + l1*fbs, dual_offset + cbs*(time_degree+1) + face_j*(time_degree+1)*fbs + l2*fbs, fbs, fbs)
                            -= sigma.block(cbs + face_i*fbs, cbs + face_j*fbs, fbs, fbs*num_faces) * time_mass(l1,l2);

        /* Jump penalization - previous interface */
        // primal - primal
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, l2*cbs, cbs, cbs)
                    += (1./dt) * mass * time_loc(l1,l2);

        // primal - tertial
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            lhs.block(l1*cbs, tertial_offset, cbs, cbs)
                -= (1./dt) * mass * time_loc(l1,0);

        // tertial - primal
        for(size_t l2 = 0; l2 <= time_degree; l2++)
            lhs.block(tertial_offset, l2*cbs, cbs, cbs)
                -= (1./dt) * mass * time_loc(l2,0);

        // tertial - tertial
        lhs.block(tertial_offset, tertial_offset, cbs, cbs)
            += (1./dt) * mass;


        /* Jump penalization - next interface */
        // primal - primal
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, l2*cbs, cbs, cbs)
                    += (1./dt) * mass * time_loc_bis(l1,l2);

        // primal - tertial
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            lhs.block(l1*cbs, tertial_offset + cbs, cbs, cbs)
                -= (1./dt) * mass * time_loc_bis(l1,0);

        // tertial - primal
        for(size_t l2 = 0; l2 <= time_degree; l2++)
            lhs.block(tertial_offset + cbs, l2*cbs, cbs, cbs)
                -= (1./dt) * mass * time_loc_bis(l2,0);

        // tertial - tertial
        lhs.block(tertial_offset + cbs, tertial_offset + cbs, cbs, cbs)
            += (1./dt) * mass;


        // at this point all the lhs terms have been implemented

        // Matrix<scalar_type, Dynamic, Dynamic> mat_rhs = Matrix<scalar_type, Dynamic, Dynamic>::Zero(ah.cols()*(time_degree+1), ah.cols()*(time_degree+1));
        // // jump term (cell - cell only)
        // for(size_t l1 = 0; l1 <= time_degree; l1++)
        //     for(size_t l2 = 0; l2 <= time_degree; l2++)
        //         mat_rhs.block(l1*cbs, l2*cbs, cbs, cbs) += mass.block(0, 0, cbs, cbs) * time_loc_bis(l1,l2);

        // the rhs is computed for constant functions (in time)
        Matrix<scalar_type, Dynamic, 1> rhs = Matrix<scalar_type, Dynamic, 1>::Zero(lhs.cols());
        auto space_rhs = make_rhs(msh, cl, cb, rhs_fun, 1);
        // TODO : check this vector ...
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            rhs.block(l1*cbs, 0, cbs, 1) = space_rhs * time_mass(l1,0);

        // loop on the time steps
        for(int step_i = 0; step_i < time_steps; step_i++) {
            // compute RHS -> TODO
            // compute data
            if( varpi_fun(barycenter(msh,cl)) > 0.5 ) {
                // compute (u , w_T)_{I_n x T}
                Matrix<scalar_type, Dynamic, 1> data_rhs
                    = Matrix<scalar_type, Dynamic, 1>::Zero( cbs * (time_degree+1) );
                const auto qps = integrate(msh, cl, 2*hdi.cell_degree());
                auto t_cell = *(time_mesh.cells_begin()+step_i);
                auto t_cb = make_scalar_monomial_basis(time_mesh, t_cell, time_degree);
                const auto qpst = integrate(time_mesh, t_cell , 2*time_degree);
                for(auto& qpt : qpst)
                {
                    auto t_phi   = t_cb.eval_functions( qpt.point() );
                    T time_point = qpt.point().x();

                    for(auto& qp : qps)
                    {
                        auto x_phi   = cb.eval_functions( qp.point() );
                        for(size_t l1 = 0; l1 <= time_degree; l1++)
                        {
                            data_rhs.block(l1*cbs, 0, cbs, 1) += qp.weight() * qpt.weight() * t_phi[l1] * sol_fun( time_point, qp.point() ) * x_phi;
                        }
                    }
                }
                
                rhs.block(0, 0, cbs*(time_degree+1), 1) = data_rhs;
            }

            assembler.assemble(msh, cl, step_i, lhs, rhs);
        }


    }
    cout << "end assembly loop" << endl;

    cout << "LHS.rows() = " << assembler.LHS.rows() << "  "
         << "LHS.cols() = " << assembler.LHS.cols() << endl;
    cout << "2xcbsx(l+1)xNcxNt + (Nf+Nfi)xfbsx(l+1)xNt + cbsxNcx(Nt+1) = "
         << 2*cbs*(time_degree+1)*num_cells*time_steps + (assembler.num_assembled_faces() + num_faces) * fbs * (time_degree+1) * time_steps + cbs * num_cells * (time_steps+1) << endl;

    assembler.finalize();

    auto LHS = assembler.LHS;
    // auto MAT_RHS = assembler.MAT_RHS;
    auto RHS = assembler.RHS;
    // auto RHS_F = assembler.RHS_F;

    cout << "LHS.rows() = " << LHS.rows() << endl;

    tc.toc();
    std::cout << " Assembly time: " << tc << std::endl;

    /* Solving linear system */
    cout << "RHS.norm() = " << RHS.norm() << endl;
    tc.tic();
    Matrix<scalar_type, Dynamic, 1> u;
    disk::solvers::pardiso_params<scalar_type> pparams;
    mkl_pardiso(pparams, LHS, RHS, u);
    tc.toc();
    
    std::cout << " Solving time: " << tc << std::endl;

    cout << "u.norm() = " << u.norm() << endl;
    
    scalar_type t = 0.0;
    // scalar_type dt = final_time/time_steps;
    scalar_type L2H1_error = 0.;

    size_t freq_exp = 1;

    /* Post-processing */
    // time loop
    for(size_t step_i = 0; step_i<time_steps; step_i++)
    {
        if(step_i % freq_exp == 0)
            std::cout << "Step " << step_i << std::endl;

        Matrix<scalar_type, Dynamic, 1> sol_silo = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());

#if false
        if(step_i % freq_exp == 0)
        {
            // be careful : this exports a very approximated solution
            // TODO : modify this export ...
            for (size_t i = 0; i < msh.cells_size(); i++)
                sol_silo(i) = u(i*cbs*(time_degree+1)*time_steps + cbs*(time_degree+1) * step_i);
            export_to_silo( msh, sol_silo, step_i );
        }
#endif

        auto t_cell = *(time_mesh.cells_begin()+step_i);
        auto t_cb = make_scalar_monomial_basis(time_mesh, t_cell, time_degree);
        const auto qpst = integrate(time_mesh, t_cell , 2*time_degree);

        //     // update RHS with the previous solution
        //     RHS = RHS_F + MAT_RHS * u;
        size_t cell_i = 0;
        for (auto& cl : msh)
        {
            auto cb = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());

            /* compute the L2 projection in space and time */
            Matrix<scalar_type, Dynamic, 1> rhs_proj
                = Matrix<scalar_type, Dynamic, 1>::Zero( cbs * (time_degree+1) );
            const auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for(auto& qpt : qpst)
            {
                auto t_phi   = t_cb.eval_functions( qpt.point() );
                T time_point = qpt.point().x();

                for(auto& qp : qps)
                {
                    auto x_phi   = cb.eval_functions( qp.point() );
                    for(size_t l1 = 0; l1 <= time_degree; l1++)
                    {
                        rhs_proj.block(l1*cbs, 0, cbs, 1) += qp.weight() * qpt.weight() * t_phi[l1] * sol_fun( time_point, qp.point() ) * x_phi;
                    }
                }
            }

            // use the mass matrix to compute the coordinates of the projection
            auto cell_mass   = make_mass_matrix(msh, cl, cb);
            Matrix<scalar_type, Dynamic, Dynamic> mass_matrix
                = Matrix<scalar_type, Dynamic, Dynamic>::Zero(cbs*(time_degree+1) , cbs*(time_degree+1));
            for(size_t l1 = 0; l1 <= time_degree; l1++)
            {
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                {
                    mass_matrix.block(l1*cbs, l2*cbs, cbs, cbs) = time_mass(l1,l2) * cell_mass;
                }
            }
            Matrix<scalar_type, Dynamic, 1> proj
                = Matrix<scalar_type, Dynamic, 1>::Zero( cbs * (time_degree+1) );
            LLT< Matrix<scalar_type, Dynamic, Dynamic> > mat_llt;
            mat_llt.compute(mass_matrix);
            proj = mat_llt.solve(rhs_proj);


            /* compute L2-H1-error of the current time step */
            Matrix<scalar_type, Dynamic, 1> diff
                = u.block(cell_i*cbs*(time_degree+1)*time_steps + cbs * (time_degree+1) * step_i, 0, cbs*(time_degree+1), 1) - proj;

            Matrix<scalar_type, Dynamic, Dynamic> grad_matrix
                = Matrix<scalar_type, Dynamic, Dynamic>::Zero(cbs , cbs);
            for(auto& qp : qps)
            {
                const auto g_phi = cb.eval_gradients( qp.point() );
                grad_matrix += qp.weight() * g_phi * g_phi.transpose();
            }


            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    for(size_t I = 0; I < cbs; I++)
                        for(size_t J = 0; J < cbs; J++)
                        {
                            L2H1_error += diff(l1 * cbs + I) * diff(l2 * cbs + J) * time_mass(l1,l2) * grad_matrix(I,J);
                        }

            cell_i++;
        }
        t += dt;
    } // time loop

    cout << "final time = " << t << endl;
    std::cout << "L2-H1-error = " << std::sqrt(L2H1_error) << std::endl;

    return true;
}

/* run main with :
   ./heat_dG -m ../../../diskpp/meshes/2D_quads/diskpp/testmesh-16-16.quad -k 1 -N 8 -l 0
*/


int main(int argc, char **argv)
{
    using T = double;
    // disk::cartesian_mesh<T, 2> msh;

    size_t      degree = 1;
    size_t      time_degree = 0;
    size_t      N = 8;
    char *      mesh_filename = nullptr;
    size_t      num_elems = 16; // for 1D only
    int ch;
    while ( (ch = getopt(argc, argv, "k:m:N:M:l:")) != -1 )
    {
	switch(ch)
        {
            case 'k':
                degree = std::stoi(optarg);
                break;

            case 'm':
                mesh_filename = optarg;
                break;

            case 'N':
                N = std::stoi(optarg);
                break;

            case 'M':
                num_elems = std::stoi(optarg);
                break;

            case 'l':
                time_degree = std::stoi(optarg);
                break;

            default:
                std::cout << "Invalid option" << std::endl;
                return 1;
	}
    }

    // msh = load_cartesian_2d_mesh<T>(mesh_filename);


    if (mesh_filename == nullptr)
    {
        std::cout << "Mesh format: 1D uniform" << std::endl;

        typedef disk::generic_mesh<T, 1>  mesh_type;

        mesh_type msh;
        disk::uniform_mesh_loader<T, 1> loader(0, 1, num_elems);
        loader.populate_mesh(msh);
        std::cout << "1D Mesh loaded ..." << std::endl;

        UC_heat_solver(msh, degree, N, time_degree);

        return 0;
    }

    auto msh = disk::load_fvca5_2d_mesh<T>(mesh_filename);
    std::cout << "2D Mesh loaded ..." << std::endl;
    UC_heat_solver(msh, degree, N, time_degree);
    return 0;
}

