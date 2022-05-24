/*
 *       /\        Matteo Cicuttin (C) 2016-2021
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
 *
 * This file is copyright of the following authors:
 * Guillaume Delay  (C) 2021, 2022         guillaume.delay@sorbonne-universite.fr
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

    scalar_type operator()(const T t, const point_type& pt) const
    {
        // return 0.0;
        // return M_PI*M_PI*std::sin( M_PI * pt.x() );
        return ( M_PI*M_PI*std::cos(t) - std::sin(t) ) * std::sin( M_PI * pt.x() );
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct rhs_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const T t, const point_type& pt) const
    {
        return ( 2*M_PI*M_PI*std::cos(t) - std::sin(t) ) * std::sin( M_PI * pt.x() )
            * std::sin( M_PI * pt.y() );
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

    solution_functor< Mesh<T, 1, Storage> >()
        {}

    scalar_type operator()(T t, const point_type& pt) const
    {
        return std::sin( M_PI * pt.x() ) * std::cos(t);
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct solution_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    solution_functor< Mesh<T, 2, Storage> >()
        {}

    scalar_type operator()(T t, const point_type& pt) const
    {
        return std::cos(t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    }
};

template<typename Mesh>
auto make_solution_function(const Mesh& msh)
{
    return solution_functor<Mesh>();
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
        if( Ndom4 )
            ret = 0.0;
        else
            ret = 1.0;

        // return 1.;
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

        // bool Ndom4 = true;
        bool Ndom4 = (pt.x() >= 0.125) && (pt.x() <= 0.875);
        if( Ndom4 )
            ret = 1.0;
        else
            ret = 0.0;

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
        bool Ndom4 = (pt.x() >= 0.125) && (pt.x() <= 0.875) && (pt.y() >= 0.125) && (pt.y() <= 0.875);
        if( Ndom4 )
            ret = 1.0;
        else
            ret = 0.0;

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
    test_info() : H1_Om(0.) , L2_Om(0.) , H1_B(0.) , L2_B(0.) , H1_z(0.) {}
    T H1_Om; // H1-error in Omega
    T L2_Om; // L2-error in Omega
    T H1_B; // H1-error in B
    T L2_B; // L2-error in B
    T H1_z; // H1-error for the dual variable
};

// B function for the time domain
template<typename T>
bool time_B(T time) {
    T eps = 0.2;
    T fin_time = 2.;

    return ( eps < time ) && ( time < fin_time - eps );
}

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
    bool                    BC_known;

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

    // BC_known : true if we know the Dirichlet values of the solution, false otherwise
    heat_UC_assembler(const Mesh& msh, hho_degree_info hdi, size_t t_degree, size_t t_steps, bool BC_known=false)
	: di(hdi), time_degree(t_degree), time_steps(t_steps), BC_known(BC_known)
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
        size_t sol_faces = num_all_faces;
        if(BC_known) sol_faces = num_other_faces;

        size_t space_system_size = fbs * (num_other_faces + sol_faces) + 2 * cbs * msh.cells_size();
        system_size = space_system_size * (time_degree + 1) * time_steps;

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
        size_t loc_size = 2 * ( fcs.size() * fbs + cbs ) * (time_degree + 1);
        asm_map.reserve(loc_size);

        auto cell_offset = offset(msh, cl);
        // first degrees of freedom are the cell components of the primal variable
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step + i, true));

        // then face components of the primal variable
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto face_offset     = fcs_id[face_i]; // offset(msh, fc);
            auto face_LHS_offset = num_cells * cbs * (time_degree+1) * time_steps;
            if(BC_known)
                face_LHS_offset += compress_table.at(face_offset) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step; // compress table
            else
                face_LHS_offset += face_offset * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step; // no Dirichlet BC so no compress table

            if(BC_known)
            {
                const auto fc = fcs[face_i];
                const bool dirichlet = is_dirichlet(fc);

                for (size_t i = 0; i < fbs*(time_degree+1); i++)
                    asm_map.push_back(assembly_index(face_LHS_offset + i, !dirichlet));
            }
            else
            {
                for (size_t i = 0; i < fbs*(time_degree+1); i++)
                    asm_map.push_back(assembly_index(face_LHS_offset + i, true)); // no test on Dirichlet (no BC)
            }
        }

        // then cell components of the dual variable
        size_t num_sol_faces = num_all_faces;
        if( BC_known ) num_sol_faces = num_other_faces;

        size_t dual_offset = num_cells * cbs * (time_degree+1) * time_steps + num_sol_faces * fbs * (time_degree+1) * time_steps;
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

        // no initial data for the moment !!
	// compute initial datum contribution to RHS
        // vector_type u0 = vector_type::Zero(loc_size);
        // u0.block(0,0,cbs,1) = project_function(msh, cl, di.cell_degree(), init_fun, di.cell_degree());
        // auto rhs_modif = mat_rhs * u0 + rhs;

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

    /*
     * lhs must have the size of two time cells
     * n_step is between 1 and time_steps-1 (both included)
     * the faces are not taken into account (the time coupling occurs through space cells only)
     */
    void
    add_time_coupling(const Mesh&                     msh,
                      const typename Mesh::cell_type& cl,
                      const size_t                    n_step,
                      const matrix_type&              lhs)
    {
        const auto cbs    = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fbs    = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);

        std::vector<assembly_index> asm_map;
        size_t loc_size = 4 * cbs * (time_degree + 1);
        /*
         * degrees of freedom (cell variables) :
         * - primal variable (previous time step)
         * - dual variable (previous time step)
         * - primal variable (next time step)
         * - dual variable (next time step)
         */
        asm_map.reserve(loc_size);

        /* previous time step */
        auto cell_offset = offset(msh, cl);
        size_t num_sol_faces = num_all_faces;
        if( BC_known ) num_sol_faces = num_other_faces;
        size_t dual_offset = num_cells * cbs * (time_degree+1) * time_steps + num_sol_faces * fbs * (time_degree+1) * time_steps;

        // cell components of the primal variable (previous time step)
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * (n_step-1) + i, true));

        // cell components of the dual variable (previous time step)
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(dual_offset + cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * (n_step-1) + i, true));

        /* next time step */
        // cell components of the primal variable (next time step)
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step + i, true));

        // cell components of the dual variable (next time step)
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
            asm_map.push_back(assembly_index(dual_offset + cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step + i, true));

        /* add the triplets */
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if (asm_map[j].assemble())
                {
                    triplets.push_back(Triplet<T>(asm_map[i], asm_map[j], lhs(i, j)));
		}
            }
        }
    }

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

        vector_type ret = vector_type::Zero( 2*(num_faces * fbs + cbs) * (time_degree + 1) );

        auto cell_offset = offset(msh, cl);

        // primal variable : cell components
        ret.block(0, 0, (time_degree+1)*cbs, 1)
            = solution.block(cell_offset * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step, 0, (time_degree+1)*cbs, 1);


        // primal variable : face components
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            const auto fc = fcs[face_i];


            // Dirichlet data not taken into account
            // if (dirichlet)
            // {
            // 	for(int l=0; l <= time_degree; l++)
            // 	    ret.block(cbs * (time_degree+1) + (face_i * time_degree + l) * fbs, 0, fbs, 1) =
            // 		project_function(msh, fc, di.face_degree(), dirichlet_bf, di.face_degree());
            // }

            const auto face_offset     = offset(msh, fc);
            auto face_LHS_offset = num_cells * cbs * (time_degree+1) * time_steps;
            if(BC_known) face_LHS_offset += compress_table.at(face_offset) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step; // compress table
            else face_LHS_offset += face_offset * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step; // no Dirichlet BC so no compress table


            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool { return msh.is_boundary(fc); };

            const bool dirichlet = is_dirichlet(fc);

            if(BC_known && dirichlet)
                ret.block(cbs * (time_degree+1) + face_i * (time_degree+1) * fbs, 0, (time_degree+1) * fbs, 1)
                    = vector_type::Zero((time_degree+1) * fbs);
            else
                ret.block(cbs * (time_degree+1) + face_i * (time_degree+1) * fbs, 0, (time_degree+1) * fbs, 1)
                    = solution.block(face_LHS_offset, 0, fbs * (time_degree+1), 1);
        }

        // dual variable : cell components
        size_t num_sol_faces = num_all_faces;
        if(BC_known) num_sol_faces = num_other_faces;
        size_t dual_offset = num_cells * cbs * (time_degree+1) * time_steps + num_sol_faces * fbs * (time_degree+1) * time_steps;
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
make_heat_UC_assembler(const Mesh& msh, const hho_degree_info& hdi, size_t time_degree, size_t time_steps, bool BC_known = false)
{
    return heat_UC_assembler<Mesh>(msh, hdi, time_degree, time_steps, BC_known);
}

//////////////////////////////////////////////////////

template<template<typename, size_t, typename> class Mesh,
         typename T, typename Storage>
void
export_to_silo(const Mesh<T, 1, Storage>& msh,
               const Matrix<T, Dynamic, 1>& data, const Matrix<T, Dynamic, 1>& varpi,
               const Matrix<T, Dynamic, 1>& B, int cycle = -1)
{
    std::stringstream ss_data, ss_varpi, ss_B;
    ss_varpi << "varpi.txt";
    ss_B << "B.txt";
    if(cycle == -1)
    {
        ss_data << "sol.txt";
    }
    else
    {
        ss_data << "out_data_" << cycle << ".txt";
    }


    std::ofstream data_file(ss_data.str(), std::ios::out | std::ios::trunc);
    if(!data_file)
        std::cerr << "error opening file !!" << std::endl;

    std::ofstream varpi_file(ss_varpi.str(), std::ios::out | std::ios::trunc);
    if(!varpi_file)
        std::cerr << "error opening file !!" << std::endl;

    std::ofstream B_file(ss_B.str(), std::ios::out | std::ios::trunc);
    if(!B_file)
        std::cerr << "error opening file !!" << std::endl;


    int cell_i = 0;
    for(auto& cl : msh)
    {
        auto x = barycenter(msh, cl).x();

        data_file << x << "  " << data[cell_i] << std::endl;
        varpi_file << x << "  " << varpi[cell_i] << std::endl;
        B_file << x << "  " << B[cell_i] << std::endl;
        cell_i++;
    }

    data_file.close();
    varpi_file.close();
    B_file.close();



    //// For tests with the exact sol (modif N)
    std::stringstream ss_ex_sol;
    if(cycle == -1)
    {
        ss_ex_sol << "ex_sol.txt";
    }
    else
    {
        ss_ex_sol << "ex_sol_" << cycle << ".txt";
    }


    std::ofstream ex_sol_file(ss_ex_sol.str(), std::ios::out | std::ios::trunc);
    if(!ex_sol_file)
        std::cerr << "error opening file !!" << std::endl;

    auto sol_fun = make_solution_function(msh);

    int N = 32;
    double t = (2./N) * (cycle+0.5);

    cell_i = 0;
    for(auto& cl : msh) {
        auto x = barycenter(msh, cl).x();
        auto sol = sol_fun(t , barycenter(msh, cl));

        ex_sol_file << x << "  " << sol << std::endl;
        cell_i++;
    }

    ex_sol_file.close();
}

/////////////////////////

template<template<typename, size_t, typename> class Mesh,
         typename T, typename Storage>
void
export_to_silo(const Mesh<T, 2, Storage>& msh,
               const Matrix<T, Dynamic, 1>& data, const Matrix<T, Dynamic, 1>& varpi,
               const Matrix<T, Dynamic, 1>& B, int cycle = -1)
{
    disk::silo_database silo;

    if (cycle == -1)
        silo.create("UC_heat.silo");
    else
    {
        std::stringstream ss;
        ss << "out_" << cycle << ".silo";
        silo.create(ss.str());
    }

    silo.add_mesh(msh, "mesh");

    silo.add_variable("mesh", "sol", data, disk::zonal_variable_t );
    silo.add_variable("mesh", "varpi", varpi, disk::zonal_variable_t );
    silo.add_variable("mesh", "B", B, disk::zonal_variable_t );
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

template<typename Mesh>
test_info<typename Mesh::coordinate_type>
UC_heat_solver(const Mesh& msh, size_t degree, size_t time_steps, size_t time_degree)
{
    typedef typename Mesh::coordinate_type  scalar_type;
    typedef typename Mesh::point_type       point_type;
    using T = scalar_type;

    hho_degree_info hdi(degree, degree, degree+1);

    auto num_cells = msh.cells_size();
    auto nb_tot_faces = msh.faces_size();

    auto assembler = make_heat_UC_assembler(msh, hdi, time_degree, time_steps, false);

    auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
    auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension-1);

    timecounter tc;

    scalar_type final_time = 2.;

    auto rhs_fun = make_rhs_function(msh);
    auto sol_fun = make_solution_function(msh);

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
        time_loc_bis += qp.weight() * phi_prev * phi_prev.transpose();
    }

    ////// 3rd jump term
    Matrix<T, Dynamic, Dynamic> time_loc_cross = Matrix<T, Dynamic, Dynamic>::Zero(time_degree+1, time_degree+1);
    auto qps_f_t_cross = integrate(time_mesh, t_fcs[1], 2*time_degree);
    for (auto& qp : qps_f_t_cross)
    {
        auto phi_prev   = time_cb.eval_functions( qp.point() );
        auto phi_next   = time_cb_next.eval_functions( qp.point() );
        time_loc_cross += qp.weight() * phi_prev * phi_next.transpose();
    }

    // min and max h_T
    T h_min = 1000, h_max = 0;
    for (auto& cl : msh)
    {
        const auto hT = diameter(msh, cl);
        if(hT < h_min) h_min = hT;
        if(hT > h_max) h_max = hT;
    }

    T gamma = 1.e-3;  // Tikhonov coefficient
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
        auto ah     = make_DGH_laplacian(msh, cl, hdi);
        auto stab   = make_no_proj_stabilization(msh, cl, hdi);
        auto mass   = make_mass_matrix(msh, cl, cb);
        auto stiff   = make_stiffness_matrix(msh, cl, cb);

        size_t loc_size = 2*ah.cols()*(time_degree+1);

        /*** Matrix lhs for the terms contained in a cell an a time step ***/
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

        // diffusion term (primal - dual)
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
                            = ah.block(cbs + face_i*fbs, cbs + face_j*fbs, fbs, fbs) * time_mass(l1,l2);

        // the (dual - primal) component is obtained by symmetry (see further)

        // derivative term
        // (xi_T , d_t w_T)_{I_n x T}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                lhs.block(l1*cbs, dual_offset + l2*cbs, cbs, cbs) += mass * time_deriv(l2,l1); // reversed l2 / l1 since the derivative is on the test function


        /* the time coupling terms are taken into account later */

        /* Dual - Primal */
        // obtained by symmetry
        lhs.block(dual_offset, 0, dual_offset, dual_offset)
            = lhs.block(0, dual_offset, dual_offset, dual_offset).transpose();

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
                            -= sigma.block(cbs + face_i*fbs, cbs + face_j*fbs, fbs, fbs) * time_mass(l1,l2);


        // at this point all the lhs terms have been implemented

        /*** Matrix coupling for the terms coupling two time steps ***/
        size_t coupling_size = 4 * cbs * (time_degree+1); // we consider the cell dof on two time steps
        Matrix<scalar_type, Dynamic, Dynamic> coupling = Matrix<scalar_type, Dynamic, Dynamic>::Zero(coupling_size, coupling_size);


        /* coupling coming from the time derivative */
        size_t ts_c = coupling_size/2; // time step coupling
        size_t ts_dual = coupling_size/4;
        // (w_T(t_{n-1}^+) , xi_T(t_{n-1}^+) )_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(ts_c + l1*cbs, ts_c + ts_dual + l2*cbs, cbs, cbs) += mass * time_loc(l1,l2);

        // - (w_T(t_{n-1}^-) , xi_T(t_{n-1}^+))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(l1*cbs, ts_c + ts_dual + l2*cbs, cbs, cbs) -= mass * time_loc_cross(l1,l2);

        // symmetric terms
        coupling.block(ts_c + ts_dual, ts_c, ts_dual, ts_dual)
            = coupling.block(ts_c, ts_c + ts_dual, ts_dual, ts_dual).transpose();
        coupling.block(ts_c + ts_dual, 0, ts_dual, ts_dual)
            = coupling.block(0, ts_c + ts_dual, ts_dual, ts_dual).transpose();

        /* coupling coming from the time jump penalization */
        T jump_coeff = 1.; // 1./dt;
        // (v_T(t_{n-1}^+) , w_T(t_{n-1}^+))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(ts_c + l1*cbs, ts_c + l2*cbs, cbs, cbs)
                    += jump_coeff * mass * time_loc(l1,l2);

        // (v_T(t_{n-1}^-) , w_T(t_{n-1}^-))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(l1*cbs, l2*cbs, cbs, cbs)
                    += jump_coeff * mass * time_loc_bis(l1,l2);

        // - (v_T(t_{n-1}^-) , w_T(t_{n-1}^+))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(ts_c + l1*cbs, l2*cbs, cbs, cbs)
                    -= jump_coeff * mass * time_loc_cross(l2,l1);

        // - (v_T(t_{n-1}^+) , w_T(t_{n-1}^-))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(l1*cbs, ts_c + l2*cbs, cbs, cbs)
                    -= jump_coeff * mass * time_loc_cross(l1,l2);

        /* coupling coming from the gradient time jump penalization */
        // (\GRAD v_T(t_{n-1}^+) , \GRAD w_T(t_{n-1}^+))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(ts_c + l1*cbs, ts_c + l2*cbs, cbs, cbs)
                    += jump_coeff * stiff * time_loc(l1,l2);

        // (\GRAD v_T(t_{n-1}^-) , \GRAD w_T(t_{n-1}^-))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(l1*cbs, l2*cbs, cbs, cbs)
                    += jump_coeff * stiff * time_loc_bis(l1,l2);

        // - (\GRAD v_T(t_{n-1}^-) , \GRAD w_T(t_{n-1}^+))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(ts_c + l1*cbs, l2*cbs, cbs, cbs)
                    -= jump_coeff * stiff * time_loc_cross(l2,l1);

        // - (\GRAD v_T(t_{n-1}^+) , \GRAD w_T(t_{n-1}^-))_{Omega}
        for(size_t l1 = 0; l1 <= time_degree; l1++)
            for(size_t l2 = 0; l2 <= time_degree; l2++)
                coupling.block(l1*cbs, ts_c + l2*cbs, cbs, cbs)
                    -= jump_coeff * stiff * time_loc_cross(l1,l2);

        // at this point all the coupling terms have been implemented

        /*** loop on the time steps ***/
        for(int step_i = 0; step_i < time_steps; step_i++) {
            Matrix<scalar_type, Dynamic, 1> rhs = Matrix<scalar_type, Dynamic, 1>::Zero(lhs.cols());

            auto t_cell = *(time_mesh.cells_begin()+step_i);
            const auto qpst = integrate(time_mesh, t_cell , 2*time_degree);
            const auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            auto t_cb = make_scalar_monomial_basis(time_mesh, t_cell, time_degree);

            // T noise_size = 0.;
            T noise_size = 0.001;

            // heat equation RHS
            for(auto& qpt : qpst) // time integration
            {
                auto t_phi   = t_cb.eval_functions( qpt.point() );
                T time_point = qpt.point().x();

                for(auto& qp : qps) // space integration
                {
                    auto x_phi   = cb.eval_functions( qp.point() );

                    // T noise = noise_size * ((std::rand() % 200)-100) * 0.01;
                    T noise = 0.0;
                    T noised_data = rhs_fun( time_point , qp.point() ) + noise;

                    for(size_t l1 = 0; l1 <= time_degree; l1++) // loop on time dofs
                    {
                        rhs.block(dual_offset + l1*cbs, 0, cbs, 1)
                            += qp.weight() * qpt.weight() * t_phi[l1] *
                            noised_data * x_phi;
                    }
                }
            }

            // solution measurements
            if( varpi_fun(barycenter(msh,cl)) > 0.5 ) {
                // compute (u , w_T)_{I_n x T}
                Matrix<scalar_type, Dynamic, 1> data_rhs
                    = Matrix<scalar_type, Dynamic, 1>::Zero( cbs * (time_degree+1) );

                // time integration
                for(auto& qpt : qpst)
                {
                    auto t_phi   = t_cb.eval_functions( qpt.point() );
                    T time_point = qpt.point().x();

                    // space integration
                    for(auto& qp : qps)
                    {
                        auto x_phi   = cb.eval_functions( qp.point() );

                        T noise = noise_size * ((std::rand() % 200)-100) * 0.01;
                        // T noise = 0.0;
                        T noised_data = sol_fun( time_point , qp.point() ) + noise;

                        for(size_t l1 = 0; l1 <= time_degree; l1++)
                        {
                            data_rhs.block(l1*cbs, 0, cbs, 1) += qp.weight() * qpt.weight() * t_phi[l1] * noised_data * x_phi;
                        }
                    }
                }
                
                rhs.block(0, 0, cbs*(time_degree+1), 1) = data_rhs;
            }

            assembler.assemble(msh, cl, step_i, lhs, rhs);
        }

        /*** add the time coupling terms triplets ***/
        for(int step_i = 1; step_i < time_steps; step_i++) {
            assembler.add_time_coupling(msh, cl, step_i, coupling);
        }

    }
    cout << "end assembly loop" << endl;

    cout << "LHS.rows() = " << assembler.LHS.rows() << "  "
         << "LHS.cols() = " << assembler.LHS.cols() << endl;
    cout << "2xcbsx(l+1)xNcxNt + (Nf+Nfi)xfbsx(l+1)xNt = "
         << 2*cbs*(time_degree+1)*num_cells*time_steps + (assembler.num_assembled_faces() + nb_tot_faces) * fbs * (time_degree+1) * time_steps << endl;

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
    
    scalar_type t = dt * 0.5; // middle of the current time cell
    // scalar_type dt = final_time/time_steps;
    scalar_type L2H1_error = 0.;
    scalar_type L2L2_error = 0.;
    scalar_type L2H1_B_error = 0.;
    scalar_type L2L2_B_error = 0.;
    scalar_type L2H1_z = 0.;

    size_t freq_exp = 100000;

    /* Post-processing */
    // time loop
    for(size_t step_i = 0; step_i<time_steps; step_i++)
    {
        if(step_i % freq_exp == 0)
            std::cout << "Step " << step_i << std::endl;

        Matrix<scalar_type, Dynamic, 1> sol_silo = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());
        Matrix<scalar_type, Dynamic, 1> data_varpi = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());
        Matrix<scalar_type, Dynamic, 1> data_B = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());

        if(step_i % freq_exp == 0)
        {
            // be careful : this exports a very approximated solution
            // TODO : modify this export ...
            for (size_t i = 0; i < msh.cells_size(); i++)
                sol_silo(i) = u(i*cbs*(time_degree+1)*time_steps + cbs*(time_degree+1) * step_i);

            int cell_i = 0;
            for(auto& cl : msh)
            {
                const auto& bar = barycenter(msh,cl);
                data_varpi(cell_i) = varpi_fun( bar );
                data_B(cell_i) = B_fun( bar );
                cell_i++;
            }
            export_to_silo( msh, sol_silo, data_varpi, data_B, step_i );
        }

        auto t_cell = *(time_mesh.cells_begin()+step_i);
        auto t_cb = make_scalar_monomial_basis(time_mesh, t_cell, time_degree);
        const auto qpst = integrate(time_mesh, t_cell , 2*time_degree);

        size_t cell_i = 0;
        for (auto& cl : msh)
        {
            auto fcs    = faces(msh, cl);
            auto num_faces = fcs.size();
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

            /******** compute errors ********/

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

            const auto& bar = barycenter(msh,cl);
            auto loc_sol = assembler.take_local_solution(msh, cl, step_i, u, sol_fun);

            // coordinates of the dual sol (cell components)
            Matrix<scalar_type, Dynamic, 1> dual_sol
                = loc_sol.block((cbs+num_faces*fbs)*(time_degree+1), 0, cbs*(time_degree+1), 1);

            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    for(size_t I = 0; I < cbs; I++)
                        for(size_t J = 0; J < cbs; J++)
                        {
                            L2H1_error += diff(l1 * cbs + I) * diff(l2 * cbs + J) * time_mass(l1,l2) * grad_matrix(I,J);
                            L2L2_error += diff(l1 * cbs + I) * diff(l2 * cbs + J) * time_mass(l1,l2) * cell_mass(I,J);
                            if(time_B(t) && (B_fun(bar) > 0.5) )
                            {
                                L2H1_B_error += diff(l1 * cbs + I) * diff(l2 * cbs + J) * time_mass(l1,l2) * grad_matrix(I,J);
                                L2L2_B_error += diff(l1 * cbs + I) * diff(l2 * cbs + J) * time_mass(l1,l2) * cell_mass(I,J);
                            }
                            // L2H1-norm of the dual variable
                            L2H1_z += dual_sol(l1*cbs+I) * dual_sol(l2*cbs+J)* time_mass(l1,l2) * grad_matrix(I,J);
                        }

            cell_i++;
        }
        t += dt;
    } // time loop

    cout << "final time = " << t - dt*0.5 << endl;
    std::cout << "L2-H1-error = " << std::sqrt(L2H1_error) << std::endl;
    std::cout << "L2-L2-error = " << std::sqrt(L2L2_error) << std::endl;
    std::cout << "L2-H1-B-error = " << std::sqrt(L2H1_B_error) << std::endl;
    std::cout << "L2-L2-B-error = " << std::sqrt(L2L2_B_error) << std::endl;
    std::cout << "L2-H1-z-norm = " << std::sqrt(L2H1_z) << std::endl;

    test_info<double> TI;
    TI.H1_Om = std::sqrt(L2H1_error);
    TI.L2_Om = std::sqrt(L2L2_error);
    TI.H1_B = std::sqrt(L2H1_B_error);
    TI.L2_B = std::sqrt(L2L2_B_error);
    TI.H1_z = std::sqrt(L2H1_z);

    return TI;
}



template<typename T>
void
tests_auto_1d()
{
    typedef disk::generic_mesh<T, 1>  mesh_type;

    /*********************  REFINEMENT IN SPACE  **************************/
    {
        size_t nb_meshes = 4;


        // list of export files
        std::vector<std::string> files;
        files.push_back("./test_space_k0.txt");
        files.push_back("./test_space_k1.txt");
        files.push_back("./test_space_k2.txt");
        files.push_back("./test_space_k3.txt");

        // we test space degrees from 1 to 3
        for(int s_degree=1; s_degree < 4; s_degree++)
        {
            std::cout << blue << " WORKING WITH k = " << s_degree << std::endl;
            std::cout << nocolor;

            size_t t_degree = 3;
            size_t N = 128;

            // open the output file
            std::ofstream file;
            file.open (files.at(s_degree), std::ios::in | std::ios::trunc);
            if (!file.is_open())
                throw std::logic_error("file not open");

            // init the file
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1" << std::endl;

            // we test all the meshes in the list
            size_t num_elems = 8;
            for(size_t i=0; i < nb_meshes; i++)
            {
                num_elems *= 2;
                // load the mesh : 1d mesh only ATM
                mesh_type msh;
                disk::uniform_mesh_loader<T, 1> loader(0, 1, num_elems);
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << std::endl;
            }

            // close the file
            file.close();
        }

    }
    /*********************  REFINEMENT IN TIME  **************************/
    {
        size_t nb_meshes = 4;

        std::vector<std::string> files;
        files.push_back("./test_time_k0.txt");
        files.push_back("./test_time_k1.txt");
        files.push_back("./test_time_k2.txt");
        files.push_back("./test_time_k3.txt");

        // we test time degrees from 1 to 3
        for(int t_degree=1; t_degree < 4; t_degree++)
        {
            std::cout << blue << " WORKING WITH l = " << t_degree << std::endl;
            std::cout << nocolor;

            size_t s_degree = 3;
            size_t M = 256;

            // open the output file
            std::ofstream file;
            file.open (files.at(t_degree), std::ios::in | std::ios::trunc);
            if (!file.is_open())
                throw std::logic_error("file not open");

            // init the file
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1" << std::endl;

            // we test all the meshes in the list
            size_t N = 8;
            for(size_t i=0; i < nb_meshes; i++)
            {
                N *= 2;
                // load the mesh : 1d mesh only ATM
                mesh_type msh;
                disk::uniform_mesh_loader<T, 1> loader(0, 1, M);
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << std::endl;
            }

            // close the file
            file.close();
        }
    }
}



template<typename T>
void
tests_auto_2d()
{
    typedef disk::simplicial_mesh<T, 2>  mesh_type;

    /*********************  REFINEMENT IN SPACE  **************************/
    if(true)
    {
        // list of mesh files
        std::vector<std::string> meshes;
        meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d");
        meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri02.mesh2d");
        meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d");

        size_t nb_meshes = meshes.size();

        // list of export files
        std::vector<std::string> files;
        files.push_back("./test_space_k0.txt");
        files.push_back("./test_space_k1.txt");
        files.push_back("./test_space_k2.txt");
        files.push_back("./test_space_k3.txt");

        // we test space degree 1 and 2 only
        for(int s_degree=1; s_degree <= 2; s_degree++)
        {
            std::cout << blue << " WORKING WITH k = " << s_degree << std::endl;
            std::cout << nocolor;

            size_t t_degree = 1;
            size_t N = 20;

            // open the output file
            std::ofstream file;
            file.open (files.at(s_degree), std::ios::in | std::ios::trunc);
            if (!file.is_open())
                throw std::logic_error("file not open");

            // init the file
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1" << std::endl;

            // we test all the meshes in the list
            for(size_t i=0; i < nb_meshes; i++)
            {
                mesh_type msh;
                disk::netgen_mesh_loader<T, 2> loader;
                if( !loader.read_mesh(meshes.at(i)) )
                    std::cout << "error loading mesh !" << std::endl;
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << std::endl;
            }

            // close the file
            file.close();
        }

    }
    /*********************  REFINEMENT IN TIME  **************************/
    if(true)
    {
        size_t nb_meshes = 3;

        std::vector<std::string> files;
        files.push_back("./test_time_k0.txt");
        files.push_back("./test_time_k1.txt");
        files.push_back("./test_time_k2.txt");
        files.push_back("./test_time_k3.txt");

        // we test time degree 0 and 1 only
        for(int t_degree=0; t_degree < 2; t_degree++)
        {
            std::cout << blue << " WORKING WITH l = " << t_degree << std::endl;
            std::cout << nocolor;

            size_t s_degree = 2;

            // open the output file
            std::ofstream file;
            file.open (files.at(t_degree), std::ios::in | std::ios::trunc);
            if (!file.is_open())
                throw std::logic_error("file not open");

            // init the file
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1" << std::endl;

            // we test all the meshes in the list
            size_t N = 2;
            for(size_t i=0; i < nb_meshes; i++)
            {
                N *= 2;
                mesh_type msh;
                disk::netgen_mesh_loader<T, 2> loader;
                if( !loader.read_mesh("./../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d") )
                    std::cout << "error loading mesh !" << std::endl;
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << std::endl;
            }

            // close the file
            file.close();
        }
    }
}

/* run main with :
   ./heat_UC
*/
int main(int argc, char **argv)
{
    tests_auto_1d<double>();
    // tests_auto_2d<double>();
    return 0;
}

#if 0
/* run main with :
   ./heat_UC -m ../../../diskpp/meshes/2D_quads/diskpp/testmesh-16-16.quad -k 1 -N 8 -l 0
   ./heat_UC -M 8 -k 1 -N 8 -l 0
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
#endif
