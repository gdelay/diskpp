/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *
 * Matteo Cicuttin (C) 2016-2026
 * matteo.cicuttin@polito.it
 *
 * Politecnico di Torino - DISMA
 * Dipartimento di Matematica
 *
 * This file is copyright of the following author:
 * Guillaume Delay (C) 2020-2021         guillaume.delay@sorbonne-universite.fr
 * Sorbonne Universite
 * Laboratoire Jacques-Louis Lions (LJLL)
 *
 */
/*
 * The content of this file corresponds to solving an obstacle problem with Pk elements
 * using Pk Lagrange multipliers.
 */

#include "gnuplot_output.hpp"
#include "Lagrange_hho.hpp"

// For the mesh data structure
#include "diskpp/mesh/mesh.hpp"

// For the loaders and related helper functions
#include "diskpp/loaders/loader.hpp"

#include "diskpp/solvers/direct_solvers.hpp"

using disk::cells;
using disk::faces;


//////////////////////////////////////////////////////////////////////////////////
///////////////////////////   MEMBRANE ASSEMBLERS   //////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<typename Mesh>
class membrane_assembler
{
    using T = typename Mesh::coordinate_type;
    typedef disk::BoundaryConditions<Mesh, true> boundary_type;

    std::vector<size_t>     compress_table;
    std::vector<size_t>     expand_table;
    hho_degree_info         di;
    std::vector<Triplet<T>> triplets;
    bool                    use_bnd;
    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> >       loc_RHS1, loc_RHS2;

    size_t num_all_faces, num_dirichlet_faces, num_other_faces, system_size;

    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

public:
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    SparseMatrix<T> LHS;
    vector_type     RHS;

    membrane_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi), use_bnd(false)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        num_all_faces       = msh.faces_size();
        num_dirichlet_faces = std::count_if(msh.faces_begin(), msh.faces_end(), is_dirichlet);
        num_other_faces     = num_all_faces - num_dirichlet_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );

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

        auto num_cells = msh.cells_size();
        loc_LHS.resize( num_cells );
        loc_RHS1.resize( num_cells );
        loc_RHS2.resize( num_cells );

        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
        system_size = 2 * fbs * num_other_faces + 3 * cbs * num_cells;

        LHS = SparseMatrix<T>(system_size, system_size);
        RHS = vector_type::Zero(system_size);
    }

    void
    set_loc_mat(const Mesh&                     msh,
                const typename Mesh::cell_type& cl,
                const matrix_type&              lhs,
                const vector_type&              rhs1,
                const vector_type&              rhs2)
    {
        auto cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS1.at( cell_offset ) = rhs1;
        loc_RHS2.at( cell_offset ) = rhs2;
    }


    void
    assemble_mat(const Mesh&                     msh,
                 const typename Mesh::cell_type& cl,
                 const matrix_type&              lhs)
    {
        if(use_bnd)
            throw std::invalid_argument("membrane_assembler: you have to use boundary type");

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        const auto fcs = faces(msh, cl);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + fcs.size() * fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = offset(msh, fc);
            const auto face_LHS_offset = msh.cells_size() * cbs + compress_table.at(face_offset) * fbs;

            const bool dirichlet = is_dirichlet(fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
        }

        // membrane 1
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
            }
        }

        // membrane 2
        size_t offset2 = msh.cells_size() * cbs + num_other_faces * fbs;
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(offset2 + asm_map[i], offset2 + asm_map[j], lhs(i,j)) );
            }
        }

    } // assemble_mat()

    template<typename Function1, typename Function2>
    void
    assemble_rhs(const Mesh&                     msh,
                 const typename Mesh::cell_type& cl,
                 const matrix_type&              lhs,
                 const vector_type&              rhs1,
                 const vector_type&              rhs2,
                 const Function1&                 dirichlet_bf1,
                 const Function2&                 dirichlet_bf2)
    {
        if(use_bnd)
            throw std::invalid_argument("membrane_assembler: you have to use boundary type");

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };


        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        const auto fcs = faces(msh, cl);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + fcs.size() * fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        vector_type dirichlet_data1 = vector_type::Zero(cbs + fcs.size()*fbs);
        vector_type dirichlet_data2 = vector_type::Zero(cbs + fcs.size()*fbs);

        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = offset(msh, fc);
            const auto face_LHS_offset = msh.cells_size() * cbs + compress_table.at(face_offset) * fbs;

            const bool dirichlet = is_dirichlet(fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                auto fb = make_scalar_Lagrange_basis(msh, fc, di.face_degree());
                dirichlet_data1.block(cbs + face_i * fbs, 0, fbs, 1) =
                    project_function(msh, fc, fb, dirichlet_bf1, di.face_degree());
                dirichlet_data2.block(cbs + face_i * fbs, 0, fbs, 1) =
                    project_function(msh, fc, fb, dirichlet_bf2, di.face_degree());
            }
        }

        // membrane 1
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( !asm_map[j].assemble() )
                    RHS[ asm_map[i] ] -= lhs(i,j) * dirichlet_data1(j);
            }
        }

        for (size_t i = 0; i < rhs1.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS[ asm_map[i] ] += rhs1(i);
        }

        // membrane 2
        size_t offset2 = msh.cells_size() * cbs + num_other_faces * fbs;

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( !asm_map[j].assemble() )
                    RHS[ offset2 + asm_map[i] ] -= lhs(i,j) * dirichlet_data2(j);
            }
        }

        for (size_t i = 0; i < rhs2.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS[ offset2 + asm_map[i] ] += rhs2(i);
        }

    } // assemble_rhs()


    // init : set no contact constraint and assemble matrix
    // (matrix for the first iteration)
    template<typename Function1, typename Function2>
    void
    init(const Mesh&                      msh,
         const Function1&                 dirichlet_bf1,
         const Function2&                 dirichlet_bf2)
    {
        const auto facdeg  = di.face_degree();

        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            auto cell_offset = offset(msh, cl);
            assemble_mat(msh, cl, loc_LHS.at(cell_offset));
            assemble_rhs(msh, cl, loc_LHS.at(cell_offset), loc_RHS1.at(cell_offset),
                         loc_RHS2.at(cell_offset), dirichlet_bf1, dirichlet_bf2);
        }
        // assemble constraints (no contact)
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto mult_offset = 2*cbs * msh.cells_size() + 2*fbs * num_other_faces;
        auto nb_mult = cbs*msh.cells_size();
        for(size_t i = 0; i < nb_mult; i++)
        {
            triplets.push_back( Triplet<T>(mult_offset + i, mult_offset + i, 1.0) );
            triplets.push_back( Triplet<T>(i, mult_offset + i, -1.0) );
            triplets.push_back( Triplet<T>(mult_offset/2 + i, mult_offset + i, 1.0) );
        }

        size_t offset2 = msh.cells_size() * cbs + num_other_faces * fbs;

        // end assembly
        finalize();
    }


    // update_mat : assemble matrix according to the previous iteration solution
    void
    update_mat(const Mesh&                     msh,
               const vector_type&              prev_sol)
    {
        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            auto cell_offset = offset(msh, cl);
            assemble_mat( msh, cl, loc_LHS.at(cell_offset) );
        }

        // assemble constraints
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto nb_mult = msh.cells_size() * cbs;
        auto offset2 = nb_mult + num_other_faces * fbs;
        for(size_t i = 0; i < nb_mult; i++)
        {
            auto delta_u = prev_sol[i] - prev_sol[offset2+i];
            auto mult = prev_sol[2*offset2+i];

            if(delta_u <= mult)
            {
                triplets.push_back( Triplet<T>(2*offset2 + i, i, 1.0) );
                triplets.push_back( Triplet<T>(2*offset2 + i, offset2 + i, -1.0) );
            }
            else
                triplets.push_back( Triplet<T>(2*offset2 + i, 2*offset2 + i, 1.0) );
        }

        // identity blocks
        for(size_t i = 0; i < nb_mult; i++)
        {
            triplets.push_back( Triplet<T>(i, 2*offset2 + i, -1.0) );
        }
        for(size_t i = 0; i < nb_mult; i++)
        {
            triplets.push_back( Triplet<T>(offset2 + i, 2*offset2 + i, 1.0) );
        }

        // end assembly
        finalize();
    }

    bool
    stop(const Mesh& msh, const vector_type& sol)
    {
        T TOL = 1e-12;

        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);

        bool ret = true;

        // offset
        auto offset2 = msh.cells_size() * cbs + num_other_faces * fbs;

        // test the cells
        for(size_t i = 0; i < msh.cells_size() * cbs; i++)
        {
            auto delta_u = sol(i) - sol(offset2 + i);
            auto mult = sol(2 * offset2 + i);

            if( delta_u < -TOL || mult < -TOL)
            {
                ret = false;
                break;
            }
        }

        return ret;
    }



    template<typename Function1, typename Function2>
    vector_type
    take_u(const Mesh& msh, const typename Mesh::cell_type& cl,
           const vector_type& solution, const Function1& dirichlet_bf1,
           const Function2& dirichlet_bf2)
    {
        auto cell_offset        = offset(msh, cl);
        auto celdeg = di.cell_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);
        auto fcs = faces(msh, cl);

        auto offset2 = msh.cells_size() * cbs + num_other_faces * fbs;

        auto cell_SOL_offset = cell_offset * cbs;
        vector_type ret = vector_type::Zero(2*cbs + 2*num_faces*fbs);
        // membrane 1
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);
        // membrane 2
        ret.block(cbs + num_faces*fbs, 0, cbs, 1)
            = solution.block(offset2 + cell_SOL_offset, 0, cbs, 1);

        for(size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                return msh.is_boundary(fc);
            };

            bool dirichlet = is_dirichlet(fc);

            if (dirichlet)
            {
                auto fb = make_scalar_Lagrange_basis(msh, fc, di.face_degree());

                matrix_type mass = make_mass_matrix(msh, fc, fb, di.face_degree());
                vector_type rhs1 = make_rhs(msh, fc, fb, dirichlet_bf1, di.face_degree());
                ret.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs1);
                vector_type rhs2 = make_rhs(msh, fc, fb, dirichlet_bf2, di.face_degree());
                ret.block(2*cbs + (num_faces+face_i)*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs2);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = msh.cells_size() * cbs + compress_table.at(face_offset)*fbs;
                ret.block(cbs + face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
                ret.block(2*cbs + (num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(offset2 + face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    vector_type
    take_mult(const Mesh& msh, const typename Mesh::cell_type& cl,
              const vector_type& solution)
    {
        auto cell_offset        = offset(msh, cl);
        auto celdeg = di.cell_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
        const auto cb = make_scalar_Lagrange_basis(msh, cl, celdeg);

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);

        size_t offset2 = msh.cells_size() * cbs + num_other_faces * fbs;
        auto multT_dual = solution.block(2*offset2 + cell_offset * cbs, 0, cbs, 1);
        auto mass_matrixT = make_mass_matrix(msh, cl, cb);
        vector_type multT_primal = mass_matrixT.ldlt().solve(multT_dual);

        return multT_primal;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();

        dump_sparse_matrix(LHS, "diff.dat");
    }

    size_t num_assembled_faces() const
    {
        return num_other_faces;
    }

};
template<typename Mesh>
auto make_membrane_assembler_Lag(const Mesh& msh, hho_degree_info hdi)
{
    return membrane_assembler<Mesh>(msh, hdi);
}


/*
 * TODO : ajouter le code pour SC
 */

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   MEMBRANES SOLVER   //////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<typename Mesh>
typename Mesh::coordinate_type
run_membranes_solver(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;

    hho_degree_info hdi(degree+1, degree);

#if 0
    auto rhs_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 8.0 * R2 - 16.0 * r2;
        else
            return - 8.0 * R2;
    };
    auto rhs_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 8.0 * R2;
    };
    auto sol_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_fun2 = [](const point_type& pt) -> T {
        return 0.0;
    };
    auto sol_grad1 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*2.0*(r2 - R2);
            ret(0) =  coeff * x1;
            ret(1) =  coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }
        return ret;
    };
    auto sol_grad2 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;

        ret(0) = 0.0;
        ret(1) = 0.0;

        return ret;
    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 8.0 * R2;
    };
#elif 0
    auto rhs_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -4.0;
        else
            return -6.0;
    };
    auto rhs_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        auto r = std::sqrt(r2);
        if(r2 > R2)
            return (9.0*r2 - 4.0*r - R2) / r;
        else
            return -2.0;
    };
    auto sol_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;

        return r2 - R2;
    };
    auto sol_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        auto r = std::sqrt(r2);
        if(r2 > R2)
            return (1.0-r) * (r2 - R2);
        else
            return r2 - R2;
    };
    auto sol_grad1 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        ret(0) = 2*x1;
        ret(1) = 2*y1;
        return ret;
    };
    auto sol_grad2 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        auto r = std::sqrt(r2);
        if(r2 > R2)
        {
            T coeff = (2.*r - 3.*r2 - R2) / r;
            ret(0) = coeff * x1;
            ret(1) = coeff * y1;
        }
        else
        {
            ret(0) = 2*x1;
            ret(1) = 2*y1;
        }
        return ret;

    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 2.0;
    };
#else
    auto rhs_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -24.0*(r2-R2)*(r2-R2)*(r2-R2)*(r2-R2)*(6.0*r2-R2);
        else
            return -1000.0*r2*std::sqrt(r2)*(R2-r2)*(R2-r2)*(R2-r2);
    };
    auto rhs_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 24.0*(r2-R2)*(r2-R2)*(r2-R2)*(r2-R2)*(6.0*r2-R2);
        else
            return 1000.0*r2*std::sqrt(r2)*(R2-r2)*(R2-r2)*(R2-r2);
    };
    auto sol_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_grad1 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*6.0*(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
            ret(0) =  coeff * x1;
            ret(1) =  coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }

        return ret;
    };
    auto sol_grad2 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*6.0*(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
            ret(0) =  -coeff * x1;
            ret(1) =  -coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }

        return ret;
    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 1000.0*r2*std::sqrt(r2)*(R2-r2)*(R2-r2)*(R2-r2);
    };
#endif

    clock_t t1, t2;
    T assembly_time = 0.0, solve_time = 0.0;

    auto assembler = make_membrane_assembler_Lag(msh, hdi);
    // auto assembler_sc = make_membrane_condensed_assembler_Lag(msh, hdi);
    auto assembler_sc = make_membrane_assembler_Lag(msh, hdi);

    bool scond = false; // static condensation

    t1 = clock();
    for (auto& cl : msh)
    {
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto gr     = make_vector_hho_gradrec_Lag(msh, cl, hdi);
        auto stab   = make_scalar_hdg_stabilization_Lag(msh, cl, hdi);
        auto rhs1    = make_rhs(msh, cl, cb, rhs_fun1);
        auto rhs2    = make_rhs(msh, cl, cb, rhs_fun2);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;
        if(scond)
        {
            assembler_sc.set_loc_mat(msh, cl, A, rhs1, rhs2);
        }
        else
        {
            assembler.set_loc_mat(msh, cl, A, rhs1, rhs2);
        }
    }

    t2 = clock();
    assembly_time += (T)(t2-t1)/CLOCKS_PER_SEC;

    if(scond)
        std::cout << "end assembly : nb dof = " << assembler_sc.RHS.size() << std::endl;
    else
        std::cout << "end assembly : nb dof = " << assembler.RHS.size() << std::endl;


    size_t systsz, nnz;
    dynamic_vector<T> sol;
    size_t Newton_iter = 0;
    size_t max_Newton_iter = 300;
    bool stop_loop = false;

    // Newton loop
    while(!stop_loop and Newton_iter < max_Newton_iter)
    {
        t1 = clock();
        if(scond)
        {
            if(Newton_iter == 0)
                assembler_sc.init(msh, sol_fun1, sol_fun2);
            else
            {
                // assembler_sc.update_mat(msh, sol, sol_fun1, sol_fun2);
                assembler_sc.update_mat(msh, sol);
            }
            systsz = assembler_sc.LHS.rows();
        }
        else
        {
            if(Newton_iter == 0)
                assembler.init(msh, sol_fun1, sol_fun2);
            else
                assembler.update_mat(msh, sol);
            systsz = assembler.LHS.rows();
        }
        t2 = clock();
        assembly_time += (T)(t2-t1)/CLOCKS_PER_SEC;

        sol = dynamic_vector<T>::Zero(systsz);

        /*
          disk::solvers::pardiso_params<T> pparams;
          pparams.report_factorization_Mflops = false;
        */

        t1 = clock();
        if(scond)
        {
            std::cout << "Running solver..." << std::flush;
            disk::solvers::sparse_lu(assembler_sc.LHS, assembler_sc.RHS, sol);
            std::cout << "done" << std::endl;
            // mkl_pardiso(pparams, assembler_sc.LHS, assembler_sc.RHS, sol);
        }
        else
        {
            std::cout << "Running solver..." << std::flush;
            disk::solvers::sparse_lu(assembler.LHS, assembler.RHS, sol);
            std::cout << "done" << std::endl;
            // mkl_pardiso(pparams, assembler.LHS, assembler.RHS, sol);
        }

        t2 = clock();
        solve_time += (T)(t2-t1)/CLOCKS_PER_SEC;

        Newton_iter++;
        std::cout << "end Newton iter nb " << Newton_iter << std::endl;

        if(scond)
        {
            // stop_loop = assembler_sc.stop(msh, sol, sol_fun1, sol_fun2);
            stop_loop = assembler_sc.stop(msh, sol);
        }
        else
            stop_loop = assembler.stop(msh, sol);
    } // Newton loop

    std::cout << "Start post-process" << std::endl;

    T u_H1_error = 0.0;
    T u_L2_error = 0.0;
    T mult_L2_error = 0.0;

    
    postprocess_output<T>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<T> >("uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<T> >("uT2.dat");
    auto multT_gp  = std::make_shared< gnuplot_output_object<T> >("multT.dat");

    auto uF1_gp  = std::make_shared< gnuplot_output_object<T> >("uF1.dat");
    auto uF2_gp  = std::make_shared< gnuplot_output_object<T> >("uF2.dat");
    auto multF_gp  = std::make_shared< gnuplot_output_object<T> >("multF.dat");
    

    for (auto& cl : msh)
    {
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto num_faces = howmany_faces(msh, cl);

        Eigen::Matrix<T, Eigen::Dynamic, 1> fullsol, mult_sol;

        t1 = clock();
        if(scond)
        {
            fullsol = assembler_sc.take_u(msh, cl, sol, sol_fun1, sol_fun2);
            // mult_sol = assembler_sc.take_mult(msh, cl, sol, sol_fun1, sol_fun2);
            mult_sol = assembler_sc.take_mult(msh, cl, sol);
        }
        else
        {
            fullsol = assembler.take_u(msh, cl, sol, sol_fun1, sol_fun2);
            mult_sol = assembler.take_mult(msh, cl, sol);
        }
        t2 = clock();
        assembly_time += (T)(t2-t1)/CLOCKS_PER_SEC;

        auto cell_dofs1 = fullsol.head( cbs );
        Matrix<T, Dynamic, 1> cell_dofs2 = fullsol.block( cbs + num_faces * fbs, 0, cbs, 1);
        auto mult_cell_dofs = mult_sol.head( cbs );

        // errors
        const auto celdeg = hdi.cell_degree();
        const auto qps = integrate(msh, cl, 2*celdeg);
        for (auto& qp : qps)
        {
            auto grad_ref1 = sol_grad1( qp.point() );
            auto grad_ref2 = sol_grad2( qp.point() );
            auto t_dphi = cb.eval_gradients( qp.point() );
            Matrix<T, 1, 2> grad1 = Matrix<T, 1, 2>::Zero();
            Matrix<T, 1, 2> grad2 = Matrix<T, 1, 2>::Zero();

            for (size_t i = 0; i < cbs; i++ )
            {
                grad1 += cell_dofs1(i) * t_dphi.block(i, 0, 1, 2);
                grad2 += cell_dofs2(i) * t_dphi.block(i, 0, 1, 2);
            }

            // H1-error
            u_H1_error += qp.weight() * (grad_ref1 - grad1).dot(grad_ref1 - grad1);
            u_H1_error += qp.weight() * (grad_ref2 - grad2).dot(grad_ref2 - grad2);

            // L2-error
            auto t_phi = cb.eval_functions( qp.point() );
            T v1 = cell_dofs1.dot( t_phi );
            T v2 = cell_dofs2.dot( t_phi );
            u_L2_error += qp.weight() * (sol_fun1(qp.point()) - v1) * (sol_fun1(qp.point()) - v1);
            u_L2_error += qp.weight() * (sol_fun2(qp.point()) - v2) * (sol_fun2(qp.point()) - v2);

            // mult-L2-error
            T mult = mult_cell_dofs.dot( t_phi );
            T mult_sol = mult_fun(qp.point());
            mult_L2_error += qp.weight() * (mult_sol - mult) * (mult_sol - mult);
        }

        
        // gnuplot output for cells
        auto pts = points(msh, cl);
        for(size_t i=0; i < pts.size(); i++)
        {
            T sol_uT1 = cell_dofs1.dot( cb.eval_functions( pts[i] ) );
            uT1_gp->add_data( pts[i], sol_uT1 );
            T sol_uT2 = cell_dofs2.dot( cb.eval_functions( pts[i] ) );
            uT2_gp->add_data( pts[i], sol_uT2 );
            T sol_multT = mult_cell_dofs.dot( cb.eval_functions(pts[i]) );
            multT_gp->add_data( pts[i], sol_multT );
        }
        /*
        // gnuplot output for faces
        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc = fcs[face_i];
            auto face_sol = fullsol.block(cbs+face_i*fbs, 0, fbs, 1);

            const auto fb = make_scalar_Lagrange_basis(msh, fc, hdi.face_degree());
            auto barF = barycenter(msh, fc);

            T solbarF = fb.eval_functions(barF).dot(face_sol);
            uF_gp->add_data( barF, solbarF );

            if( mult_sol.size() > cbs )
            {
                auto face_mult = mult_sol.block(cbs+face_i*fbs, 0, fbs, 1);
                T multbarF = fb.eval_functions(barF).dot(face_mult);
                multF_gp->add_data( barF, multbarF );
            }
        }
        */
    }
    
    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(multT_gp);
    postoutput.add_object(uF1_gp);
    postoutput.add_object(uF2_gp);
    postoutput.add_object(multF_gp);
    postoutput.write();
    

    std::cout << "ended run : H1-error is " << std::sqrt(u_H1_error) << std::endl;
    std::cout << "            L2-error is " << std::sqrt(u_L2_error) << std::endl;
    std::cout << "            mult-L2-error is " << std::sqrt(mult_L2_error) << std::endl;
    std::cout << "time : assembly : " << assembly_time << " sec" << std::endl;
    std::cout << "       solve    : " << solve_time    << " sec" << std::endl;
    
    return std::sqrt(u_H1_error);
}


//////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////   MAIN   /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

/*
 * TODO : prepare this section
 * faire tous les ajouts de la partie contact
 * check solver
 * check outputs
 * Static condensation
 * rename file
 * test info
 * auto tests
 * reference to the article (file and algo)
 * cleaning the file
 * tolerance
 * check error mult
 * check no faces on mult
 * compare the results with the ones of the article
 */


int main(void)
{
    using T = double;

    // degree of the polynomials on the faces
    size_t degree = 0;
    
    typedef disk::generic_mesh<T, 2>  mesh_type;
    typedef disk::simplicial_mesh<T, 2>  mesh_type2;
    
    if(1)
    {
        std::vector<std::string> meshfiles;
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_2.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");

        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j3.mesh2d");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j4.mesh2d");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j5.mesh2d");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j6.mesh2d");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j7.mesh2d");

        meshfiles.push_back("../../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d");
        meshfiles.push_back("../../../../diskpp/meshes/2D_triangles/netgen/tri02.mesh2d");
        meshfiles.push_back("../../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d");
        meshfiles.push_back("../../../../diskpp/meshes/2D_triangles/netgen/tri04.mesh2d");
        meshfiles.push_back("../../../../diskpp/meshes/2D_triangles/netgen/tri05.mesh2d");

        for(size_t i=0; i < meshfiles.size(); i++)
        {
            // mesh_type msh;
            mesh_type2 msh;
            // disk::fvca5_mesh_loader<T, 2> loader;
            disk::netgen_mesh_loader<T, 2> loader;
            if (!loader.read_mesh(meshfiles.at(i)) )
            {
                std::cout << "Problem loading mesh." << std::endl;
            }
            loader.populate_mesh(msh);
            run_membranes_solver(msh, degree);
        }

    }
    else
    {
        mesh_type2 msh;
        // disk::fvca5_mesh_loader<T, 2> loader;
        disk::netgen_mesh_loader<T, 2> loader;
        // std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1";
        // std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/netgen/mesh_j7.mesh2d";
        std::string mesh_filename = "../../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d";
        if (!loader.read_mesh(mesh_filename) )
        {
            std::cout << "Problem loading mesh." << std::endl;
        }
        loader.populate_mesh(msh);
        run_membranes_solver(msh, degree);
    }

    std::cout << "\a" << std::endl;
    return 0;
}
