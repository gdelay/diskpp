/*
 *       /\         DISK++, a template library for DIscontinuous SKeletal
 *      /__\        methods.
 *     /_\/_\
 *    /\    /\      Matteo Cicuttin (C) 2016, 2017, 2018
 *   /__\  /__\     matteo.cicuttin@enpc.fr
 *  /_\/_\/_\/_\    École Nationale des Ponts et Chaussées - CERMICS
 *
 * This file is copyright of the following authors:
 * Guillaume Delay (C) 2022                     guillaume.delay@sorbonne-universite.fr
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


#include <iostream>
#include "bases/bases.hpp"
#include "quadratures/quadratures.hpp"
#include "methods/hho"
#include "core/loaders/loader.hpp"
#include "solvers/solver.hpp"
#include "output/silo.hpp"


//////////////////   test case   ///////////////////
/* RHS definition */
template<typename Mesh>
struct rhs_functor;


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct rhs_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    Matrix<scalar_type, 2, 1> operator()(const point_type& pt) const
    {
        Matrix<scalar_type, 2, 1> ret;

        // scalar_type x1 = pt.x();
        // scalar_type x2 = x1 * x1;
        // scalar_type y1 = pt.y();
        // scalar_type y2 = y1 * y1;

        // scalar_type ax =  x2 * (x2 - 2. * x1 + 1.);
        // scalar_type ay =  y2 * (y2 - 2. * y1 + 1.);
        // scalar_type bx =  x1 * (4. * x2 - 6. * x1 + 2.);
        // scalar_type by =  y1 * (4. * y2 - 6. * y1 + 2.);
        // scalar_type cx = 12. * x2 - 12.* x1 + 2.;
        // scalar_type cy = 12. * y2 - 12.* y1 + 2.;
        // scalar_type dx = 24. * x1 - 12.;
        // scalar_type dy = 24. * y1 - 12.;

        // ret(0) = - cx * by - ax * dy + 5.* x2 * x2;
        // ret(1) = + cy * bx + ay * dx + 5.* y2 * y2;

        ret(0) = 0.;
        ret(1) = 0.;

        return ret;
    }
};

template<typename Mesh>
auto make_rhs_function(const Mesh& msh)
{
    return rhs_functor<Mesh>();
}

/***************************************************************************/
/* Expected velocity solution definition */
template<typename Mesh>
struct velocity_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct velocity_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    Matrix<scalar_type, 2, 1> operator()(const point_type& pt) const
    {
        Matrix<scalar_type, 2, 1> ret;

        // scalar_type x1 = pt.x();
        // scalar_type x2 = x1 * x1;
        // scalar_type y1 = pt.y();
        // scalar_type y2 = y1 * y1;

        // ret(0) =  x2 * (x2 - 2. * x1 + 1.)  * y1 * (4. * y2 - 6. * y1 + 2.);
        // ret(1) = -y2 * (y2 - 2. * y1 + 1. ) * x1 * (4. * x2 - 6. * x1 + 2.);

        ret(0) = 1000 * pt.y() * (1 - pt.y()) * 0.5;
        ret(1) = 0.;

        return ret;
    }
};

template<typename Mesh>
auto make_velocity_function(const Mesh& msh)
{
    return velocity_functor<Mesh>();
}

/***************************************************************************/
/* Expected pressure solution definition */
template<typename Mesh>
struct pressure_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct pressure_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        // return std::pow(pt.x(), 5.)  +  std::pow(pt.y(), 5.)  - 1./3.;
        return 1000 * (1-pt.x());
    }
};

template<typename Mesh>
auto make_pressure_function(const Mesh& msh)
{
    return pressure_functor<Mesh>();
}
////////////////////////////////

//////////////////  ASSEMBLER  ///////////////////////
// currently, this assembler deals with Dirichlet BC or Neumann (no interface is considered)
template<typename Mesh>
class interface_assembler
{
    using T = typename Mesh::coordinate_type;
    typedef disk::vector_boundary_conditions<Mesh> boundary_type;

    std::vector<size_t> compress_table;

    boundary_type           m_bnd;
    disk::hho_degree_info         di;
    std::vector<Triplet<T>> triplets;

    size_t num_all_faces, num_dirichlet_faces, num_other_faces;
    size_t cbs_A, cbs_B, fbs_A;
    size_t system_size;

    bool dirichlet_only;

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
    typedef disk::dynamic_matrix<T> matrix_type;
    typedef disk::dynamic_vector<T> vector_type;

    SparseMatrix<T> LHS;
    vector_type     RHS;

    interface_assembler(const Mesh& msh, const disk::hho_degree_info& hdi, const boundary_type& bnd)  : di(hdi), m_bnd(bnd)
    {
        auto is_dirichlet = [&](const typename Mesh::face& fc) -> bool {
                                auto fc_id = msh.lookup(fc);
                                return bnd.is_dirichlet_face(fc_id);
                            };


        num_all_faces       = msh.faces_size();
        num_dirichlet_faces = std::count_if(msh.faces_begin(), msh.faces_end(), is_dirichlet);
        num_other_faces     = num_all_faces - num_dirichlet_faces;

        compress_table.resize(num_all_faces);

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = *std::next(msh.faces_begin(), i);
            if (!is_dirichlet(fc))
            {
                compress_table.at(i)               = compressed_offset;
                compressed_offset++;
            }
        }

        // dirichlet_only : used to know if we need a null mean constraint for the pressure
        dirichlet_only = ( m_bnd.nb_faces_dirichlet() == m_bnd.nb_faces_boundary() );

        assert(m_bnd.nb_faces_dirichlet() == num_dirichlet_faces);

        cbs_A = disk::vector_basis_size(di.cell_degree(), Mesh::dimension, Mesh::dimension);
        fbs_A = disk::vector_basis_size(di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        cbs_B = disk::scalar_basis_size(di.face_degree(), Mesh::dimension);


        system_size = cbs_A * msh.cells_size() + fbs_A * num_other_faces + cbs_B * msh.cells_size();
        if(dirichlet_only)
            system_size++; // if only dirichlet BC : add the null mean pressure constraint

        LHS = SparseMatrix<T>(system_size, system_size);
        RHS = vector_type::Zero(system_size);
    }

    void
    assemble(const Mesh&                     msh,
             const typename Mesh::cell_type& cl,
             const matrix_type&              lhs_A,
             const matrix_type&              lhs_B,
             const vector_type&              rhs)
    {
        auto fcs = faces(msh, cl);

        // we reserve the size of the local unknowns (velocity only)
        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs_A + fcs.size() * fbs_A);

        auto cell_offset     = offset(msh, cl);
        auto cell_LHS_offset = cell_offset * cbs_A;

        // we assemble the velocity cell dofs
        for (size_t i = 0; i < cbs_A; i++)
            asm_map.push_back(assembly_index(cell_LHS_offset + i, true));

        // dirichlet_data for Dirichlet faces
        vector_type dirichlet_data = vector_type::Zero(cbs_A + fcs.size() * fbs_A);

        // we assemble the velocity faces dofs that are not on Dirichlet faces
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            auto fc              = fcs[face_i];
            auto face_offset     = offset(msh, fc);
            auto face_LHS_offset = cbs_A * msh.cells_size() + compress_table.at(face_offset) * fbs_A;

            auto fc_id     = msh.lookup(fc);
            bool dirichlet = m_bnd.is_dirichlet_face(fc_id);

            for (size_t i = 0; i < fbs_A; i++)
                asm_map.push_back(assembly_index(face_LHS_offset + i, !dirichlet));

            if (dirichlet)
            {
                const auto face_id = msh.lookup(fc);

                auto dirichlet_fun = m_bnd.dirichlet_boundary_func(face_id);

                dirichlet_data.block(cbs_A + face_i * fbs_A, 0, fbs_A, 1) =
                    project_function(msh, fc, di.face_degree(), dirichlet_fun, di.face_degree());
            }
        }

        assert(asm_map.size() == lhs_A.rows() && asm_map.size() == lhs_A.cols());

        // assemble the lhs_A part
        for (size_t i = 0; i < lhs_A.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_A.cols(); j++)
            {
                if (asm_map[j].assemble())
                    triplets.push_back(Triplet<T>(asm_map[i], asm_map[j], lhs_A(i, j)));
                else
                    RHS(asm_map[i]) -= lhs_A(i, j) * dirichlet_data(j);
            }
        }


        // assemble the lhs_B part
        auto B_offset = cbs_A * msh.cells_size() + fbs_A * num_other_faces + cbs_B * cell_offset;
        for (size_t i = 0; i < lhs_B.rows(); i++)
        {
            for (size_t j = 0; j < lhs_B.cols(); j++)
            {
                auto global_i = B_offset + i;
                auto global_j = asm_map[j];
                if (asm_map[j].assemble())
                {
                    // the problem is symmetric : we assemble (i,j) and (j,i)
                    triplets.push_back(Triplet<T>(global_i, global_j, lhs_B(i, j)));
                    triplets.push_back(Triplet<T>(global_j, global_i, lhs_B(i, j)));
                }
                else
                    RHS(global_i) -= lhs_B(i, j) * dirichlet_data(j);
            }
        }

        // add zero-mean value constraint
        auto        scalar_cell_basis = make_scalar_monomial_basis(msh, cl, di.face_degree());
        auto        qps               = integrate(msh, cl, di.face_degree());
        vector_type mult              = vector_type::Zero(scalar_cell_basis.size());
        for (auto& qp : qps)
        {
            auto phi = scalar_cell_basis.eval_functions(qp.point());
            mult += qp.weight() * phi;
        }

        // if only dirichlet BC : add the null mean pressure constraint
        if(dirichlet_only)
        {
            auto mult_offset = cbs_A * msh.cells_size() + fbs_A * num_other_faces + cbs_B * msh.cells_size();

            for (size_t i = 0; i < mult.rows(); i++)
            {
                triplets.push_back(Triplet<T>(B_offset + i, mult_offset, mult(i)));
                triplets.push_back(Triplet<T>(mult_offset, B_offset + i, mult(i)));
            }
        }

        // add rhs
        RHS.block(cell_LHS_offset, 0, cbs_A, 1) += rhs.block(0, 0, cbs_A, 1);

    } // end assemble()


    vector_type
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl, const vector_type& sol) const
    {
        auto num_faces = howmany_faces(msh, cl);
        auto dim       = Mesh::dimension;
        auto cell_ofs  = offset(msh, cl);

        vector_type svel(cbs_A + num_faces * fbs_A);
        svel.block(0, 0, cbs_A, 1) = sol.block(cell_ofs * cbs_A, 0, cbs_A, 1);
        auto fcs                   = faces(msh, cl);
        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto       fc      = fcs[i];
            const auto face_id = msh.lookup(fc);

            if (m_bnd.is_dirichlet_face(face_id))
            {
                auto velocity = m_bnd.dirichlet_boundary_func(face_id);

                svel.block(cbs_A + i * fbs_A, 0, fbs_A, 1) =
                    project_function(msh, fc, di.face_degree(), velocity, di.face_degree());
            }
            else
            {
                auto face_ofs   = offset(msh, fc);
                auto global_ofs = cbs_A * msh.cells_size() + compress_table.at(face_ofs) * fbs_A;
                svel.block(cbs_A + i * fbs_A, 0, fbs_A, 1) = sol.block(global_ofs, 0, fbs_A, 1);
            }
        }
        return svel;
    }


    vector_type
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl, const vector_type& sol) const
    {
        auto cell_ofs = offset(msh, cl);
        auto pres_ofs = cbs_A * msh.cells_size() + fbs_A * num_other_faces + cbs_B * cell_ofs;

        vector_type spres = sol.block(pres_ofs, 0, cbs_B, 1);
        return spres;
    }

    size_t
    num_assembled_faces() const
    {
        return num_other_faces;
    }

    size_t
    global_system_size() const
    {
        return system_size;
    }

    auto
    global_face_offset(const Mesh& msh, const typename Mesh::face_type& fc) const
    {
        auto cbs_A = disk::vector_basis_size(di.cell_degree(), Mesh::dimension, Mesh::dimension);
        auto fbs_A = disk::vector_basis_size(di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
        auto cbs_B = disk::scalar_basis_size(di.face_degree(), Mesh::dimension);

        auto face_offset = offset(msh, fc);
        return cbs_A * msh.cells_size() + compress_table.at(face_offset) * fbs_A;
    }

    void
    impose_neumann_boundary_conditions(const Mesh& msh)
    {
        if (m_bnd.nb_faces_neumann() == 0)
            return;

        for (auto itor = msh.boundary_faces_begin(); itor != msh.boundary_faces_end(); itor++)
        {
            const auto bfc     = *itor;
            const auto face_id = msh.lookup(bfc);

            if (m_bnd.is_neumann_face(face_id))
            {
                if (m_bnd.is_dirichlet_face(face_id))
                {
                    throw std::invalid_argument("You tried to impose"
                                                "both Dirichlet and Neumann conditions on the same face");
                }
                else if (m_bnd.is_robin_face(face_id))
                {
                    throw std::invalid_argument("You tried to impose"
                                                "both Robin and Neumann conditions on the same face");
                }
                else
                {
                    const size_t                face_degree = di.face_degree();
                    auto fbs_A = disk::vector_basis_size(di.face_degree(), Mesh::dimension - 1, Mesh::dimension);
                    std::vector<assembly_index> asm_map;
                    asm_map.reserve(fbs_A);

                    auto face_LHS_offset = global_face_offset(msh, bfc);

                    for (size_t i = 0; i < fbs_A; i++)
                    {
                        asm_map.push_back(assembly_index(face_LHS_offset + i, true));
                    }

                    const auto fb = disk::make_vector_monomial_basis(msh, bfc, di.face_degree());
                    const vector_type neumann =
                        make_rhs(msh, bfc, fb, m_bnd.neumann_boundary_func(face_id), di.face_degree());

                    assert(neumann.size() == fbs_A);
                    for (size_t i = 0; i < neumann.size(); i++)
                    {
                        RHS[ asm_map[i] ] += neumann(i);
                    }
                }
            }
        }
    }

    void finalize(void)
    {
        LHS.setFromTriplets(triplets.begin(), triplets.end());
        triplets.clear();
    }
};


template<typename Mesh, typename BoundaryType>
auto
make_interface_assembler(const Mesh& msh, disk::hho_degree_info hdi, const BoundaryType& bnd)
{
    return interface_assembler<Mesh>(msh, hdi, bnd);
}
//////////////////////////////////////////////////////

template<typename Mesh, typename Velocity, typename Pressure, typename Assembler>
auto
compute_errors(const Mesh& msh,
                const disk::dynamic_vector<typename Mesh::coordinate_type>& sol,
                const typename disk::hho_degree_info & hdi,
                const Velocity& velocity,
                const Pressure& pressure,
                const Assembler& assembler,
                const bool& use_sym_grad)
{
    typedef Mesh mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;

    auto dim =  Mesh::dimension;

    scalar_type factor = (use_sym_grad)? 2. : 1.;

    scalar_type error(0), error_vel(0), error_pres(0);

    for (auto& cl : msh)
    {
    	auto bar = barycenter(msh, cl);
    	Matrix<scalar_type, Dynamic, 1> p = project_function(msh, cl, hdi, velocity);
    	auto cbs = disk::vector_basis_size(hdi.cell_degree(), dim, dim);
    	auto cell_ofs = disk::offset(msh, cl);
    	Matrix<scalar_type, Dynamic, 1> s = sol.block(cell_ofs * cbs, 0, cbs, 1);
    	Matrix<scalar_type, Dynamic, 1> diff = s - p.head(cbs);
    	auto cb = disk::make_vector_monomial_basis(msh, cl, hdi.cell_degree());
    	Matrix<scalar_type, Dynamic, Dynamic> mm = disk::make_mass_matrix(msh, cl, cb);
    	error += diff.dot(mm * diff);
    	//ofs << bar.x() << " " << bar.y() << " " << s(0) << " " << s(1) << std::endl;

        //pressure error
        Matrix<scalar_type, Dynamic, 1> ppres = disk::project_function(msh, cl, hdi.face_degree(), pressure);
        auto fbs = disk::vector_basis_size(hdi.face_degree(), dim - 1, dim);
        auto pbs = disk::scalar_basis_size(hdi.face_degree(), dim);
        auto pb  = disk::make_scalar_monomial_basis(msh, cl, hdi.face_degree());
        auto num_other_faces = assembler.num_assembled_faces();
        auto pres_ofs = cbs * msh.cells_size() + fbs * num_other_faces + pbs * cell_ofs;

        Matrix<scalar_type, Dynamic, 1> spres = sol.block(pres_ofs, 0, pbs, 1);
    	Matrix<scalar_type, Dynamic, 1> diff_pres = spres - ppres.head(pbs);
    	Matrix<scalar_type, Dynamic, Dynamic> scalar_mm = disk::make_mass_matrix(msh, cl, pb);
    	error_pres += diff_pres.dot(scalar_mm*diff_pres);

        //energy error
        auto num_faces = howmany_faces(msh, cl);
        Matrix<scalar_type, Dynamic, 1> svel(cbs + num_faces * fbs );
        svel.block(0, 0, cbs, 1) = sol.block(cell_ofs * cbs, 0, cbs, 1);
        auto fcs = faces(msh, cl);
        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto fc = fcs[i];

            if (msh.is_boundary(fc))
            {
                svel.block(cbs + i * fbs, 0, fbs, 1) = project_function(msh, fc, hdi.face_degree(), velocity, 2);
            }
            else
            {
                auto face_offset = assembler.global_face_offset(msh, fc);
                svel.block(cbs + i*fbs, 0, fbs, 1) = sol.block(face_offset, 0, fbs, 1);
            }
        }
        Matrix<scalar_type, Dynamic, 1> diff_vel = svel - p;
        auto gr = disk::make_hho_stokes(msh, cl, hdi, use_sym_grad);
        Matrix<scalar_type, Dynamic, Dynamic> stab;
        stab = make_vector_hho_stabilization(msh, cl, gr.first, hdi);
        error_vel += diff_vel.dot(factor * (gr.second + stab)*diff_vel);
    }

    //ofs.close();
    return std::make_pair(std::sqrt(error_vel), std::sqrt(error_pres));
}

/////////////////////////////////////////////

template<typename Mesh>
auto
run_interface(const Mesh& msh, size_t degree)
{
    typedef Mesh mesh_type;
    typedef typename mesh_type::cell        cell_type;
    typedef typename mesh_type::face        face_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::coordinate_type T;
    typedef disk::static_vector<T, 2>       result_type;


    typedef disk::dynamic_matrix<scalar_type>     matrix_type;

    // exact solution and rhs function
    auto rhs_fun = make_rhs_function(msh);
    auto velocity = make_velocity_function(msh);
    auto pressure = make_pressure_function(msh);


    // we use polynomials of degree k+1 on cells and k on faces (velocity)
    typename disk::hho_degree_info hdi(degree + 1, degree);

    // choose Dirichlet boundary conditions
    typedef disk::vector_boundary_conditions<Mesh> boundary_type;
    boundary_type bnd(msh);
    // bnd.addDirichletEverywhere(velocity);

    auto zero = [](const disk::point<T, 2>& p) -> result_type { return result_type{0.0, 0}; };
    auto neumann = [](const disk::point<T, 2>& p) -> result_type {
                       return result_type{std::pow(p.y(), 5.)  - 1./3.,
                               2.*p.y()*p.y()*(p.y()-1)*(p.y()-1)}; };
    scalar_type p_in = 1000.;
    auto p_neumann = [p_in](const disk::point<T, 2>& p) -> result_type {return result_type{p_in,0.}; };
    // on netgen meshes : id = 2 on x=0 and id = 1 on the other parts of the boundary
    if(false) {
        bnd.addDirichletBC(disk::DIRICHLET, 1, zero);
        bnd.addNeumannBC(disk::NEUMANN, 2, neumann);
    }
    // on gmsh meshes : id = 0 on y=0, id = 1 on x=1, id = 2 on y=1, id = 3 on x=0
    else
    {
        bnd.addDirichletBC(disk::DIRICHLET, 0, zero);
        bnd.addNeumannBC(disk::NEUMANN, 1, zero);
        bnd.addDirichletBC(disk::DIRICHLET, 2, zero);
        bnd.addNeumannBC(disk::NEUMANN, 3, p_neumann);
    }

    // assembler for the problem
    auto assembler = make_interface_assembler(msh, hdi, bnd);

    // assembly loop on the cells
    scalar_type factor = 1.; // used for symmetric gradient
    for (auto cl : msh)
    {
        auto gr = disk::make_hho_stokes(msh, cl, hdi, false); // vector Laplacian operator
        Matrix<scalar_type, Dynamic, Dynamic> stab;
        stab = make_vector_hho_stabilization(msh, cl, gr.first, hdi);
        auto dr = make_hho_divergence_reconstruction_rhs(msh, cl, hdi); // divergence reconstruction
        auto cell_basis = disk::make_vector_monomial_basis(msh, cl, hdi.cell_degree());
        auto rhs = make_rhs(msh, cl, cell_basis, rhs_fun);
        assembler.assemble(msh, cl, factor * (gr.second + stab), -dr, rhs);
    }
    assembler.impose_neumann_boundary_conditions(msh);
    assembler.finalize();

    size_t systsz = assembler.LHS.rows();
    disk::dynamic_vector<scalar_type> sol = disk::dynamic_vector<scalar_type>::Zero(systsz);

    // solving the linear problem using pardiso
    disk::solvers::pardiso_params<scalar_type> pparams;
    mkl_pardiso_ldlt(pparams, assembler.LHS, assembler.RHS, sol);


    /* export solution */
    disk::silo_database silo_db;
    silo_db.create("stokes.silo");
    silo_db.add_mesh(msh, "mesh");

    Matrix<scalar_type, Dynamic, 1> data_p = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());
    Matrix<scalar_type, Dynamic, 1> data_vx = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());
    Matrix<scalar_type, Dynamic, 1> data_vy = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());

    size_t cpt = 0;
    for (auto cl : msh)
    {
        auto cell_p = assembler.take_pressure(msh,cl,sol);
        auto cell_v = assembler.take_velocity(msh,cl,sol);

        data_p(cpt) = cell_p(0);
        data_vx(cpt) = cell_v(0);
        data_vy(cpt) = cell_v(1);

        cpt++;
    }
    silo_db.add_variable("mesh", "pressure", data_p, disk::zonal_variable_t );
    silo_db.add_variable("mesh", "vx", data_vx, disk::zonal_variable_t );
    silo_db.add_variable("mesh", "vy", data_vy, disk::zonal_variable_t );
    silo_db.close();

    // compute H^1 velocity error and L^2 pressure error
    auto err = compute_errors(msh, sol, hdi, velocity, pressure, assembler, false);
    std::cout << "velocity error = " << err.first << std::endl;
    std::cout << "pressure error = " << err.second << std::endl;

    return err;
}

////////////////////////////////////////////////////////////////

void convergence_test_typ1(void)
{
    using T = double;
    bool use_sym_grad = true;
    std::vector<std::string> meshfiles;

    meshfiles.push_back("../../../diskpp/meshes/gmsh/gmsh1.geo2s");
    meshfiles.push_back("../../../diskpp/meshes/gmsh/gmsh2.geo2s");
    meshfiles.push_back("../../../diskpp/meshes/gmsh/gmsh3.geo2s");
    meshfiles.push_back("../../../diskpp/meshes/gmsh/gmsh4.geo2s");
    meshfiles.push_back("../../../diskpp/meshes/gmsh/gmsh5.geo2s");
    meshfiles.push_back("../../../diskpp/meshes/gmsh/gmsh6.geo2s");

    /*
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri02.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri04.mesh2d");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/tri05.mesh2d");
    */

    /*
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");
    //meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_6.typ1");
    */

    /*
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_5.typ1");
    */
    /*
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_hex/fvca5/hexagonal_5.typ1");
    */
    std::cout << "                   velocity H1-error";
    std::cout << "    -     pressure L2-error "<< std::endl;

    for (size_t k = 0; k < 2; k++)
    {
        std::cout << "DEGREE " << k << std::endl;

        std::vector<T> mesh_hs;
        std::vector<std::pair<T,T>> errors;

        for (size_t i = 0; i < meshfiles.size(); i++)
        {
            // typedef disk::generic_mesh<T, 2>  mesh_type;
            // typedef disk::simplicial_mesh<T, 2>  mesh_type;

            // mesh_type msh;

            /*
            disk::fvca5_mesh_loader<T, 2> loader;
            if (!loader.read_mesh(meshfiles.at(i)))
            {
                std::cout << "Problem loading mesh." << std::endl;
                continue;
            }
            loader.populate_mesh(msh);
            */
            /*
            disk::netgen_mesh_loader<T, 2> loader;
            if (!loader.read_mesh(meshfiles.at(i)))
            {
                std::cout << "Problem loading mesh." << std::endl;
                continue;
            }
            loader.populate_mesh(msh);
            */


            disk::simplicial_mesh<T,2> msh;
            disk::gmsh_geometry_loader< disk::simplicial_mesh<T,2> > loader;

            if (!loader.read_mesh(meshfiles.at(i)))
            {
                std::cout << "Problem loading mesh." << std::endl;
                continue;
            }
            loader.populate_mesh(msh);

            auto error = run_interface(msh, k);

            mesh_hs.push_back( disk::average_diameter(msh) );
            errors.push_back(error);
        }

        for (size_t i = 0; i < mesh_hs.size(); i++)
        {
            if (i == 0)
            {
                std::cout << "    ";
                std::cout << std::scientific << std::setprecision(4) << mesh_hs.at(i) << "    ";
                std::cout << std::scientific << std::setprecision(4) << errors.at(i).first;
                std::cout << "     -- " << "          ";
                std::cout << std::scientific << std::setprecision(4) << errors.at(i).second;
                std::cout << "     -- " << std::endl;
            }
            else
            {
                auto rate = std::log( errors.at(i).first/errors.at(i-1).first ) /
                            std::log( mesh_hs.at(i)/mesh_hs.at(i-1) );
                std::cout << "    ";
                std::cout << std::scientific  << std::setprecision(4) << mesh_hs.at(i) << "    ";
                std::cout << std::scientific  << std::setprecision(4) << errors.at(i).first << "    ";
                std::cout << std::fixed<< std::setprecision(2) << rate << "          ";

                auto pres_rate = std::log( errors.at(i).second/errors.at(i-1).second ) /
                            std::log( mesh_hs.at(i)/mesh_hs.at(i-1) );
                std::cout << std::scientific  << std::setprecision(4) << errors.at(i).second << "    ";
                std::cout << std::fixed << std::setprecision(2) << pres_rate << std::endl;
            }
        }
    }
}


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
void test_Neumann_meshes(const Mesh<T, 2, Storage>& msh) {
    std::cout << "starting tests for the Neumann meshes" << std::endl;

    typedef Mesh<T, 2, Storage>                            mesh_type;
    typedef disk::vector_boundary_conditions<mesh_type>    Bnd_type;
    typedef disk::static_vector<T, 2>                      result_type;


    auto zero = [](const disk::point<T, 2>& p) -> result_type { return result_type{0.0, 0}; };

    Bnd_type bnd(msh);
    bnd.addDirichletBC(disk::DIRICHLET, 0, zero);
    // bnd.addDirichletBC(disk::DIRICHLET, 2, zero);
    // bnd.addDirichletBC(disk::DIRICHLET, 2, zero);
    bnd.addNeumannBC(disk::NEUMANN, 1, zero);
    // bnd.addNeumannBC(disk::NEUMANN, 3, zero);
    // bnd.addNeumannBC(disk::NEUMANN, 0, zero);

    size_t num_all_faces = msh.faces_size();
    for (size_t i = 0; i < num_all_faces; i++)
    {
        auto fc = *std::next(msh.faces_begin(), i);
        // auto fc_id = msh.lookup(fc);
        auto bar = barycenter(msh,fc);
        if ( bnd.is_dirichlet_face(fc) )
        {
            std::cout << "Dirichlet face in : " << bar << std::endl;
        }
    }

    for (size_t i = 0; i < num_all_faces; i++)
    {
        auto fc = *std::next(msh.faces_begin(), i);
        // auto fc_id = msh.lookup(fc);
        auto bar = barycenter(msh,fc);
        if ( bnd.is_neumann_face(fc) )
        {
            std::cout << "Neumann face in : " << bar << std::endl;
        }
    }

    std::cout << std::endl;
}


void run_Neumann_tests(void)
{
    using T = double;
    // auto msh = disk::load_netgen_2d_mesh<T>("../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d");
    // auto msh = disk::load_netgen_2d_mesh<T>("../../../diskpp/meshes/2D_triangles/netgen/tri02.mesh2d");
    // auto msh = disk::load_netgen_2d_mesh<T>("../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d");
    // auto msh = disk::load_fvca5_2d_mesh<T>("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");

    disk::simplicial_mesh<T,2> msh;
    disk::gmsh_geometry_loader< disk::simplicial_mesh<T,2> > loader;

    // loader.read_mesh("../../../diskpp/meshes/gmsh/gmsh1.geo2s");
    loader.read_mesh("../../../diskpp/meshes/gmsh/gmsh2.geo2s");
    // loader.read_mesh("../../../diskpp/meshes/gmsh/gmsh3.geo2s");
    // loader.read_mesh("../../../diskpp/meshes/gmsh/gmsh4.geo2s");
    loader.populate_mesh(msh);

    std::cout << "mesh loaded !" << std::endl;

    test_Neumann_meshes(msh);
    run_interface(msh, 0);

}


// int main(int argc, char **argv)
// {
//     cout << "end main" << endl;
//     return 0;
// }


int main(void)
{

    // run_Neumann_tests();
    convergence_test_typ1();
    std::cout << "end main" << std::endl;
    return 1;
}
