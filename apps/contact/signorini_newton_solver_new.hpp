/*
*       /\         DISK++, a template library for DIscontinuous SKeletal
*      /__\        methods.
*     /_\/_\
*    /\    /\      Matteo Cicuttin (C) 2016, 2017, 2018
*   /__\  /__\     matteo.cicuttin@enpc.fr
*  /_\/_\/_\/_\    École Nationale des Ponts et Chaussées - CERMICS
*
* This file is copyright of the following authors:
* Karol Cascavita (C) 2018                     karol.cascavita@enpc.fr
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

#include "revolution/bases"
#include "revolution/quadratures"
#include "revolution/methods/hho"


#include "output/silo.hpp"
#include "common_temporal.hpp"
#include "solvers/solver.hpp"

template<typename Mesh, typename Function, typename Analytical>
//dynamic_vector<typename Mesh::coordinate_type> //This is for hierarchical (solve conflicts!!)
std::pair<typename Mesh::coordinate_type, typename Mesh::coordinate_type>
solve_cells_full(const Mesh&  msh, const Function& rhs_fun, const Analytical& sol_fun,
    const algorithm_parameters<typename Mesh::coordinate_type>& ap,
    const disk::mechanics::BoundaryConditionsScalar<Mesh>& bnd)
{
    std::cout << "Im in CELLS FULL in NEW" << std::endl;
    std::cout << ap << std::endl;
    using T = typename Mesh::coordinate_type;

    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>    matrix_type;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1>                 vector_type;

    auto is_contact_vector = make_is_contact_vector(msh, bnd);

    hho_degree_info      hdi(ap.degree +1, ap.degree); //Not allow (degree, degree)
    std::cout << " * cell degree :"<< hdi.cell_degree() << std::endl;
    std::cout << " * face degree :"<< hdi.face_degree() << std::endl;

    auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension-1);
    auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);


    auto num_full_dofs = cbs*msh.cells_size() + 2 * fbs*msh.faces_size()
                                    - fbs*msh.boundary_faces_size() ;

    auto offset_vector = full_offset(msh, hdi);

    dynamic_vector<T>  full_sol = dynamic_vector<T>::Zero(num_full_dofs);


    auto max_iter = 15;
    auto tol = 1.e-6;

    for(size_t iter = 0; iter < max_iter; iter++)
    {
        auto assembler = make_contact_full_assembler_new(msh, hdi, bnd);

        assembler.imposed_dirichlet_boundary_conditions(msh, bnd, full_sol);

        auto cl_count = 0;
        for (auto& cl : msh)
        {
            auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());

            auto cell_ofs = offset_vector.at(cl_count);
            auto num_total_dofs = cbs + howmany_faces(msh, cl) * fbs;

            vector_type  u_full = full_sol.block(cell_ofs, 0, num_total_dofs, 1);

            vector_type b  = vector_type::Zero(num_total_dofs);
            matrix_type A  = matrix_type::Zero(num_total_dofs, num_total_dofs);

            if (is_contact_vector.at(cl_count) == 1)
            {
                auto gr   = make_hho_contact_scalar_laplacian(msh, cl, hdi, bnd);
                auto stab = make_hdg_scalar_stabilization(msh, cl, hdi);

                matrix_type Ah  = gr.second + stab;
                vector_type Lh  = make_rhs(msh, cl, cb, rhs_fun);//, hdi.cell_degree());

                matrix_type  Anitsche   = make_hho_nitsche(msh, cl, hdi, gr.first, ap.gamma_0, ap.theta, bnd );

                //vector_type  Bnegative  = make_hho_negative_par(msh, cl, hdi, gr.first, ap.gamma_0, ap.theta, bnd, u_full, 1.);
                //matrix_type  Aheaviside = make_hho_heaviside_par(msh, cl, hdi, gr.first, ap.gamma_0, ap.theta, bnd, u_full, 1.);

                vector_type  Bnegative  = make_hho_negative(msh, cl, hdi, gr.first, ap.gamma_0, ap.theta, bnd, u_full);
                matrix_type  Aheaviside = make_hho_heaviside(msh, cl, hdi, gr.first, ap.gamma_0, ap.theta, bnd, u_full);

                //Original
                A =   Ah - Anitsche + Aheaviside;
                b = -(Ah - Anitsche) * u_full - Bnegative;
                b.block(0, 0, cbs, 1) += Lh;
            }
            else
            {
                auto gr   = make_hho_scalar_laplacian(msh, cl, hdi);
                auto stab = make_hdg_scalar_stabilization(msh, cl, hdi);

                matrix_type Ah = gr.second + stab;
                vector_type Lh = make_rhs(msh, cl, cb, rhs_fun);//, hdi.cell_degree());

                A = Ah;
                b = -Ah * u_full;
                b.block(0, 0, cbs, 1) += Lh;
            }

            assembler.assemble(msh, cl, A, b);

            cl_count++;
        }

        assembler.impose_neumann_boundary_conditions(msh, bnd);
        assembler.finalize();

        size_t systsz = assembler.LHS.rows();
        size_t nnz = assembler.LHS.nonZeros();

        dynamic_vector<T> dsol = dynamic_vector<T>::Zero(systsz);

        disk::solvers::pardiso_params<T> pparams;
        pparams.report_factorization_Mflops = true;
        mkl_pardiso(pparams, assembler.LHS, assembler.RHS, dsol);


        T H1_increment  = 0.0 ;
        T L2_increment  = 0.0 ;

        cl_count = 0;
        dynamic_vector<T> diff_sol = dynamic_vector<T>::Zero(num_full_dofs);

        for (auto& cl : msh)
        {
            const auto cell_ofs = offset_vector.at(cl_count);
            const auto num_total_dofs = cbs + howmany_faces(msh, cl) * fbs;

            vector_type  u_full = full_sol.block(cell_ofs, 0, num_total_dofs, 1);
            matrix_type  Ah  = matrix_type::Zero(num_total_dofs, num_total_dofs);
            vector_type  du_full  = vector_type::Zero(num_total_dofs);

            auto stab = make_hdg_scalar_stabilization(msh, cl, hdi);

            if (is_contact_vector.at(cl_count)==1)
            {
                auto gr  = make_hho_contact_scalar_laplacian(msh, cl, hdi, bnd);
                Ah  = gr.second + stab;
            }
            else
            {
                auto gr  = make_hho_scalar_laplacian(msh, cl, hdi);
                Ah  = gr.second + stab;
            }

            //Erase this function, since it coulb be used to take data from full_sol
            //dsol has zero dirichlet conditions so nothing is computed in this faces/
            du_full = assembler.take_local_data_increment(msh, cl, dsol);
            diff_sol.block(cell_ofs, 0, num_total_dofs ,1) = du_full;

            auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
            matrix_type mass  = make_mass_matrix(msh, cl, cb);//, hdi.cell_degree());
            vector_type u_diff = du_full.block(0, 0, cbs, 1);

            H1_increment += du_full.dot(Ah * du_full);
            L2_increment += u_diff.dot(mass * u_diff);

            cl_count++;
        }


        full_sol += diff_sol;

        std::cout << "  "<< iter << "  "<< std::scientific<< std::setprecision(3);
        std::cout << std::sqrt(H1_increment)<< "   "<< std::sqrt(L2_increment)<< std::endl;

        if( std::sqrt(H1_increment)  < tol)
        {
            std::ofstream efs("solution_whho_cnew_i" + tostr(iter) + ".dat");

            if(!efs.is_open())
                std::cout<< "Error opening file"<<std::endl;


            T H1_error  = 0.0 ;
            T L2_error  = 0.0 ;


            auto cl_count = 0;
            for(auto& cl : msh)
            {
                const auto cell_ofs = offset_vector.at(cl_count);
                const auto num_total_dofs = cbs + howmany_faces(msh, cl) * fbs;

                vector_type  u_full   = full_sol.block(cell_ofs, 0, num_total_dofs, 1);
                vector_type  du_full  = diff_sol.block(cell_ofs, 0, num_total_dofs, 1);
                matrix_type  Ah  = matrix_type::Zero(num_total_dofs, num_total_dofs);

                auto stab = make_hdg_scalar_stabilization(msh, cl, hdi);

                if (is_contact_vector.at(cl_count)==1)
                {
                    auto gr  = make_hho_contact_scalar_laplacian(msh, cl, hdi, bnd);
                    Ah  = gr.second + stab;
                }
                else
                {
                    auto gr  = make_hho_scalar_laplacian(msh, cl, hdi);
                    Ah  = gr.second + stab;
                }

                vector_type realsol = project_function(msh, cl, hdi, sol_fun, 2);

                vector_type diff = realsol - u_full;
                H1_error += diff.dot(Ah*diff);

                auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
                matrix_type mass  = make_mass_matrix(msh, cl, cb, hdi.cell_degree());
                vector_type u_diff = diff.block(0, 0, cbs, 1);
                L2_error += u_diff.dot(mass * u_diff);

                //plot
                auto bar = barycenter(msh, cl);

                efs << bar.x() << " " << bar.y() <<" "<< u_full(0) << " "<< du_full(0) << std::endl;
                cl_count++;

            }
            efs.close();

            return std::make_pair(std::sqrt(H1_error), std::sqrt(L2_error));
        }

    }
    return std::make_pair(0., 0.);
}

template<typename Mesh>
std::pair<typename Mesh::coordinate_type, typename Mesh::coordinate_type>
run_signorini_unknown(  const Mesh& msh, const algorithm_parameters<typename Mesh::coordinate_type>& ap,
                const typename Mesh::coordinate_type& parameter)
{
    typedef typename Mesh::point_type  point_type;
    using T =  typename Mesh::coordinate_type;


    dump_to_matlab(msh,"mesh.m");

    auto force = [](const point_type& p) -> T {
        return - 2.* M_PI *  std::sin(2. * M_PI * p.x());
    };

    auto zero_fun = [](const point_type& p) -> T {
        return 0.;
    };

    typedef disk::mechanics::BoundaryConditionsScalar<Mesh> boundary_type;
    boundary_type  bnd(msh);

    /*--------------------------------------------------------------------------
    *  Check boundary labels for the unitary square domain
    *          Netgen     _____          Medit     _____
    *                4   |     | 2                |     |
    *                    |_____|                  |_____|
    *                       3                        2
    *-------------------------------------------------------------------------*/

    bnd.addDirichletBC(disk::mechanics::DIRICHLET,1, zero_fun); //TOP
    bnd.addNeumannBC(disk::mechanics::NEUMANN, 2, zero_fun); //
    bnd.addNeumannBC(disk::mechanics::NEUMANN, 4, zero_fun); //
    //bnd.addNeumannBC(disk::mechanics::NEUMANN, 3, zero_fun); //TOP
    bnd.addContactBC(disk::mechanics::SIGNORINI,3); //BOTTOM

    switch (ap.solver)
    {
        case EVAL_IN_CELLS_FULL:
            return solve_cells_full(msh, force, zero_fun, ap, bnd);
            break;
        default:
            throw std::invalid_argument("Invalid solver");
    }
}

template<typename Mesh>
std::pair<typename Mesh::coordinate_type, typename Mesh::coordinate_type>
run_signorini_analytical(  const Mesh& msh, const algorithm_parameters<typename Mesh::coordinate_type>& ap,
                const typename Mesh::coordinate_type& parameter)
{
    typedef typename Mesh::point_type  point_type;
    using T =  typename Mesh::coordinate_type;


    dump_to_matlab(msh,"mesh.m");

    auto force = [](const point_type& p) -> T {
        return 0.;
    };

    auto zero_fun = [](const point_type& p) -> T {
        return 0.;
    };

    auto left = [](const point_type& p) -> T {
        T radio = std::sqrt(p.x()*p.x() + p.y()*p.y());
        T theta = std::atan2(p.y(), p.x());
        T sintcos = std::sin(theta) *std::cos(4.5 *theta);
        T sincost = std::sin(4.5 *theta) * std::cos(theta);
         return  4.5 * std::pow(radio, 3.5) *( sincost - sintcos);
    };

    auto right = [](const point_type& p) -> T {
        T radio = std::sqrt(p.x()*p.x() + p.y()*p.y());
        T theta = std::atan2(p.y(), p.x());
        T sintcos = std::sin(theta) *std::cos(4.5 *theta);
        T sincost = std::sin(4.5 *theta) * std::cos(theta);
         return  -4.5 * std::pow(radio, 3.5) *( sincost - sintcos);
    };


    auto fun = [](const point_type& p) -> T {
        T radio = std::sqrt(p.x()*p.x() + p.y()*p.y());
        T theta = std::atan2(p.y(), p.x());

        return  std::pow(radio, 4.5) * std::sin(4.5 *theta);
    };

    typedef disk::mechanics::BoundaryConditionsScalar<Mesh> boundary_type;
    boundary_type  bnd(msh);

    /*--------------------------------------------------------------------------
    *  Check boundary labels for the unitary square domain
    *          Netgen     _____          Medit     _____
    *                4   |     | 2                |     |
    *                    |_____|                  |_____|
    *                       3                        2
    *-------------------------------------------------------------------------*/

    bnd.addDirichletBC(disk::mechanics::DIRICHLET,1, fun); //TOP
    bnd.addDirichletBC(disk::mechanics::DIRICHLET,2, fun); //TOP
    bnd.addDirichletBC(disk::mechanics::DIRICHLET,4, fun); //TOP

    //bnd.addNeumannBC(disk::mechanics::NEUMANN, 2, left); //
    //bnd.addNeumannBC(disk::mechanics::NEUMANN, 4, right); //
    //bnd.addNeumannBC(disk::mechanics::NEUMANN, 3, zero_fun); //TOP
    bnd.addContactBC(disk::mechanics::SIGNORINI,3); //BOTTOM

    switch (ap.solver)
    {
        case EVAL_IN_CELLS_FULL:
            return solve_cells_full(msh, force, fun, ap, bnd);
            break;
        default:
            throw std::invalid_argument("Invalid solver");
    }
}

template<typename Mesh>
std::pair<typename Mesh::coordinate_type, typename Mesh::coordinate_type>
run_signorini(  const Mesh& msh, const algorithm_parameters<typename Mesh::coordinate_type>& ap,
                const typename Mesh::coordinate_type& parameter)
{
    return run_signorini_analytical(msh, ap, parameter);
}
