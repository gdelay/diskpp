/*
 *       /\        Matteo Cicuttin (C) 2016-2021
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
 *
 * This file is copyright of the following authors:
 * Guillaume Delay  (C) 2024          guillaume.delay@sorbonne-universite.fr
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
// #include "output/silo.hpp"
#include "timecounter.h"
#include "colormanip.h"
#include "bases/bases.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
// #include "solvers/mumps.hpp"



using namespace disk;
using namespace Eigen;
using namespace std;



///////////////   noise representation /////////////
// the time-space geometry is divided in sub-domains (M sub-domains along each dimension)
// the space domain is necesseraly \Omega = [0,1]^d
// a noise value is affected to each sub-domain (strored in tab)
template<typename Mesh>
class noise_representation;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class noise_representation< Mesh<T, 1, Storage> >
{
private:
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    size_t M;                     // number of domain sub-divisions along each dimension
    std::vector<scalar_type> tab; // tabular for the noise level in each sub-division

public:
    noise_representation(size_t M, scalar_type noise_size) : M(M)
    {
        size_t M_tot = pow(M,2); // total number of sub-divisions
        tab.reserve(M_tot);

        // we need to compute the noise for each sub-division
        for(int i=0; i<M_tot; i++)
            tab.push_back( noise_size * ((std::rand() % 200)-100) * 0.01 );
    }
    scalar_type operator()(const scalar_type t, const point_type& pt) const
    {
        scalar_type x = pt.x();
        assert(0 <= x && x <= 1);
        assert(0 <= t && t <= 2);

        size_t x_pos = x * M;
        size_t t_pos = (t/2.) * M;
        assert(x_pos < M && t_pos < M);

        return tab[t_pos*M + x_pos];
    }
};


template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class noise_representation< Mesh<T, 2, Storage> >
{
private:
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    size_t M;                     // number of domain sub-divisions along each dimension
    std::vector<scalar_type> tab; // tabular for the noise level in each sub-division

public:
    noise_representation(size_t M, scalar_type noise_size) : M(M)
    {
        size_t M_tot = pow(M,3); // total number of sub-divisions
        tab.reserve(M_tot);

        // we need to compute the noise for each sub-division
        for(int i=0; i<M_tot; i++)
            tab.push_back( noise_size * ((std::rand() % 200)-100) * 0.01 );
    }
    scalar_type operator()(const scalar_type t, const point_type& pt) const
    {
        scalar_type x = pt.x(), y = pt.y();
        assert(0 <= x && x <= 1);
        assert(0 <= y && y <= 1);
        assert(0 <= t && t <= 2);

        size_t x_pos = x * M;
        size_t y_pos = y * M;
        size_t t_pos = (t/2.) * M;
        assert(x_pos < M && y_pos < M && t_pos < M);

        return tab[t_pos*M*M + x_pos*M + y_pos];
    }
};

template<typename Mesh>
auto make_noise_representation(const Mesh& msh, size_t M, typename Mesh::coordinate_type noise_level)
{
    return noise_representation<Mesh>(M, noise_level);
}

/////////////////   noise file    //////////////////
template<size_t>
string init_noise_file();

template<> // dim = 1
string init_noise_file<1>()
{
    return "#x\tt\tnoise";
}

template<> // dim = 2
string init_noise_file<2>()
{
    return "#x\ty\tt\tnoise";
}


template<typename Mesh>
class line_noise_file;

// dim = 1
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class line_noise_file< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
public:
    std::string operator() (scalar_type t, const point_type& pt, scalar_type noise) {
        std::stringstream ss_noise;
        ss_noise << pt.x() << "\t" << t << "\t" << noise;
        return ss_noise.str();
    }
};

// dim = 2
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class line_noise_file< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
public:
    std::string operator() (scalar_type t, const point_type& pt, scalar_type noise)
    {
        std::stringstream ss_noise;
        ss_noise << pt.x() << "\t" << pt.y() << "\t" << t << "\t" << noise;
        return ss_noise.str();
    }
};

template<typename Mesh>
auto make_line_noise_file(const Mesh& msh)
{
    return line_noise_file<Mesh>();
}


//////////////////   test case   ///////////////////
/* finite dimensional trace space at initial time */
template<typename Mesh>
struct finite_trace_init;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct finite_trace_init< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, Dynamic, 1> function_type;
    typedef Matrix<scalar_type, Dynamic, Dynamic>  matrix_type;

    size_t basis_size;

    // constructor
    finite_trace_init(size_t basis_size) : basis_size(basis_size) {}

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // choose the basis functions here
        for (size_t k = 0; k < basis_size; k++)
        {
            // ret(k) = std::sqrt(2) * std::sin((k+1)*M_PI*pt.x());
            ret(k) = std::sqrt(2) * std::cos((k+1)*M_PI*pt.x());
            // ret(k) = 1.;
        }

        return ret;
    }

    // compute the mass matrix associated to this basis
    matrix_type
    make_mass_matrix(const mesh_type& msh) const
    {
        // in some particular cases, the matrix is identity
        return matrix_type::Identity(basis_size, basis_size);

        matrix_type mass_mat = matrix_type::Zero(basis_size,basis_size);
        for(auto& cl : msh) {
            // compute local mass matrices
            const auto qps = integrate(msh, cl, 4); // 4 is the integration degree here
            for (auto& qp : qps)
            {
                const auto phi = eval_functions(qp.point());
                mass_mat += qp.weight() * phi * phi.transpose();
            }
        }

        return mass_mat;
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct finite_trace_init< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, Dynamic, 1> function_type;
    typedef Matrix<scalar_type, Dynamic, Dynamic>  matrix_type;

    size_t basis_size;

    // constructor
    finite_trace_init(size_t basis_size) : basis_size(basis_size) {}

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // choose the basis functions here
        for (size_t k = 0; k < basis_size; k++)
        {
            // ret(k) = std::sqrt(2) * std::sin((k+1)*M_PI*pt.x());
            ret(k) = std::sqrt(2) * std::cos((k+1)*M_PI*pt.x()) * std::cos((k+1)*M_PI*pt.y());
            // ret(k) = 1.;
        }

        return ret;
    }

    // compute the mass matrix associated to this basis
    matrix_type
    make_mass_matrix(const mesh_type& msh) const
    {
        // in some particular cases, the matrix is identity
        // return matrix_type::Identity(basis_size, basis_size);

        matrix_type mass_mat = matrix_type::Zero(basis_size,basis_size);
        for(auto& cl : msh) {
            // compute local mass matrices
            const auto qps = integrate(msh, cl, 4); // 4 is the integration degree here
            for (auto& qp : qps)
            {
                const auto phi = eval_functions(qp.point());
                mass_mat += qp.weight() * phi * phi.transpose();
            }
        }

        return mass_mat;
    }
};


template<typename Mesh>
auto make_finite_trace_init(const Mesh& msh, size_t nb_basis)
{
    return finite_trace_init<Mesh>(nb_basis);
}

/* finite dimensional trace space on the boundary */
template<typename Mesh>
struct finite_trace_bound;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct finite_trace_bound< Mesh<T, 1, Storage> >
{
    typedef Mesh<T,1,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, Dynamic, 1> function_type;
    typedef Matrix<scalar_type, Dynamic, Dynamic>  matrix_type;

    size_t basis_size;

    // constructor
    finite_trace_bound(size_t basis_size) : basis_size(basis_size) {}

    function_type
    eval_functions(const T t, const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // choose the basis functions here
        for (size_t k = 0; k < basis_size; k++)
        {
            ret(k) = std::cos(0.5*(k+1)*M_PI*t) * std::cos(0.5*(k+1)*M_PI*pt.x());
            // ret(k) = pt.x();
            // ret(k) = std::sqrt(0.5) * std::cos(0.5*(k+1)*M_PI*t);
            // ret(k) = 1.;
        }

        return ret;
    }

    function_type
    eval_time_ders(const T t, const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // write the basis functions derivatives here
        for (size_t k = 0; k < basis_size; k++)
        {
            T freq = 0.5*(k+1)*M_PI;
            ret(k) = -freq*std::sin(freq*t) * std::cos(freq*pt.x());
            // ret(k) = 0.;
        }

        return ret;
    }

    matrix_type
    eval_gradients(const T t, const point_type& pt) const
    {
        matrix_type ret = matrix_type::Zero(basis_size, 1); // one dim

        // write the basis functions gradients here
        for (size_t k = 0; k < basis_size; k++)
        {
            T freq = 0.5*(k+1)*M_PI;
            ret(k) = -freq*std::cos(freq*t) * std::sin(freq*pt.x());
            ret(k) = 0.;
        }

        return ret;
    }

    bool is_dirichlet(const mesh_type& msh, const typename mesh_type::face_type& fc) {
        auto bar = barycenter(msh,fc);
        if(std::abs(bar.x() - 1) < 1e-6 or std::abs(bar.x()) < 1e-6 )
            return true;
        return false;
    }

    bool is_dirichlet(const mesh_type& msh, const typename mesh_type::cell_type& cl) {

        const auto fcs = faces(msh, cl);

        for(size_t face_i = 0; face_i < fcs.size(); face_i++) // loop on faces
        {
            auto fc = fcs[face_i];
            if( is_dirichlet(msh, fc) )
                return true;
        }

        return false;
    }

    // compute the mass matrix associated to this basis
    matrix_type
    make_mass_matrix(const mesh_type& msh, const disk::generic_mesh<T, 1>& time_msh) const
    {
        matrix_type mass_mat = matrix_type::Zero(basis_size,basis_size);

        for(auto& cl : msh) {
            // in some particular cases, the matrix is identity
            // return matrix_type::Identity(basis_size, basis_size);

            const auto fcs    = faces(msh, cl);
            for (size_t face_i = 0; face_i < fcs.size(); face_i++)
            {
                const auto fc = fcs[face_i];
                if(!this->is_dirichlet(msh, fc)) // loop on the dirichlet faces
                    continue;

                const auto qps_f = integrate(msh, fc, 4); // 4 is the integration degree here

                // compute the local contributions to the mass matrix
                for(auto& t_cl : time_msh)
                {
                    const auto qps_t = integrate(time_msh, t_cl, 4); // 4 is the integration degree here
                    // compute the local contribution to the mass matrix
                    for (auto& qpt : qps_t)
                    {
                        for (auto& qpf : qps_f)
                        {
                            const auto phi = eval_functions(qpt.point().x(), qpf.point());
                            mass_mat += qpt.weight() * qpf.weight() * phi * phi.transpose();
                        }
                    }
                }
            }
        }

        return mass_mat;
    }

    // compute the time-stiffness matrix associated to this basis
    matrix_type
    make_time_stiffness_matrix(const mesh_type& msh, const disk::generic_mesh<T, 1>& time_msh) const
    {
        matrix_type stiff_mat = matrix_type::Zero(basis_size,basis_size);

        for(auto& cl : msh) {
            const auto fcs    = faces(msh, cl);
            for (size_t face_i = 0; face_i < fcs.size(); face_i++)
            {
                const auto fc = fcs[face_i];
                if(!this->is_dirichlet(msh, fc)) // loop on the dirichlet faces
                    continue;

                const auto qps_f = integrate(msh, fc, 4); // 4 is the integration degree here

                // compute the local contributions to the time-stiffness matrix
                for(auto& t_cl : time_msh)
                {
                    const auto qps_t = integrate(time_msh, t_cl, 4); // 4 is the integration degree here
                    // compute the local contribution to the time-stiffness matrix
                    for (auto& qpt : qps_t)
                    {
                        for (auto& qpf : qps_f)
                        {
                            const auto dt_phi = eval_time_ders(qpt.point().x(), qpf.point());
                            stiff_mat += qpt.weight() * qpf.weight() * dt_phi * dt_phi.transpose();
                        }
                    }
                }
            }
        }

        return stiff_mat;
    }

    // compute the space-tangential-stiffness matrix associated to this basis
    matrix_type
    make_tang_stiffness_matrix(const mesh_type& msh, const disk::generic_mesh<T, 1>& time_msh) const
    {
        // this matrix is null in one dimension
        // (tangential component only)
        return matrix_type::Zero(basis_size,basis_size);
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct finite_trace_bound< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;
    typedef Matrix<scalar_type, Dynamic, 1> function_type;
    typedef Matrix<scalar_type, Dynamic, Dynamic>  matrix_type;

    size_t basis_size;

    // constructor
    finite_trace_bound(size_t basis_size) : basis_size(basis_size) {}

    function_type
    eval_functions(const T t, const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // choose the basis functions here
        for (size_t k = 0; k < basis_size; k++)
        {
            ret(k) = std::cos(0.5*(k+1)*M_PI*t) * std::cos(0.5*(k+1)*M_PI*pt.x()) * std::cos(0.5*(k+1)*M_PI*pt.y());
            // ret(k) = pt.x();
            // ret(k) = std::sqrt(0.5) * std::cos(0.5*(k+1)*M_PI*t);
            // ret(k) = 1.;
        }

        return ret;
    }

    function_type
    eval_time_ders(const T t, const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // write the basis functions derivatives here
        for (size_t k = 0; k < basis_size; k++)
        {
            T freq = 0.5*(k+1)*M_PI;
            ret(k) = -freq*std::sin(freq*t) * std::cos(freq*pt.x()) * std::cos(freq*pt.y());
            // ret(k) = 0.;
        }

        return ret;
    }

    matrix_type
    eval_gradients(const T t, const point_type& pt) const
    {
        matrix_type ret = matrix_type::Zero(basis_size, 2); // 2 dim

        // write the basis functions gradients here
        for (size_t k = 0; k < basis_size; k++)
        {
            T freq = 0.5*(k+1)*M_PI;
            ret(k,0) = -freq*std::cos(freq*t) * std::sin(freq*pt.x()) * std::cos(freq*pt.y());
            ret(k,1) = -freq*std::cos(freq*t) * std::cos(freq*pt.x()) * std::sin(freq*pt.y());
        }

        return ret;
    }

    bool is_dirichlet(const mesh_type& msh, const typename mesh_type::face_type& fc) {
        auto bar = barycenter(msh,fc);
        if(std::abs(bar.x() - 1) < 1e-6 or std::abs(bar.x()) < 1e-6 or std::abs(bar.y() - 1) < 1e-6 or std::abs(bar.y()) < 1e-6)
            return true;
        return false;
    }

    bool is_dirichlet(const mesh_type& msh, const typename mesh_type::cell_type& cl) {

        const auto fcs = faces(msh, cl);

        for(size_t face_i = 0; face_i < fcs.size(); face_i++) // loop on faces
        {
            auto fc = fcs[face_i];
            if( is_dirichlet(msh, fc) )
                return true;
        }

        return false;
    }

    // compute the mass matrix associated to this basis
    matrix_type
    make_mass_matrix(const mesh_type& msh, const disk::generic_mesh<T, 1>& time_msh) const
    {
        matrix_type mass_mat = matrix_type::Zero(basis_size,basis_size);

        for(auto& cl : msh) {
            // in some particular cases, the matrix is identity
            // return matrix_type::Identity(basis_size, basis_size);

            const auto fcs    = faces(msh, cl);
            for (size_t face_i = 0; face_i < fcs.size(); face_i++)
            {
                const auto fc = fcs[face_i];
                if(!this->is_dirichlet(msh, fc)) // loop on the dirichlet faces
                    continue;

                const auto qps_f = integrate(msh, fc, 4); // 4 is the integration degree here

                // compute the local contributions to the mass matrix
                for(auto& t_cl : time_msh)
                {
                    const auto qps_t = integrate(time_msh, t_cl, 4); // 4 is the integration degree here
                    // compute the local contribution to the mass matrix
                    for (auto& qpt : qps_t)
                    {
                        for (auto& qpf : qps_f)
                        {
                            const auto phi = eval_functions(qpt.point().x(), qpf.point());
                            mass_mat += qpt.weight() * qpf.weight() * phi * phi.transpose();
                        }
                    }
                }
            }
        }

        return mass_mat;
    }

    // compute the time-stiffness matrix associated to this basis
    matrix_type
    make_time_stiffness_matrix(const mesh_type& msh, const disk::generic_mesh<T, 1>& time_msh) const
    {
        matrix_type stiff_mat = matrix_type::Zero(basis_size,basis_size);

        for(auto& cl : msh) {
            const auto fcs    = faces(msh, cl);
            for (size_t face_i = 0; face_i < fcs.size(); face_i++)
            {
                const auto fc = fcs[face_i];
                if(!this->is_dirichlet(msh, fc)) // loop on the dirichlet faces
                    continue;

                const auto qps_f = integrate(msh, fc, 4); // 4 is the integration degree here

                // compute the local contributions to the time-stiffness matrix
                for(auto& t_cl : time_msh)
                {
                    const auto qps_t = integrate(time_msh, t_cl, 4); // 4 is the integration degree here
                    // compute the local contribution to the time-stiffness matrix
                    for (auto& qpt : qps_t)
                    {
                        for (auto& qpf : qps_f)
                        {
                            const auto dt_phi = eval_time_ders(qpt.point().x(), qpf.point());
                            stiff_mat += qpt.weight() * qpf.weight() * dt_phi * dt_phi.transpose();
                        }
                    }
                }
            }
        }

        return stiff_mat;
    }

    // compute the space-tangential-stiffness matrix associated to this basis
    matrix_type
    make_tang_stiffness_matrix(const mesh_type& msh, const disk::generic_mesh<T, 1>& time_msh) const
    {
        // compute this matrix
        matrix_type tang_mat = matrix_type::Zero(basis_size,basis_size);

        for(auto& cl : msh) {
            const auto fcs    = faces(msh, cl);
            for (size_t face_i = 0; face_i < fcs.size(); face_i++)
            {
                const auto fc = fcs[face_i];
                if(!this->is_dirichlet(msh, fc)) // loop on the dirichlet faces
                    continue;

                const auto qps_f = integrate(msh, fc, 4); // 4 is the integration degree here

                const auto n  = normal(msh, cl, fc); // normal vector

                // compute the local contributions to the tangential-stiffness matrix
                for(auto& t_cl : time_msh)
                {
                    const auto qps_t = integrate(time_msh, t_cl, 4); // 4 is the integration degree here
                    // compute the local contribution to the tangential-stiffness matrix
                    for (auto& qpt : qps_t)
                    {
                        for (auto& qpf : qps_f)
                        {
                            const auto fs_dphi = eval_gradients(qpt.point().x(), qpf.point());
                            const auto fs_dphi_n = fs_dphi * n;
                            tang_mat += qpt.weight() * qpf.weight() * fs_dphi * fs_dphi.transpose(); // full stiffness matrix
                            tang_mat -= qpt.weight() * qpf.weight() * fs_dphi_n * fs_dphi_n.transpose(); // remove normal components
                        }
                    }
                }
            }
        }
        return matrix_type::Zero(basis_size,basis_size);
    }
};

template<typename Mesh>
auto make_finite_trace_bound(const Mesh& msh, size_t nb_basis)
{
    return finite_trace_bound<Mesh>(nb_basis);
}

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
        // return ( M_PI*M_PI*std::sin(M_PI * t) + M_PI * std::cos(M_PI * t) ) * std::sin( M_PI * pt.x() );
        // return ( M_PI*M_PI*std::cos(M_PI * t) - M_PI * std::sin(M_PI * t) ) * std::sin( M_PI * pt.x() );
        return ( M_PI*M_PI*std::cos(M_PI * t) - M_PI * std::sin(M_PI * t) ) * std::cos( M_PI * pt.x() );
        // return 0.;
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
        // return ( 2*M_PI*M_PI*std::cos(M_PI*t) - M_PI*std::sin(M_PI*t) ) * std::sin( M_PI * pt.x() )
        //    * std::sin( M_PI * pt.y() );
        return (2*M_PI*M_PI*std::cos(M_PI*t) - M_PI*std::sin(M_PI*t) ) * std::cos(M_PI*pt.x()) * std::cos(M_PI*pt.y());
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
        // return std::sin( M_PI * pt.x() ) * std::sin(M_PI * t);
        // return std::sin( M_PI * pt.x() ) * std::cos(M_PI * t);
        return std::cos( M_PI * pt.x() ) * std::cos(M_PI * t);
        // return pt.x();
        // return 1.0;
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
        // return std::cos(M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        return std::cos(M_PI*t) * std::cos(M_PI*pt.x()) * std::cos(M_PI*pt.y());
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
    test_info() : H1_Om(0.) , L2_Om(0.) , H1_B(0.) , L2_B(0.) , H1_z(0.), nb_dof(0) {}
    T H1_Om; // H1-error in Omega
    T L2_Om; // L2-error in Omega
    T H1_B; // H1-error in B
    T L2_B; // L2-error in B
    T H1_z; // H1-error for the dual variable
    size_t nb_dof; // number of degrees of freedom
    T h_max; // maximal cell diameter
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

    // list of boundaries that are not actual boundaries (has to be updated according to the considred mesh)
    const vector<int> not_bnd;

  public:
    typedef dynamic_matrix<T> matrix_type;
    typedef dynamic_vector<T> vector_type;

    SparseMatrix<T> LHS; //, MAT_RHS;
    vector_type     RHS;

    // BC_known : true if we know the Dirichlet values of the solution, false otherwise
    heat_UC_assembler(const Mesh& msh, hho_degree_info hdi, size_t t_degree, size_t t_steps, bool BC_known=false)
	: di(hdi), time_degree(t_degree), time_steps(t_steps), BC_known(BC_known), not_bnd({6,7,8,9,10,11})
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                                if( !msh.is_boundary(fc) )
                                    return false;
                                auto bnd_id = msh.boundary_id(fc);
                                for (auto it = not_bnd.begin(); it != not_bnd.end(); it++)
                                    if(bnd_id == *it)
                                        return false;
                                return true;
                                    };

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
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                                if( !msh.is_boundary(fc) )
                                    return false;
                                auto bnd_id = msh.boundary_id(fc);
                                for (auto it = not_bnd.begin(); it != not_bnd.end(); it++)
                                    if(bnd_id == *it)
                                        return false;
                                return true;
                            };

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
     * cross_assemble : to assemble terms on distinct cells
     */
    void
    cross_assemble(const Mesh&                     msh,
                   const typename Mesh::cell_type& cl1,
                   const typename Mesh::cell_type& cl2,
                   const size_t                    n_step1,
                   const size_t                    n_step2,
                   const matrix_type&              lhs,
                   const vector_type&              rhs)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                                if( !msh.is_boundary(fc) )
                                    return false;
                                auto bnd_id = msh.boundary_id(fc);
                                for (auto it = not_bnd.begin(); it != not_bnd.end(); it++)
                                    if(bnd_id == *it)
                                        return false;
                                return true;
                            };

        const auto fbs    = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs    = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fcs1    = faces(msh, cl1);
        const auto fcs_id1 = faces_id(msh, cl1);
        const auto fcs2    = faces(msh, cl2);
        const auto fcs_id2 = faces_id(msh, cl2);

        // we assume fcs1.size() == fcs2.size()
        assert( fcs1.size() == fcs2.size() );

        std::vector<assembly_index> asm_map1, asm_map2;
        size_t loc_size = 2 * ( fcs1.size() * fbs + cbs ) * (time_degree + 1);
        asm_map1.reserve(loc_size);
        asm_map2.reserve(loc_size);

        auto cell_offset1 = offset(msh, cl1);
        auto cell_offset2 = offset(msh, cl2);

        // first degrees of freedom are the cell components of the primal variable
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
        {
            asm_map1.push_back(assembly_index(cell_offset1 * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step1 + i, true));
            asm_map2.push_back(assembly_index(cell_offset2 * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step2 + i, true));
        }

        // then face components of the primal variable
        for (size_t face_i = 0; face_i < fcs1.size(); face_i++) // fcs1.size() == fcs2.size()
        {
            const auto face_offset1     = fcs_id1[face_i]; // offset(msh, fc);
            const auto face_offset2     = fcs_id2[face_i]; // offset(msh, fc);
            auto face_LHS_offset1 = num_cells * cbs * (time_degree+1) * time_steps;
            auto face_LHS_offset2 = num_cells * cbs * (time_degree+1) * time_steps;
            if(BC_known)
            {
                face_LHS_offset1 += compress_table.at(face_offset1) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step1; // compress table
                face_LHS_offset2 += compress_table.at(face_offset2) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step2; // compress table
            }
            else
            {
                face_LHS_offset1 += face_offset1 * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step1; // no Dirichlet BC so no compress table
                face_LHS_offset2 += face_offset2 * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step2; // no Dirichlet BC so no compress table
            }

            if(BC_known)
            {
                const auto fc1 = fcs1[face_i];
                const auto fc2 = fcs2[face_i];
                const bool dirichlet1 = is_dirichlet(fc1);
                const bool dirichlet2 = is_dirichlet(fc2);

                for (size_t i = 0; i < fbs*(time_degree+1); i++)
                {
                    asm_map1.push_back(assembly_index(face_LHS_offset1 + i, !dirichlet1));
                    asm_map2.push_back(assembly_index(face_LHS_offset2 + i, !dirichlet2));
                }
            }
            else
            {
                for (size_t i = 0; i < fbs*(time_degree+1); i++)
                {
                    asm_map1.push_back(assembly_index(face_LHS_offset1 + i, true)); // no test on Dirichlet (no BC)
                    asm_map2.push_back(assembly_index(face_LHS_offset2 + i, true)); // no test on Dirichlet (no BC)
                }
            }
        }

        // then cell components of the dual variable
        size_t num_sol_faces = num_all_faces;
        if( BC_known ) num_sol_faces = num_other_faces;

        size_t dual_offset = num_cells * cbs * (time_degree+1) * time_steps + num_sol_faces * fbs * (time_degree+1) * time_steps;
        for(size_t i = 0; i < cbs*(time_degree+1); i++)
        {
            asm_map1.push_back(assembly_index(dual_offset + cell_offset1 * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step1 + i, true));
            asm_map2.push_back(assembly_index(dual_offset + cell_offset2 * cbs * (time_degree+1) * time_steps + cbs * (time_degree+1) * n_step2 + i, true));
        }

        // then face components of the dual variable (with Dirichlet BC)
        for (size_t face_i = 0; face_i < fcs1.size(); face_i++) // fcs1.size() == fcs2.size()
        {
            const auto fc1              = fcs1[face_i];
            const auto fc2              = fcs2[face_i];
            const auto face_offset1     = fcs_id1[face_i]; // offset(msh, fc);
            const auto face_offset2     = fcs_id2[face_i]; // offset(msh, fc);
            const auto face_LHS_offset1 = dual_offset + num_cells * cbs * (time_degree+1) * time_steps + compress_table.at(face_offset1) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step1;
            const auto face_LHS_offset2 = dual_offset + num_cells * cbs * (time_degree+1) * time_steps + compress_table.at(face_offset2) * fbs * (time_degree+1) * time_steps + fbs * (time_degree+1) * n_step2;

            const bool dirichlet1 = is_dirichlet(fc1);
            const bool dirichlet2 = is_dirichlet(fc2);

            for (size_t i = 0; i < fbs*(time_degree+1); i++)
            {
                asm_map1.push_back(assembly_index(face_LHS_offset1 + i, !dirichlet1));
                asm_map2.push_back(assembly_index(face_LHS_offset2 + i, !dirichlet2));
            }
        }

        // asm_map1 corresponds to rows
        // asm_map2 corresponds to cols

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map1[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if (asm_map2[j].assemble())
                {
                    triplets.push_back(Triplet<T>(asm_map1[i], asm_map2[j], lhs(i, j)));
		}
            }

            RHS(asm_map1[i]) += rhs(i);
        }
    } // cross_assemble()

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

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                                if( !msh.is_boundary(fc) )
                                    return false;
                                auto bnd_id = msh.boundary_id(fc);
                                for (auto it = not_bnd.begin(); it != not_bnd.end(); it++)
                                    if(bnd_id == *it)
                                        return false;
                                return true;
                            };

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

// template<template<typename, size_t, typename> class Mesh,
//          typename T, typename Storage>
// void
// export_to_silo(const Mesh<T, 1, Storage>& msh,
//                const Matrix<T, Dynamic, 1>& data, const Matrix<T, Dynamic, 1>& varpi,
//                const Matrix<T, Dynamic, 1>& B, int cycle = -1)
// {
//     std::stringstream ss_data, ss_varpi, ss_B;
//     ss_varpi << "varpi.txt";
//     ss_B << "B.txt";
//     if(cycle == -1)
//     {
//         ss_data << "sol.txt";
//     }
//     else
//     {
//         ss_data << "out_data_" << cycle << ".txt";
//     }


//     std::ofstream data_file(ss_data.str(), std::ios::out | std::ios::trunc);
//     if(!data_file)
//         std::cerr << "error opening file !!" << std::endl;

//     std::ofstream varpi_file(ss_varpi.str(), std::ios::out | std::ios::trunc);
//     if(!varpi_file)
//         std::cerr << "error opening file !!" << std::endl;

//     std::ofstream B_file(ss_B.str(), std::ios::out | std::ios::trunc);
//     if(!B_file)
//         std::cerr << "error opening file !!" << std::endl;


//     int cell_i = 0;
//     for(auto& cl : msh)
//     {
//         auto x = barycenter(msh, cl).x();

//         data_file << x << "  " << data[cell_i] << std::endl;
//         varpi_file << x << "  " << varpi[cell_i] << std::endl;
//         B_file << x << "  " << B[cell_i] << std::endl;
//         cell_i++;
//     }

//     data_file.close();
//     varpi_file.close();
//     B_file.close();



//     //// For tests with the exact sol (modif N)
//     std::stringstream ss_ex_sol;
//     if(cycle == -1)
//     {
//         ss_ex_sol << "ex_sol.txt";
//     }
//     else
//     {
//         ss_ex_sol << "ex_sol_" << cycle << ".txt";
//     }


//     std::ofstream ex_sol_file(ss_ex_sol.str(), std::ios::out | std::ios::trunc);
//     if(!ex_sol_file)
//         std::cerr << "error opening file !!" << std::endl;

//     auto sol_fun = make_solution_function(msh);

//     int N = 32;
//     double t = (2./N) * (cycle+0.5);

//     cell_i = 0;
//     for(auto& cl : msh) {
//         auto x = barycenter(msh, cl).x();
//         auto sol = sol_fun(t , barycenter(msh, cl));

//         ex_sol_file << x << "  " << sol << std::endl;
//         cell_i++;
//     }

//     ex_sol_file.close();
// }

/////////////////////////

// template<template<typename, size_t, typename> class Mesh,
//          typename T, typename Storage>
// void
// export_to_silo(const Mesh<T, 2, Storage>& msh,
//                const Matrix<T, Dynamic, 1>& data, const Matrix<T, Dynamic, 1>& varpi,
//                const Matrix<T, Dynamic, 1>& B, int cycle = -1)
// {
//     disk::silo_database silo;

//     if (cycle == -1)
//         silo.create("UC_heat.silo");
//     else
//     {
//         std::stringstream ss;
//         ss << "out_" << cycle << ".silo";
//         silo.create(ss.str());
//     }

//     silo.add_mesh(msh, "mesh");

//     silo.add_variable("mesh", "sol", data, disk::zonal_variable_t );
//     silo.add_variable("mesh", "varpi", varpi, disk::zonal_variable_t );
//     silo.add_variable("mesh", "B", B, disk::zonal_variable_t );
//     silo.close();
// }

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

/****************************************************************/
/**********      Finite trace penalizations       ***************/
// compute the q0 term for penalization at initial time
// this has to be applied to basis functions at initial time (!)
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_space_q0(const Mesh& msh, const typename Mesh::cell_type& cl1,
              const typename Mesh::cell_type& cl2,
        Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> mass_matrix,
        finite_trace_init<Mesh> finite_space,
        const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto cb1 = make_scalar_monomial_basis(msh, cl1, celdeg);
    const auto cb2 = make_scalar_monomial_basis(msh, cl2, celdeg);

    size_t dim_finite_space = mass_matrix.rows();
    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);

    // compute the L2-projection on the finite dim space
    // F1 is the rhs of the projection in cell 1
    matrix_type F1 = matrix_type::Zero(dim_finite_space, cbs);
    const auto qps1 = integrate(msh, cl1, 2*celdeg);
    for (auto& qp : qps1)
    {
        const auto c_phi = cb1.eval_functions(qp.point());
        const auto fs_phi = finite_space.eval_functions(qp.point());

        F1 += qp.weight() * fs_phi * c_phi.transpose();
    }

    matrix_type F2 = matrix_type::Zero(dim_finite_space, cbs);
    const auto qps2 = integrate(msh, cl2, 2*celdeg);
    for (auto& qp : qps2)
    {
        const auto c_phi = cb2.eval_functions(qp.point());
        const auto fs_phi = finite_space.eval_functions(qp.point());

        F2 += qp.weight() * fs_phi * c_phi.transpose();
    }

    // P1 is the coefficients vector of the proj 1 in the finite space
    matrix_type P1 = matrix_type::Zero(dim_finite_space, cbs);
    LLT< Matrix<T, Dynamic, Dynamic> > mat_llt;
    mat_llt.compute(mass_matrix);
    P1 = mat_llt.solve(F1);

    // compute the local cell mass matrix
    auto mass_cell1   = make_mass_matrix(msh, cl1, cb1);

    if( cl1 == cl2 )
        return mass_cell1 - P1.transpose()*F2;

    return - P1.transpose()*F2;
}


// compute the q_{\partial} term for penalization on the boundary
// we assume that both cells cl1 and cl2 are on the domain boundary
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_q_bound(const Mesh& msh, const typename Mesh::cell_type& cl1,
             const typename Mesh::cell_type& cl2,
             const disk::generic_mesh<typename Mesh::coordinate_type, 1>& time_msh,
             const typename disk::generic_mesh<typename Mesh::coordinate_type, 1>::cell_type& time_cell1,
             const typename disk::generic_mesh<typename Mesh::coordinate_type, 1>::cell_type& time_cell2,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> finite_mass_matrix,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> finite_t_stiff_matrix,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> finite_tang_stiff_matrix,
             finite_trace_bound<Mesh> finite_space,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> time_mass,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> time_stiffness,
             const hho_degree_info& di, typename Mesh::coordinate_type dt,
             typename Mesh::coordinate_type h_max)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    size_t dim_finite_space = finite_mass_matrix.rows();
    const auto celdeg = di.cell_degree();
    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto cb1 = make_scalar_monomial_basis(msh, cl1, celdeg);
    const auto cb2 = make_scalar_monomial_basis(msh, cl2, celdeg);
    const size_t time_degree = time_mass.rows()-1;

    const auto fcs1 = faces(msh, cl1);
    const auto fcs2 = faces(msh, cl2);
    const auto num_faces = fcs1.size();
    // we assume fcs1.size() == fcs2.size()

    auto t_cb1 = make_scalar_monomial_basis(time_msh, time_cell1, time_degree);
    auto t_cb2 = make_scalar_monomial_basis(time_msh, time_cell2, time_degree);

    const auto qps_t1 = integrate(time_msh, time_cell1, 2*time_degree);
    const auto qps_t2 = integrate(time_msh, time_cell2, 2*time_degree);

    // compute the L2-projection on the finite dim space
    // F1 is the rhs of the projection for cell 1
    matrix_type F1 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));

    for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
    {
        auto fc1 = fcs1[face_i];
        if( !finite_space.is_dirichlet(msh, fc1) )
            continue;

        // compute rhs proj contrib
        const auto qps_f1 = integrate(msh, fc1, 2*celdeg);

        for (auto& qpf : qps_f1)
        {
            const auto cf_phi = cb1.eval_functions(qpf.point());
            for (auto& qpt : qps_t1)
            {
                const auto ct_phi = t_cb1.eval_functions(qpt.point());
                const auto fs_phi = finite_space.eval_functions(qpt.point().x(), qpf.point());

                for(size_t l1 = 0; l1 <= time_degree; l1++)
                    F1.block(0, l1*cbs, dim_finite_space, cbs) += qpt.weight() * qpf.weight() * ct_phi[l1] * fs_phi * cf_phi.transpose();
            }
        }
    }

    // F2 is the rhs of the projection for cell 2
    matrix_type F2 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));

    for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
    {
        auto fc2 = fcs2[face_i];
        if( !finite_space.is_dirichlet(msh, fc2) )
            continue;

        // compute rhs proj contrib
        const auto qps_f2 = integrate(msh, fc2, 2*celdeg);

        for (auto& qpf : qps_f2)
        {
            const auto cf_phi = cb2.eval_functions(qpf.point());
            for (auto& qpt : qps_t2)
            {
                const auto ct_phi = t_cb2.eval_functions(qpt.point());
                const auto fs_phi = finite_space.eval_functions(qpt.point().x(), qpf.point());

                for(size_t l1 = 0; l1 <= time_degree; l1++)
                    F2.block(0, l1*cbs, dim_finite_space, cbs) += qpt.weight() * qpf.weight() * ct_phi[l1] * fs_phi * cf_phi.transpose();
            }
        }
    }

    // P1, P2 are the coefficients vectors of the proj for cells 1 and 2 in the finite space
    matrix_type P1 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));
    matrix_type P2 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));
    LLT< Matrix<T, Dynamic, Dynamic> > mat_llt;
    mat_llt.compute(finite_mass_matrix);
    P1 = mat_llt.solve(F1);
    P2 = mat_llt.solve(F2);

    /*** L2_scal : (v , w) - (P_\partial v , w) ***/
    matrix_type L2_scal = matrix_type::Zero(cbs*(time_degree+1), cbs*(time_degree+1));

    // - (P_\partial v , w)
    L2_scal = - P1.transpose()*F2;

    // (v , w) : encoded at the end for efficiency reasons

    /*** der_t_scal : (\partial_t v , \partial_t w) - (\partial_t P_{\partial} v , \partial_t w)
         - (\partial_t v , \partial_t P_{\partial} w)
         + (\partial_t P_{\partial} v , \partial_t P_{\partial} w) ***/
    matrix_type der_t_scal = matrix_type::Zero(cbs*(time_degree+1), cbs*(time_degree+1));

    // + (\partial_t P_\partial v , \partial_t P_\partial w)
    der_t_scal += P1.transpose() * finite_t_stiff_matrix * P2;

    // (\partial_t \beta_i , \partial_t \phi1_j)
    matrix_type der_t_proj1 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));

    for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
    {
        auto fc1 = fcs1[face_i];
        if( !finite_space.is_dirichlet(msh, fc1) )
            continue;

        // compute rhs proj contrib
        const auto qps_f1 = integrate(msh, fc1, 2*celdeg);

        for (auto& qpf : qps_f1)
        {
            const auto cf_phi = cb1.eval_functions(qpf.point());
            for (auto& qpt : qps_t1)
            {
                const auto ct_dphi = t_cb1.eval_gradients(qpt.point());
                const auto fs_dphi = finite_space.eval_time_ders(qpt.point().x(), qpf.point());

                for(size_t l1 = 0; l1 <= time_degree; l1++)
                    der_t_proj1.block(0, l1*cbs, dim_finite_space, cbs) += qpt.weight() * qpf.weight() * ct_dphi[l1] * fs_dphi * cf_phi.transpose();
            }
        }
    }

    // (\partial_t \beta_i , \partial_t \phi2_j)
    matrix_type der_t_proj2 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));

    for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
    {
        auto fc2 = fcs2[face_i];
        if( !finite_space.is_dirichlet(msh, fc2) )
            continue;

        // compute rhs proj contrib
        const auto qps_f2 = integrate(msh, fc2, 2*celdeg);

        for (auto& qpf : qps_f2)
        {
            const auto cf_phi = cb2.eval_functions(qpf.point());
            for (auto& qpt : qps_t2)
            {
                const auto ct_dphi = t_cb2.eval_gradients(qpt.point());
                const auto fs_dphi = finite_space.eval_time_ders(qpt.point().x(), qpf.point());

                for(size_t l1 = 0; l1 <= time_degree; l1++)
                    der_t_proj2.block(0, l1*cbs, dim_finite_space, cbs) += qpt.weight() * qpf.weight() * ct_dphi[l1] * fs_dphi * cf_phi.transpose();
            }
        }
    }

    // - (\partial_t P_\partial v , \partial_t w)
    der_t_scal -= P1.transpose() * der_t_proj2;

    // - (\partial_t v , \partial_t P_\partial w)
    der_t_scal -= der_t_proj1.transpose() * P2;

    // (\partial_t v , \partial_t w) : encoded at the end for efficiency reasons


    /*** tang_scal : (\nabla_\partial v , \nabla_\partial w)
         - (\nabla_\partial P_{\partial} v , \nabla_\partial w)
         - (\nabla_\partial v , \nabla_\partial P_{\partial} w)
         + (\nabla_\partial P_{\partial} v , \nabla_\partial P_{\partial} w) ***/
    matrix_type tang_scal = matrix_type::Zero(cbs*(time_degree+1), cbs*(time_degree+1));

    // + (\nabla_\partial P_{\partial} v , \nabla_\partial P_{\partial} w)
    tang_scal += P1.transpose() * finite_tang_stiff_matrix * P2;

    // (\nabla_\partial \beta_i , \nabla_\partial \phi1_j)
    matrix_type tang_proj1 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));

    for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
    {
        auto fc1 = fcs1[face_i];
        if( !finite_space.is_dirichlet(msh, fc1) )
            continue;

        // compute rhs proj contrib
        const auto qps_f1 = integrate(msh, fc1, 2*celdeg);

        const auto n1  = normal(msh, cl1, fc1);

        for (auto& qpf : qps_f1)
        {
            const auto cf_dphi = cb1.eval_gradients(qpf.point());
            const auto cf_dphi_n = cf_dphi * n1;
            for (auto& qpt : qps_t1)
            {
                const auto ct_phi = t_cb1.eval_functions(qpt.point());
                const auto fs_dphi = finite_space.eval_gradients(qpt.point().x(), qpf.point());
                const auto fs_dphi_n = fs_dphi * n1;

                for(size_t l1 = 0; l1 <= time_degree; l1++)
                {
                    tang_proj1.block(0, l1*cbs, dim_finite_space, cbs) += qpt.weight() * qpf.weight() * ct_phi[l1] * fs_dphi * cf_dphi.transpose(); // full gradient
                    tang_proj1.block(0, l1*cbs, dim_finite_space, cbs) -= qpt.weight() * qpf.weight() * ct_phi[l1] * fs_dphi_n * cf_dphi_n.transpose(); // remove normal component
                }
            }
        }
    }

    // (\nabla_\partial \beta_i , \nabla_\partial \phi2_j)
    matrix_type tang_proj2 = matrix_type::Zero(dim_finite_space, cbs*(time_degree+1));

    for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
    {
        auto fc2 = fcs2[face_i];
        if( !finite_space.is_dirichlet(msh, fc2) )
            continue;

        // compute rhs proj contrib
        const auto qps_f2 = integrate(msh, fc2, 2*celdeg);

        const auto n2  = normal(msh, cl2, fc2);

        for (auto& qpf : qps_f2)
        {
            const auto cf_dphi = cb2.eval_gradients(qpf.point());
            const auto cf_dphi_n = cf_dphi * n2;
            for (auto& qpt : qps_t2)
            {
                const auto ct_phi = t_cb2.eval_functions(qpt.point());
                const auto fs_dphi = finite_space.eval_gradients(qpt.point().x(), qpf.point());
                const auto fs_dphi_n = fs_dphi * n2;

                for(size_t l1 = 0; l1 <= time_degree; l1++)
                {
                    tang_proj2.block(0, l1*cbs, dim_finite_space, cbs) += qpt.weight() * qpf.weight() * ct_phi[l1] * fs_dphi * cf_dphi.transpose(); // full gradient
                    tang_proj2.block(0, l1*cbs, dim_finite_space, cbs) -= qpt.weight() * qpf.weight() * ct_phi[l1] * fs_dphi_n * cf_dphi_n.transpose(); // remove normal component
                }
            }
        }
    }

    // - (\nabla_\partial P_\partial v , \nabla_\partial w)
    tang_scal -= P1.transpose() * tang_proj2;

    // - (\nabla_\partial v , \nabla_\partial P_\partial w)
    tang_scal -= tang_proj1.transpose() * P2;

    // (\nabla_\partial v , \nabla_\partial w) : encoded at the end for efficiency reasons

    if( cl1 == cl2 and time_cell1 == time_cell2 )
    {
        matrix_type trace_f = matrix_type::Zero(cbs,cbs);
        matrix_type tang_f = matrix_type::Zero(cbs,cbs); // tangential stiffness

        for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
        {
            auto fc = fcs1[face_i];
            if( !finite_space.is_dirichlet(msh, fc) )
                continue;

            const auto n  = normal(msh, cl1, fc);

            const auto qps_f = integrate(msh, fc, 2*celdeg);
            for (auto& qpf : qps_f)
            {
                auto cf_phi = cb1.eval_functions(qpf.point());
                auto cf_dphi = cb1.eval_gradients(qpf.point());
                auto cf_dphi_n = cf_dphi * n;

                trace_f += qpf.weight() * cf_phi * cf_phi.transpose();
                tang_f += qpf.weight() * cf_dphi * cf_dphi.transpose(); // full gradients
                tang_f -= qpf.weight() * cf_dphi_n * cf_dphi_n.transpose(); // remove normal contributions
            }
        }

        for(size_t l1 = 0; l1 <= time_degree; l1++)
        {
            for(size_t l2 = 0; l2 <= time_degree; l2++)
            {
                // + (v , w)
                L2_scal.block(l1*cbs,l2*cbs,cbs,cbs) += time_mass(l1,l2) * trace_f;

                // + (\partial_t v , \partial_t w)
                der_t_scal.block(l1*cbs,l2*cbs,cbs,cbs) += time_stiffness(l1,l2) * trace_f;

                // + (\nabla_\partial v , \nabla_\partial w)
                tang_scal.block(l1*cbs,l2*cbs,cbs,cbs) += time_mass(l1,l2) * tang_f;
            }
        }
    }

    return (1./h_max) * L2_scal + (dt*dt/h_max) * der_t_scal + h_max * tang_scal;
}

///////////////////////////////////////////////

template<typename Mesh, typename NR>
test_info<typename Mesh::coordinate_type>
UC_heat_solver(const Mesh& msh, size_t degree, size_t time_steps, size_t time_degree, NR noise_fct)
{
    typedef typename Mesh::coordinate_type  scalar_type;
    typedef typename Mesh::point_type       point_type;
    using T = scalar_type;

    hho_degree_info hdi(degree, degree, degree+1);

    auto num_cells = msh.cells_size();
    auto nb_tot_faces = msh.faces_size();

    auto assembler = make_heat_UC_assembler(msh, hdi, time_degree, time_steps, false);

    const bool EXPORT_NOISE = false;

    auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
    auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension-1);

    timecounter tc;

    scalar_type final_time = 2.;

    // time mesh
    cout << "time_steps = " << time_steps << endl;
    scalar_type dt = final_time/time_steps;
    cout << "dt = " << dt << endl;
    disk::generic_mesh<T, 1>  time_mesh;
    disk::uniform_mesh_loader<T, 1> time_loader(0, final_time, time_steps);
    time_loader.populate_mesh(time_mesh);

    // finite trace elements
    auto finite_trace_init = make_finite_trace_init(msh, 1);
    auto mass_init = finite_trace_init.make_mass_matrix(msh);
    auto finite_trace_bound = make_finite_trace_bound(msh, 2);
    auto mass_bound = finite_trace_bound.make_mass_matrix(msh, time_mesh);
    auto time_stiff_bound = finite_trace_bound.make_time_stiffness_matrix(msh, time_mesh);
    auto tang_stiff_bound = finite_trace_bound.make_tang_stiffness_matrix(msh, time_mesh);

    auto rhs_fun = make_rhs_function(msh);
    auto sol_fun = make_solution_function(msh);

    auto varpi_fun = make_varpi_function(msh);
    auto B_fun = make_B_function(msh);

    auto time_cell = *time_mesh.cells_begin();
    auto next_time_cell = *(++time_mesh.cells_begin());

    auto time_cb = make_scalar_monomial_basis(time_mesh, time_cell, time_degree);
    auto time_cb_next = make_scalar_monomial_basis(time_mesh, next_time_cell, time_degree);
    auto time_mass = make_mass_matrix(time_mesh, time_cell, time_cb);
    auto time_stiff   = make_stiffness_matrix(time_mesh, time_cell, time_cb);

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

    // T Tikhonov_coeff = hk + dtl * sqrt(dt);
    T Tikhonov_coeff = 0.;

    tc.tic();

    // export the noise values for visualization purposes
    auto LNF = make_line_noise_file(msh);
    std::ofstream noise_file;
    if(EXPORT_NOISE)
    {
        std::stringstream ss_noise;
        ss_noise << "noise_k" << degree << "_l" << time_degree << "_h" << msh.cells_size() << "_N" << time_steps;
        noise_file.open (ss_noise.str(), std::ios::in | std::ios::trunc);
        if (!noise_file.is_open())
            throw std::logic_error("file not open");
        noise_file << init_noise_file<Mesh::dimension>() << std::endl;
    }

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

        /*** Matrix lhs for the terms contained in a cell and a time step ***/
        Matrix<scalar_type, Dynamic, Dynamic> lhs = Matrix<scalar_type, Dynamic, Dynamic>::Zero(loc_size, loc_size);

        /* Primal - Primal */
        Matrix<T, Dynamic, Dynamic> PP = stab;
        // PP.block(0,0,cbs,cbs) += gamma * Tikhonov_coeff * Tikhonov_coeff * mass.block(0,0,cbs,cbs);
        // add data assimilation term
        if( varpi_fun(barycenter(msh,cl)) > 0.5 )
        {
            // cout << "enter cell with data" << "..." << endl;
            PP.block(0,0,cbs,cbs) += mass;
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
                                  fbs, fbs) = PP.block(cbs + face_i*fbs, cbs + face_j*fbs, fbs, fbs) * time_mass(l1,l2);


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
        // sigma.block(0,0,cbs,cbs) += mass;

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

        /* coupling coming from the domain boundary terms : d_{\partial} */
        for(size_t face_i = 0; face_i < num_faces; face_i++) // loop on boundary faces
        {
            const auto fc = fcs[face_i];
            if( !finite_trace_bound.is_dirichlet(msh, fc) )
                continue;

            /* coupling coming from the time jump penalization */
            Matrix<scalar_type, Dynamic, Dynamic> trace_fc = Matrix<scalar_type, Dynamic, Dynamic>::Zero(cbs,cbs);
            const auto qps_f = integrate(msh, fc, 2*hdi.cell_degree());
            for (auto& qpf : qps_f)
            {
                auto cf_phi = cb.eval_functions(qpf.point());

                trace_fc += qpf.weight() * cf_phi * cf_phi.transpose();
            }
            T jump_coeff2 = 1.; // 1./h_max;

            // (v_T(t_{n-1}^+) , w_T(t_{n-1}^+))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(ts_c + l1*cbs, ts_c + l2*cbs, cbs, cbs)
                        += jump_coeff2 * trace_fc * time_loc(l1,l2);

            // (v_T(t_{n-1}^-) , w_T(t_{n-1}^-))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(l1*cbs, l2*cbs, cbs, cbs)
                        += jump_coeff2 * trace_fc * time_loc_bis(l1,l2);

            // - (v_T(t_{n-1}^-) , w_T(t_{n-1}^+))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(ts_c + l1*cbs, l2*cbs, cbs, cbs)
                        -= jump_coeff2 * trace_fc * time_loc_cross(l2,l1);

            // - (v_T(t_{n-1}^+) , w_T(t_{n-1}^-))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(l1*cbs, ts_c + l2*cbs, cbs, cbs)
                        -= jump_coeff2 * trace_fc * time_loc_cross(l1,l2);

            /* coupling coming from the gradient time jump penalization */
            Matrix<scalar_type, Dynamic, Dynamic> stiff_fc = Matrix<scalar_type, Dynamic, Dynamic>::Zero(cbs,cbs);
            for (auto& qpf : qps_f)
            {
                auto cf_dphi = cb.eval_gradients(qpf.point());

                stiff_fc += qpf.weight() * cf_dphi * cf_dphi.transpose();
            }
            T jump_coeff3 = 0.; //h_max;

            // (\GRAD v_T(t_{n-1}^+) , \GRAD w_T(t_{n-1}^+))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(ts_c + l1*cbs, ts_c + l2*cbs, cbs, cbs)
                        += jump_coeff3 * stiff_fc * time_loc(l1,l2);

            // (\GRAD v_T(t_{n-1}^-) , \GRAD w_T(t_{n-1}^-))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(l1*cbs, l2*cbs, cbs, cbs)
                        += jump_coeff3 * stiff_fc * time_loc_bis(l1,l2);

            // - (\GRAD v_T(t_{n-1}^-) , \GRAD w_T(t_{n-1}^+))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(ts_c + l1*cbs, l2*cbs, cbs, cbs)
                        -= jump_coeff3 * stiff_fc * time_loc_cross(l2,l1);

            // - (\GRAD v_T(t_{n-1}^+) , \GRAD w_T(t_{n-1}^-))_{\partial Omega}
            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    coupling.block(l1*cbs, ts_c + l2*cbs, cbs, cbs)
                        -= jump_coeff3 * stiff_fc * time_loc_cross(l1,l2);

        }

        // at this point all the coupling terms have been implemented

        /*** loop on the time steps ***/
        for(size_t step_i = 0; step_i < time_steps; step_i++) {
            Matrix<scalar_type, Dynamic, 1> rhs = Matrix<scalar_type, Dynamic, 1>::Zero(lhs.cols());

            auto t_cell = *(time_mesh.cells_begin()+step_i);
            const auto qpst = integrate(time_mesh, t_cell , 2*time_degree);
            const auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            auto t_cb = make_scalar_monomial_basis(time_mesh, t_cell, time_degree);

            // T noise_size = 0.;
            // T noise_size = 0.001;
            // T noise_size = 1.e-5;

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

                        T noise = noise_fct( time_point , qp.point() );
                        // T noise = noise_size * ((std::rand() % 200)-100) * 0.01;
                        // T noise = 0.0;
                        T noised_data = sol_fun( time_point , qp.point() ) + noise;

                        // export noise level
                        if(EXPORT_NOISE)
                            noise_file << LNF(time_point, qp.point(), noise) << std::endl;

                        for(size_t l1 = 0; l1 <= time_degree; l1++)
                        {
                            data_rhs.block(l1*cbs, 0, cbs, 1) += qp.weight() * qpt.weight() * t_phi[l1] * noised_data * x_phi;
                        }
                    }
                }
                
                rhs.block(0, 0, cbs*(time_degree+1), 1) = data_rhs;
            }

            /* finite trace terms */
            // auto lhs2 = lhs;
            // boundary finite trace (uses only cell unknowns)
            // auto q_bound = make_q_bound(msh, cl, time_mesh, t_cell, mass_bound, finite_trace_bound, time_mass, time_stiff, hdi);
            // cout << " q_bound = " << q_bound << endl << endl;
            // // // cout << "q_bound.rows() = " << q_bound.rows() << "   cbs*(time_degree+1) = " << cbs*(time_degree+1) << endl;
            // cout << "lhs2 before = " << lhs2 << endl << endl;
            // lhs2.block(0, 0, cbs*(time_degree+1), cbs*(time_degree+1)) += q_bound;
            // cout << "lhs2 after = " << lhs2 << endl << endl;

            // assembler.assemble(msh, cl, step_i, lhs2, rhs);
            assembler.assemble(msh, cl, step_i, lhs, rhs);
        }

        /*** add the time coupling terms triplets ***/
        for(int step_i = 1; step_i < time_steps; step_i++) {
            assembler.add_time_coupling(msh, cl, step_i, coupling);
        }

    }

    /*** add the finite trace penalization terms ***/
    // term q_0
    for (auto& cl1 : msh)
    {
        for (auto& cl2 : msh)
        {
            auto fcs1    = faces(msh, cl1);
            auto fcs2    = faces(msh, cl2);
            assert(fcs1.size() == fcs2.size()); // assumption to ease the code
            auto num_faces = fcs1.size();

            size_t loc_size = 2 * (cbs + num_faces * fbs) * (time_degree + 1);
            Matrix<scalar_type, Dynamic, Dynamic> lhs = Matrix<scalar_type, Dynamic, Dynamic>::Zero(loc_size, loc_size);
            Matrix<scalar_type, Dynamic, 1> rhs = Matrix<scalar_type, Dynamic, 1>::Zero(lhs.cols());
            auto mat_space_q0 = make_space_q0(msh, cl1, cl2, mass_init, finite_trace_init, hdi);

            for(size_t l1 = 0; l1 <= time_degree; l1++)
                for(size_t l2 = 0; l2 <= time_degree; l2++)
                    lhs.block(l1*cbs, l2*cbs, cbs, cbs) += mat_space_q0 * time_loc(l1,l2);

            assembler.cross_assemble(msh, cl1, cl2, 0, 0, lhs, rhs);
        }
    }
    // term q_{\partial}
    for (auto& cl1 : msh)
    {
        if( !finite_trace_bound.is_dirichlet(msh, cl1) ) // boundary faces only
            continue;
        auto fcs1    = faces(msh, cl1);
        size_t loc_size = 2 * (cbs + fcs1.size() * fbs) * (time_degree + 1);

        for (auto& cl2 : msh)
        {
            if( !finite_trace_bound.is_dirichlet(msh, cl2) ) // boundary faces only
                continue;

            for(size_t step_i = 0; step_i < time_steps; step_i++) {
                auto t_cell1 = *(time_mesh.cells_begin()+step_i);

                for(size_t step_j = 0; step_j < time_steps; step_j++) {
                    auto t_cell2 = *(time_mesh.cells_begin()+step_j);

                    Matrix<scalar_type, Dynamic, Dynamic> lhs = Matrix<scalar_type, Dynamic, Dynamic>::Zero(loc_size, loc_size);
                    Matrix<scalar_type, Dynamic, 1> rhs = Matrix<scalar_type, Dynamic, 1>::Zero(lhs.cols());
                    auto q_bound = make_q_bound(msh, cl1, cl2, time_mesh, t_cell1, t_cell2, mass_bound, time_stiff_bound, tang_stiff_bound, finite_trace_bound, time_mass, time_stiff, hdi, dt, h_max);

                    lhs.block(0, 0, cbs*(time_degree+1), cbs*(time_degree+1)) = q_bound;

                    assembler.cross_assemble(msh, cl1, cl2, step_i, step_j, lhs, rhs);
                }
            }
        }
    }

    cout << "end assembly loop" << endl;

    // close the noise file
    if(EXPORT_NOISE)
        noise_file.close();

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
    // tc.tic();
    Matrix<scalar_type, Dynamic, 1> u;

    if(true) {
        tc.tic();
        std::cout << "running Pardiso" << std::endl;
        disk::solvers::pardiso_params<scalar_type> pparams;
        mkl_pardiso(pparams, LHS, RHS, u);
        tc.toc();
        std::cout << " Pardiso Solving time: " << tc << std::endl;
    }
    if(false)
    {
        tc.tic();
        std::cout << "running MINRES" << std::endl;
        MINRES<SparseMatrix<scalar_type>> mr;
        mr.compute(LHS);
        u = mr.solve(RHS);
        std::cout << "#iterations:     " << mr.iterations() << std::endl;
        std::cout << "estimated error: " << mr.error()      << std::endl;

        tc.toc();
        std::cout << " MINRES Solving time: " << tc << std::endl;
    }
    if(false)
    {
        // JacobiSVD<SparseMatrix<scalar_type>> svd(LHS);
        // scalar_type cond = svd.singularValues()(0)
        //     / svd.singularValues()(svd.singularValues().size()-1);
        // std::cout << "conditionning = " << cond << std::endl;

        // std::cout << "test" << std::endl;

        // const auto ev = SelfAdjointEigenSolver<SparseMatrix<scalar_type>>(LHS).eigenvalues();
        // scalar_type cond = ev(ev.size() - 1) / ev(0);
        // std::cout << "conditionning = " << cond << std::endl;

        tc.tic();

        std::cout << "running GMRES" << std::endl;
        GMRES<SparseMatrix<scalar_type> > solver(LHS);
        u = solver.solve(RHS);
        std::cout << "#iterations:     " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;

        tc.toc();
        std::cout << " GMRES Solving time: " << tc << std::endl;
    }
    // if(false)
    // {
    //     tc.tic();
    //     std::cout << "running MUMPS" << std::endl;

    //     mumps_solver<scalar_type> mumps;
    //     u = mumps.solve(LHS, RHS);

    //     tc.toc();
    //     std::cout << " MUMPS Solving time: " << tc << std::endl;
    // }

    // tc.toc();
    
    // std::cout << " Solving time: " << tc << std::endl;

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

        // Matrix<scalar_type, Dynamic, 1> sol_silo = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());
        // Matrix<scalar_type, Dynamic, 1> data_varpi = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());
        // Matrix<scalar_type, Dynamic, 1> data_B = Matrix<scalar_type, Dynamic, 1>::Zero(msh.cells_size());

        /*
        if(step_i % freq_exp == 0)
        {
            // be careful : this exports a very approximated solution
            // TODO : modify this export ...
            // for (size_t i = 0; i < msh.cells_size(); i++)
            //     sol_silo(i) = u(i*cbs*(time_degree+1)*time_steps + cbs*(time_degree+1) * step_i);

            int cell_i = 0;
            for(auto& cl : msh)
            {
                const auto& bar = barycenter(msh,cl);
                // data_varpi(cell_i) = varpi_fun( bar );
                // data_B(cell_i) = B_fun( bar );
                cell_i++;
            }
            // export_to_silo( msh, sol_silo, data_varpi, data_B, step_i );
        }
        */

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
    TI.nb_dof = LHS.rows();
    TI.H1_Om = std::sqrt(L2H1_error);
    TI.L2_Om = std::sqrt(L2L2_error);
    TI.H1_B = std::sqrt(L2H1_B_error);
    TI.L2_B = std::sqrt(L2L2_B_error);
    TI.H1_z = std::sqrt(L2H1_z);
    TI.h_max = h_max;


    /** export solution with gnuplot for 1d meshes **/
    if(Mesh::dimension > 1)
        return TI;

    // open file
    std::ofstream gnu_file("gnu_solution.txt", std::ios::out | std::ios::trunc);
    if(!gnu_file)
        std::cerr << "error opening file !!" << std::endl;

    for(int step_i = 0; step_i < time_steps; step_i++) { // time loop
        T time_point = (step_i+0.5)*dt;
        size_t cell_i = 0;
        for (auto& cl : msh) { // loop on the mesh cells
            Matrix<scalar_type, Dynamic, 1> loc_sol
                = u.block(cell_i*cbs*(time_degree+1)*time_steps + cbs * (time_degree+1) * step_i, 0, cbs, 1);
            auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
            const auto qps = integrate(msh, cl, 1); // degree 1
            for (auto& qp : qps)
            {
                auto x_phi   = cb.eval_functions( qp.point() );
                T sol = x_phi.transpose() * loc_sol;
                gnu_file << qp.point().x() << " " << time_point << " " << sol << endl;
            }

            cell_i++;
        }
    }

    return TI;
}



template<typename T>
void
tests_auto_1d()
{
    typedef disk::generic_mesh<T, 1>  mesh_type;


    // T noise_size = 1.e-3;
    // T noise_size = 1.e-5;
    T noise_size = 0.;
    size_t nb_noise_subdiv = 10;
    mesh_type test_mesh;

    auto noise_fct = make_noise_representation(test_mesh, nb_noise_subdiv, noise_size);

    /*********************  REFINEMENT IN SPACE  **************************/
    if(true)
    {
        size_t nb_meshes = 5;


        // list of export files
        std::vector<std::string> files;
        files.push_back("./test_space_k0.txt");
        files.push_back("./test_space_k1.txt");
        files.push_back("./test_space_k2.txt");
        files.push_back("./test_space_k3.txt");

        // we test space degrees from 1 to 3
        for(int s_degree=1; s_degree <= 3; s_degree++)
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
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1\tdof\th" << std::endl;

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
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree, noise_fct);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << "\t" << TI.nb_dof << "\t" << TI.h_max
                     << std::endl;
            }

            // close the file
            file.close();
        }

    }
    /*********************  REFINEMENT IN TIME  **************************/
    if(true)
    {
        size_t nb_meshes = 4;

        std::vector<std::string> files;
        files.push_back("./test_time_k0.txt");
        files.push_back("./test_time_k1.txt");
        files.push_back("./test_time_k2.txt");
        files.push_back("./test_time_k3.txt");

        // we test time degrees from 0 to 3
        for(int t_degree=0; t_degree < 4; t_degree++)
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
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1\tdof\tN" << std::endl;

            // we test all the meshes in the list
            size_t N = 5;
            for(size_t i=0; i < nb_meshes; i++)
            {
                N *= 2;
                // load the mesh : 1d mesh only ATM
                mesh_type msh;
                disk::uniform_mesh_loader<T, 1> loader(0, 1, M);
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree, noise_fct);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << "\t" << TI.nb_dof << "\t" << N
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

    // list of mesh files
    std::vector<std::string> meshes;
    // meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri01.mesh2d");
    // meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri02.mesh2d");
    // meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri03.mesh2d");
    // meshes.push_back("./../../../diskpp/meshes/2D_triangles/netgen/tri04.mesh2d");

    meshes.push_back("gmsh_meshes/test2d_3bound_1.geo");
    meshes.push_back("gmsh_meshes/test2d_3bound_1_5.geo");
    meshes.push_back("gmsh_meshes/test2d_3bound_2.geo");
    meshes.push_back("gmsh_meshes/test2d_3bound_2_5.geo");
    meshes.push_back("gmsh_meshes/test2d_3bound_3.geo");
    meshes.push_back("gmsh_meshes/test2d_3bound_3_5.geo");
    meshes.push_back("gmsh_meshes/test2d_3bound_4.geo");

    // T noise_size = 1.e-3;
    // T noise_size = 1.e-5;
    T noise_size = 0.;
    size_t nb_noise_subdiv = 10;
    mesh_type test_mesh;

    auto noise_fct = make_noise_representation(test_mesh, nb_noise_subdiv, noise_size);


    /*********************  REFINEMENT IN SPACE  **************************/
    if(true)
    {
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
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1\tdof\th" << std::endl;

            // we test all the meshes in the list
            for(size_t i=0; i < nb_meshes; i++)
            {
                mesh_type msh;

                // disk::netgen_mesh_loader<T, 2> loader;
                disk::gmsh_geometry_loader< mesh_type > loader;

                if( !loader.read_mesh(meshes.at(i)) )
                    std::cout << "error loading mesh !" << std::endl;
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree, noise_fct);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << "\t" << TI.nb_dof << "\t" << TI.h_max
                     << std::endl;
            }

            // close the file
            file.close();
        }

    }
    /*********************  REFINEMENT IN TIME  **************************/
    if(true)
    {
        size_t nb_meshes = 5;
        size_t time_meshes[] = {5,10,15,20,30};

        std::vector<std::string> files;
        files.push_back("./test_time_k0.txt");
        files.push_back("./test_time_k1.txt");
        files.push_back("./test_time_k2.txt");
        files.push_back("./test_time_k3.txt");

        // we test time degree 0 and 1 only
        for(int t_degree=0; t_degree < 3; t_degree++)
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
            file << "N\tB_L2\tB_H1\tOmega_L2\tOmega_H1\tz_H1\tdof\tN" << std::endl;

            // we test all the meshes in the list
            for(size_t i=0; i < nb_meshes; i++)
            {
                mesh_type msh;
                size_t N = time_meshes[i];

                // disk::netgen_mesh_loader<T, 2> loader;
                disk::gmsh_geometry_loader< mesh_type > loader;


                if( !loader.read_mesh(meshes.at(5)) )
                    std::cout << "error loading mesh !" << std::endl;
                loader.populate_mesh(msh);

                // test this mesh
                auto TI = UC_heat_solver(msh, s_degree, N, t_degree, noise_fct);

                // write the results in the file
                file << i+1 << "\t" << TI.L2_B << "\t" << TI.H1_B << "\t"
                     << TI.L2_Om << "\t" << TI.H1_Om << "\t" << TI.H1_z
                     << "\t" << TI.nb_dof << "\t" << N
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
    // tests_auto_1d<double>();
    tests_auto_2d<double>();
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
