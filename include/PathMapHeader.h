/**
 *  \file IMP/bff/PathMapHeader.h
 *  \brief Header class for path search class PathMap
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2022 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_PATHMAPHEADER_H
#define IMPBFF_PATHMAPHEADER_H

#include <IMP/bff/bff_config.h>

#include <IMP/Particle.h>
#include <IMP/algebra/Vector3D.h>
#include <IMP/atom/Atom.h>
#include <IMP/atom/Hierarchy.h>
#include <IMP/atom/Selection.h>
#include <IMP/em/DensityHeader.h>

#include <IMP/bff/internal/json.h>
#include <IMP/bff/AV.h>

#include <algorithm>

IMPBFF_BEGIN_NAMESPACE


class PathMap;

class IMPEMEXPORT PathMapHeader {

friend class IMP::bff::PathMap;

private:

    double grid_spacing_;
    double max_path_length_;
    int neighbor_radius_;
    double obstacle_threshold_;

    IMP::em::DensityHeader density_header_;

protected:

    IMP::algebra::Vector3D path_origin_;

public:

    /*!
     *
     * @param max_path_length maximum length of path (defines also size of grid)
     * @param grid_spacing spacing between grid tiles
     * @param neighbor_radius defines size of box around tile where other
     * voxels are considered a neighbor
     * @param obstacle_threshold voxels with density larger than this
     * threshold value are considered an obstacle.
     */
    PathMapHeader(
            double max_path_length,
            double grid_spacing,
            int neighbor_radius = 2,
            double obstacle_threshold = std::numeric_limits<double>::epsilon()
    );

    //! Update the dimensions of the AV to be (nnx,nny,nnz)
    //! The origin of the map does not change. If not values
    //! are provided used linker length & radius to update.
    /**
        \param[in] nnx the new number of voxels on the X axis
        \param[in] nny the new number of voxels on the Y axis
        \param[in] nnz the new number of voxels on the Z axis
     */
    void update_map_dimensions(int nx=-1, int ny=-1, int nz=-1);

    void set_path_origin(IMP::algebra::Vector3D v){
        path_origin_ = v;
        double lr = get_grid_edge_length() / 2;
        density_header_.set_xorigin(v[0] - lr);
        density_header_.set_yorigin(v[1] - lr);
        density_header_.set_zorigin(v[2] - lr);
    }

    //! Returns position of the labeling site
    IMP::algebra::Vector3D get_path_origin() const {
        return path_origin_;
    }

    /// Maximum linker/path length from origin
    double get_max_path_length(){
        return max_path_length_;
    }

    double get_simulation_grid_resolution() {
        grid_spacing_ = (double) density_header_.get_spacing();
        return grid_spacing_;
    }

    void set_obstacle_threshold(
            double obstacle_threshold){
        obstacle_threshold_ = obstacle_threshold;
    }

    double get_obstacle_threshold() const{
        return obstacle_threshold_;
    }

    double set_neighbor_radius(
            double neighbor_radius){
        neighbor_radius_ = neighbor_radius;
    }

    int get_neighbor_radius() const{
        return neighbor_radius_;
    }

    int get_neighbor_box_size() const{
        return ceil(neighbor_radius_);
    }

    //! Returns a read-only pointer to the header of the map
    const IMP::em::DensityHeader *get_density_header() const {
        return &density_header_; }

    //! Returns a pointer to the header of the map in a writable version
    IMP::em::DensityHeader *get_density_header_writable() {
        return &density_header_; }

    //! Get origin on the PathMap (the corner of the grid)
    IMP::algebra::Vector3D get_origin() const {
        return {
                (double) density_header_.get_xorigin(),
                (double) density_header_.get_yorigin(),
                (double) density_header_.get_zorigin()
        };
    }

    double get_grid_edge_length();

    //! Set origin on the PathMap (the corner of the grid)
    void set_origin(float x, float y, float z){
        density_header_.set_xorigin(x);
        density_header_.set_yorigin(y);
        density_header_.set_zorigin(z);
    }


};

IMPBFF_END_NAMESPACE


#endif //IMPBFF_PATHMAPHEADER_H
