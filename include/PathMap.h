/**
 *  \file IMP/bff/PathMap.h
 *  \brief Class to search path on grids
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_PATHMAP_H
#define IMPBFF_PATHMAP_H

#include <IMP/bff/bff_config.h>

#include <stdlib.h>     /* malloc, free, rand */

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <vector>
#include <utility>  /* std::pair */
#include <Eigen/Dense>

#include <IMP/Object.h>
#include <IMP/Particle.h>
#include <IMP/core/XYZR.h>
#include <IMP/em/SampledDensityMap.h>

#include <IMP/em/MRCReaderWriter.h>
#include <IMP/em/XplorReaderWriter.h>
#include <IMP/em/EMReaderWriter.h>
#include <IMP/em/SpiderReaderWriter.h>

#include <IMP/bff/PathMapHeader.h>
#include <IMP/bff/PathMapTile.h>
#include <IMP/bff/PathMapTileEdge.h>

IMPBFF_BEGIN_NAMESPACE

class PathMapTile;


class IMPBFFEXPORT PathMap : public IMP::em::SampledDensityMap {

friend class PathMapTile;

private:

    // used in path search
    std::vector<bool>  visited;
    std::vector<bool>  edge_computed;
    std::vector<float> cost;


protected:

    PathMapHeader pathMapHeader_;    
    std::vector<PathMapTile> tiles;
    std::vector<int> offsets_;
    std::vector<PathMapTileEdge>& get_edges(int tile_idx);

public:

    void update_tiles(
        float obstacle_threshold=-1.0, 
        bool binarize=true, 
        float obstacle_penalty=TILE_PENALTY_DEFAULT,
        bool reset_tile_edges=true
    );

    void resize(unsigned int nvox);

    void set_data(double *input, int n_input, 
        float obstacle_threshold=-1, bool binarize=true, 
        float obstacle_penalty=TILE_PENALTY_DEFAULT);

    std::vector<int> get_neighbor_idx_offsets(double neighbor_radius = -1){
        if(neighbor_radius < 0){
            auto pmh = get_path_map_header();
            neighbor_radius = pmh->get_neighbor_radius();
        }
        const int nn = ceil(neighbor_radius);
        const double nr2 = neighbor_radius * neighbor_radius;

        const IMP::em::DensityHeader* header = get_header();
        int nx = header->get_nx();
        int ny = header->get_ny();
        int nx_ny = nx * ny;

        std::vector<int> offsets;
        for(int z = -nn; z < nn; z += 1) {
            double dz2 = z * z;
            int oz = z * nx_ny;
            for(int y = -nn; y < nn; y++) {
                int oy = y * nx;
                double dz2_dy2 = dz2 + y * y;
                for(int x = -nn; x < nn; x++) {
                    int ox = x;
                    int dz2_dy2_dx2 = dz2_dy2 + x * x;
                    if(dz2_dy2_dx2 <= nr2){
                        int d;                  // edge_cost is a float stored in an 32bit int
                        float *p = (float*) &d; // Make a float pointer point at the integer
                        *p = sqrt((float) dz2_dy2_dx2); // Pretend that the integer is a float and store the value
                        int tile_offset = oz + oy + ox;
                        offsets.emplace_back(z);
                        offsets.emplace_back(y);
                        offsets.emplace_back(x);
                        offsets.emplace_back(tile_offset);
                        offsets.emplace_back(d);
                    }
                }
            }
        }
        return offsets;
    }

    //! Get index of voxel in an axis
    /*!
     * Returns the index of a voxel on a grid in a certain
     * dimension.
     * @param index voxel index
     * @param dim dimension
     * @return index of voxel in axis of dimension
     */
    int get_dim_index_by_voxel(long index, int dim);

    //! Returns a read-only pointer to the header of the map
    const PathMapHeader *get_path_map_header() const { return &pathMapHeader_; }

    //! Returns a pointer to the header of the map in a writable version
    PathMapHeader *get_path_map_header_writable() { return &pathMapHeader_; }

    /// Set the path map header
    void set_path_map_header(PathMapHeader &path_map_header, float resolution = -1.0);

    //! Get the values of all tile
    /*!
     * A tile in an path map contains information
     * on the penalty for visiting a tile, the cost of a path
     * from the origin of a path search to the tile, the density
     * of the tile in addition to other (user-defined) information.
     *
     * When getting information from a tile, the returned values
     * can be cropped to a range.
     *
     * @param value_type specifies the type of the returned information
     * (see: PathMapTileOutputs). Depending on the value type
     * the output can be the penalty for visiting the tile, the total
     * cost of a path to the tile, the density of the tile. Additionally,
     * user-defined content can be accessed.
     * @param bounds bound for cropping the output values
     * @param feature_name name of a feature (when accessing additional
     * information.
     * @return values tiles for the specified parameters.
     * \relates PathMapTile::get_value
     */
    std::vector<float> get_tile_values(
            int value_type = PM_TILE_COST,
            std::pair<float, float> bounds = std::pair<float, float>(
                    {std::numeric_limits<float>::min(),
                     std::numeric_limits<float>::max()}),
            const std::string &feature_name=""
    );
    void get_tile_values(
            float **output, int *nx, int *ny, int *nz,
            int value_type = PM_TILE_COST,
            std::pair<float, float> bounds = std::pair<float, float>(
                    {std::numeric_limits<float>::min(),
                     std::numeric_limits<float>::max()}),
            const std::string &feature_name=""
    );

    /*!
     * Values of tiles
     * @return vector of all tiles in the accessible volume
     */
    std::vector<PathMapTile>& get_tiles();

    /// Change the value of a density inside or outside of a sphere
    /*!
     * Changes the value of the density inside or outside of a sphere.
     * The density is used in the path-search (i.e., the accessible
     * volume calculation
     * @param r0 location of the sphere
     * @param radius radius of the sphere
     * @param value value inside or outside of the sphere
     * @param inverse if set to true (default) the values outside of the sphere
     * are modified. If false the values inside of the sphere are modified.
     */
    void fill_sphere(IMP::algebra::Vector3D r0, double radius, double value, bool inverse=true);

    /*!
     *
     * @param path_begin_idx
     * @param path_end_idx
     */
    void find_path(long path_begin_idx, long path_end_idx = -1, int heuristic_mode = 0);
    void find_path_dijkstra(long path_begin_idx, long path_end_idx = -1);
    void find_path_astar(long path_begin_idx, long path_end_idx = -1);

    std::vector<IMP::algebra::Vector4D> get_xyz_density();

    /// Resamples (updated) the obstacles
    void sample_obstacles(double extra_radius=0.0);

    /*!
     *
     * @param header
     * @param name
     * @param kt
     * @param resolution
     */
    explicit PathMap(
            PathMapHeader &header,
            std::string name = "PathMap%1%",
            IMP::em::KernelType kt = IMP::em::BINARIZED_SPHERE,
            float resolution = -1.0
    );

};


//! Write a av map to a file.
/** Guess the file type from the
    file name. The file formats supported are:
    - .mrc/.map
    - .em
    - .vol
    - .xplor
    \relates AccessibleVolume
*/
IMPEMEXPORT
void write_path_map(
        PathMap *m,
        std::string filename,
        int value_type,
        std::pair<float, float> bounds = std::pair<float, float>(
                {std::numeric_limits<float>::min(),
                 std::numeric_limits<float>::max()}),
        const std::string &feature_name=""
);


IMPBFF_END_NAMESPACE

#endif //IMPBFF_PATHMAP_H
