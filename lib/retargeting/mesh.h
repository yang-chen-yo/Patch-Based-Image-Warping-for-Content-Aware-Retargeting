#ifndef RETARGETING_MESH_H
#define RETARGETING_MESH_H

#include <opencv2/core.hpp>

#include "retargeting/config.h"
#include "retargeting/types.h"

namespace retargeting {

// Builds a regular grid mesh that spans the whole image.
Mesh build_mesh(cv::Size image_size, const MeshParams &params);

// A source-image point expressed in the mesh: the quad that contains it and
// its bilinear weights for that quad's vertices (same order as Quad).
struct MeshCoordinate {
    unsigned int quad;
    float weights[4];
};

// Locates a point of the source image (clamped to the mesh) in the mesh.
MeshCoordinate locate(const Mesh &mesh, cv::Point2f point);

}  // namespace retargeting

#endif  // RETARGETING_MESH_H
