#include "retargeting/mesh.h"

#include <algorithm>

namespace retargeting {

Mesh build_mesh(cv::Size image_size, const MeshParams &params)
{
    Mesh mesh;
    mesh.cols = (unsigned int) ((image_size.width - 1) / params.grid_size) + 1;
    mesh.rows = (unsigned int) ((image_size.height - 1) / params.grid_size) + 1;
    mesh.cell_width = (float) (image_size.width - 1) / (mesh.cols - 1);
    mesh.cell_height = (float) (image_size.height - 1) / (mesh.rows - 1);

    for (unsigned int row = 0; row < mesh.rows; row++) {
        for (unsigned int col = 0; col < mesh.cols; col++) {
            mesh.vertices.push_back(cv::Vec2f(col * mesh.cell_width, row * mesh.cell_height));
        }
    }

    // Each quad adds its bottom and right edges; its left / top edge is
    // already added by the neighboring quad, except in the first column / row.
    // This way every grid edge appears exactly once.
    for (unsigned int row = 0; row < mesh.rows - 1; row++) {
        for (unsigned int col = 0; col < mesh.cols - 1; col++) {
            unsigned int index = mesh.index(row, col);
            Quad quad = {index, index + mesh.cols, index + mesh.cols + 1, index + 1};

            if (col == 0) mesh.edges.push_back({quad[0], quad[1]});  // left
            mesh.edges.push_back({quad[1], quad[2]});                // bottom
            mesh.edges.push_back({quad[3], quad[2]});                // right
            if (row == 0) mesh.edges.push_back({quad[0], quad[3]});  // top

            mesh.quads.push_back(quad);
        }
    }

    return mesh;
}

MeshCoordinate locate(const Mesh &mesh, cv::Point2f point)
{
    const float x = point.x / mesh.cell_width, y = point.y / mesh.cell_height;
    const unsigned int col = (unsigned int) std::min(std::max(x, 0.0f), (float) (mesh.cols - 2));
    const unsigned int row = (unsigned int) std::min(std::max(y, 0.0f), (float) (mesh.rows - 2));
    const float u = std::min(std::max(x - col, 0.0f), 1.0f);
    const float v = std::min(std::max(y - row, 0.0f), 1.0f);

    // Quad order: top-left, bottom-left, bottom-right, top-right.
    MeshCoordinate coordinate;
    coordinate.quad = row * (mesh.cols - 1) + col;
    coordinate.weights[0] = (1 - u) * (1 - v);
    coordinate.weights[1] = (1 - u) * v;
    coordinate.weights[2] = u * v;
    coordinate.weights[3] = u * (1 - v);
    return coordinate;
}

}  // namespace retargeting
