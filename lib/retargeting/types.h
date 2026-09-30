// Core data structures shared by every stage of the retargeting pipeline.
#ifndef RETARGETING_TYPES_H
#define RETARGETING_TYPES_H

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace retargeting {

// A patch is one region of the (merged) segmentation.
struct Patch {
    unsigned int id = 0;
    unsigned int size = 0;          // number of pixels
    cv::Scalar segment_color;       // mean BGR color of the source image
    cv::Scalar significance_color;  // mean BGR color of the saliency map
    double saliency_value = 0.0;    // normalized significance in [0, 1]
};

struct Segmentation {
    cv::Mat labels;               // CV_32S, patch id of every pixel
    std::vector<Patch> patches;   // indexed by patch id
    int raw_segment_count = 0;    // segments before merging
};

// Mesh edge, stored as indices into Mesh::vertices.
struct Edge {
    unsigned int from;
    unsigned int to;
};

// Four vertex indices of one grid cell, counterclockwise.
using Quad = std::array<unsigned int, 4>;

// Regular grid laid over the source image.
struct Mesh {
    unsigned int cols = 0;
    unsigned int rows = 0;
    float cell_width = 0.0f;
    float cell_height = 0.0f;

    std::vector<cv::Vec2f> vertices;  // row-major, rows * cols
    std::vector<Edge> edges;
    std::vector<Quad> quads;          // row-major

    unsigned int index(unsigned int row, unsigned int col) const { return row * cols + col; }
};

}  // namespace retargeting

#endif  // RETARGETING_TYPES_H
