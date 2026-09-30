#ifndef RETARGETING_RENDERER_H
#define RETARGETING_RENDERER_H

#include <vector>

#include <opencv2/core.hpp>

#include "retargeting/types.h"

namespace retargeting {

struct WarpedImage {
    cv::Mat result;           // BGRA, the retargeted image
    cv::Mat source_coverage;  // BGRA, the source pixels covered by mesh quads
};

// Maps every mesh quad of `source` onto its solved position with a
// per-quad perspective transform.
WarpedImage render_warp(const cv::Mat &source, const Mesh &mesh,
                        const std::vector<cv::Vec2f> &target_vertices, cv::Size target_size);

// Returns a copy of `image` with the mesh quads (at `vertices`) drawn on top.
cv::Mat draw_mesh(const cv::Mat &image, const Mesh &mesh, const std::vector<cv::Vec2f> &vertices,
                  const cv::Scalar &color = cv::Scalar(0, 255, 0, 255));

}  // namespace retargeting

#endif  // RETARGETING_RENDERER_H
