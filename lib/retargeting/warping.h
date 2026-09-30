#ifndef RETARGETING_WARPING_H
#define RETARGETING_WARPING_H

#include <vector>

#include <opencv2/core.hpp>

#include "retargeting/config.h"
#include "retargeting/lines.h"
#include "retargeting/types.h"

namespace retargeting {

// Solves the patch-based warping energy with CPLEX and returns the target
// position of every mesh vertex (same indexing as mesh.vertices). `lines`
// are only used when params.line_weight > 0.
// Throws std::invalid_argument for an empty target size and
// std::runtime_error if the optimization fails.
std::vector<cv::Vec2f> solve_warp(const Mesh &mesh, const Segmentation &segmentation,
                                  cv::Size target_size, const WarpParams &params,
                                  const std::vector<LineSegment> &lines = {});

}  // namespace retargeting

#endif  // RETARGETING_WARPING_H
