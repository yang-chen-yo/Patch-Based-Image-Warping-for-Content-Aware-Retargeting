// The complete retargeting pipeline in a single call, shared by the main
// program and the experiments.
#ifndef RETARGETING_RETARGET_H
#define RETARGETING_RETARGET_H

#include <vector>

#include <opencv2/core.hpp>

#include "retargeting/config.h"
#include "retargeting/lines.h"
#include "retargeting/types.h"
#include "retargeting/visualization.h"

namespace retargeting {

struct RetargetingParams {
    SegmentationParams segmentation;
    MeshParams mesh;
    WarpParams warp;

    // Defaults follow the paper; tuned() is our earlier hand-tuned setup.
    static RetargetingParams tuned()
    {
        RetargetingParams p;
        p.warp = WarpParams::tuned();
        return p;
    }
};

// Wall-clock seconds spent in each stage. Saliency detection runs in Python
// beforehand and is timed by pipeline.sh.
struct StageTimes {
    double segmentation = 0.0;
    double significance = 0.0;
    double mesh = 0.0;
    double warping = 0.0;    // building and solving the CPLEX model
    double rendering = 0.0;  // per-quad perspective warp

    double total() const { return segmentation + significance + mesh + warping + rendering; }
};

struct RetargetingOutput {
    RetargetingImages images;
    Segmentation segmentation;
    Mesh mesh;
    std::vector<cv::Vec2f> target_vertices;
    std::vector<LineSegment> lines;  // detected lines, only when warp.line_weight > 0
    StageTimes times;
};

// Retargets `source` (BGR) to `target_size` using its BGR saliency map.
// Throws like solve_warp() on an invalid target size or a failed optimization.
RetargetingOutput retarget(const cv::Mat &source, const cv::Mat &saliency, cv::Size target_size,
                           const RetargetingParams &params = RetargetingParams());

}  // namespace retargeting

#endif  // RETARGETING_RETARGET_H
