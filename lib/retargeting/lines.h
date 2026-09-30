// Straight lines of the source image: detection, sampling, and how straight
// they stay after warping. Used by the line-preservation energy, which is an
// extension beyond the paper (see WarpParams::line_weight).
#ifndef RETARGETING_LINES_H
#define RETARGETING_LINES_H

#include <vector>

#include <opencv2/core.hpp>

#include "retargeting/types.h"

namespace retargeting {

struct LineSegment {
    cv::Point2f a;
    cv::Point2f b;
};

// Line segments of a BGR image at least `min_length` pixels long
// (FastLineDetector from opencv_contrib ximgproc).
std::vector<LineSegment> detect_lines(const cv::Mat &bgr, float min_length);

// Evenly spaced points from a to b, at most `spacing` pixels apart.
std::vector<cv::Point2f> sample_line(const LineSegment &line, float spacing);

// Maps a source point to the retargeted image exactly like the renderer does
// (the perspective transform of the quad that contains it).
class PointWarper {
public:
    PointWarper(const Mesh &mesh, const std::vector<cv::Vec2f> &target_vertices);
    cv::Point2f operator()(cv::Point2f point) const;

private:
    const Mesh &mesh_;
    std::vector<cv::Matx33d> transforms_;  // one per quad
};

// How far each line bends after warping: the largest distance (px) of its
// warped points from the straight line through its warped endpoints.
std::vector<double> line_deviations(const std::vector<LineSegment> &lines, const PointWarper &warp);

}  // namespace retargeting

#endif  // RETARGETING_LINES_H
