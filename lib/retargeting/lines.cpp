#include "retargeting/lines.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>

#include "retargeting/mesh.h"

namespace retargeting {

std::vector<LineSegment> detect_lines(const cv::Mat &bgr, float min_length)
{
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec4f> found;
    cv::ximgproc::createFastLineDetector((int) min_length)->detect(gray, found);

    std::vector<LineSegment> lines;
    for (const cv::Vec4f &f : found) {
        LineSegment line = {cv::Point2f(f[0], f[1]), cv::Point2f(f[2], f[3])};
        if (cv::norm(line.b - line.a) >= min_length) lines.push_back(line);
    }
    return lines;
}

std::vector<cv::Point2f> sample_line(const LineSegment &line, float spacing)
{
    const int segments = std::max(1, (int) std::ceil(cv::norm(line.b - line.a) / spacing));
    std::vector<cv::Point2f> points;
    for (int i = 0; i <= segments; i++) {
        points.push_back(line.a + (line.b - line.a) * ((float) i / segments));
    }
    return points;
}

PointWarper::PointWarper(const Mesh &mesh, const std::vector<cv::Vec2f> &target_vertices) : mesh_(mesh)
{
    for (const Quad &quad : mesh.quads) {
        cv::Point2f src[4], dst[4];
        for (int j = 0; j < 4; j++) {
            src[j] = cv::Point2f(mesh.vertices[quad[j]][0], mesh.vertices[quad[j]][1]);
            dst[j] = cv::Point2f(target_vertices[quad[j]][0], target_vertices[quad[j]][1]);
        }
        transforms_.push_back(cv::Matx33d(cv::getPerspectiveTransform(src, dst)));
    }
}

cv::Point2f PointWarper::operator()(cv::Point2f point) const
{
    const cv::Matx33d &h = transforms_[locate(mesh_, point).quad];
    const cv::Vec3d p = h * cv::Vec3d(point.x, point.y, 1.0);
    return cv::Point2f((float) (p[0] / p[2]), (float) (p[1] / p[2]));
}

std::vector<double> line_deviations(const std::vector<LineSegment> &lines, const PointWarper &warp)
{
    std::vector<double> deviations;
    for (const LineSegment &line : lines) {
        std::vector<cv::Point2f> warped;
        for (const cv::Point2f &p : sample_line(line, 2.0f)) warped.push_back(warp(p));

        const cv::Point2f a = warped.front(), d = warped.back() - warped.front();
        const double length = cv::norm(d);
        double worst = 0.0;
        for (const cv::Point2f &p : warped) {
            const cv::Point2f ap = p - a;
            const double distance = length > 1e-6 ? std::fabs(d.x * ap.y - d.y * ap.x) / length : cv::norm(ap);
            worst = std::max(worst, distance);
        }
        deviations.push_back(worst);
    }
    return deviations;
}

}  // namespace retargeting
