#ifndef RETARGETING_VISUALIZATION_H
#define RETARGETING_VISUALIZATION_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "retargeting/types.h"

namespace retargeting {

// Every image retarget() produces, for saving and display.
struct RetargetingImages {
    cv::Mat source;
    cv::Mat segmentation;
    cv::Mat saliency;
    cv::Mat significance;
    cv::Mat source_coverage;
    cv::Mat result;
    cv::Mat result_with_grid;
    cv::Mat deformed_mesh;  // warped grid over a faded result
};

// Paints every pixel with the given color of the patch it belongs to,
// e.g. paint_patches(seg, &Patch::segment_color).
cv::Mat paint_patches(const Segmentation &segmentation, cv::Scalar Patch::*color);

// Blends `image` toward white; amount 0 keeps it, 1 gives pure white.
cv::Mat fade_to_white(const cv::Mat &image, double amount);

// Grayscale JET heatmap of a BGR image: blue = low, red = high.
cv::Mat heatmap(const cv::Mat &bgr);

struct FigurePanel {
    cv::Mat image;  // BGR, BGRA or grayscale
    std::string label;
};

// Paper-style figure row: panels side by side at their real pixel size,
// with a label under each one and optionally arrows between them.
cv::Mat compose_row(const std::vector<FigurePanel> &panels, bool arrows = false);

// Stacks figure rows vertically, left-aligned on a white background.
cv::Mat stack_rows(const std::vector<cv::Mat> &rows);

// One figure showing the whole pipeline side by side, paper style:
// source -> segmentation -> saliency -> significance -> deformed mesh -> result.
cv::Mat make_overview(const RetargetingImages &images);

// Writes all pipeline images as PNG files into `directory`.
void save_images(const RetargetingImages &images, const std::string &directory);

// Shows the images in windows and waits for 'q'. Does nothing when OpenCV
// was built without a GUI backend.
void show_images(const RetargetingImages &images);

}  // namespace retargeting

#endif  // RETARGETING_VISUALIZATION_H
