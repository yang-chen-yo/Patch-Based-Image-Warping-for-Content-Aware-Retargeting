// Patch-Based Image Warping for Content-Aware Retargeting
//
// Usage: ./output [target_width target_height]
//   Reads res/gallery.jpg and its saliency map res/gs.jpeg, writes results
//   to result/. The target size defaults to a square (side = source height).

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "retargeting/retarget.h"
#include "retargeting/visualization.h"

namespace {

const char *const kSourcePath = "res/gallery.jpg";
const char *const kSaliencyPath = "res/gs.jpeg";
const char *const kResultDir = "result";

void print_summary(const retargeting::RetargetingOutput &out, cv::Size target_size)
{
    const cv::Mat &source = out.images.source;
    const retargeting::Mesh &mesh = out.mesh;
    const retargeting::StageTimes &t = out.times;

    std::printf("Source image : %d x %d\n", source.cols, source.rows);
    std::printf("Target size  : %d x %d\n", target_size.width, target_size.height);
    std::printf("Patches      : %d raw segments -> %zu after merging tiny/similar patches\n",
                out.segmentation.raw_segment_count, out.segmentation.patches.size());
    std::printf("Mesh         : %u x %u vertices, %zu quads, cell %.2f x %.2f px\n",
                mesh.cols, mesh.rows, mesh.quads.size(), mesh.cell_width, mesh.cell_height);
    std::printf("Timing (s)\n");
    std::printf("  segmentation    %8.3f\n", t.segmentation);
    std::printf("  significance    %8.3f\n", t.significance);
    std::printf("  mesh            %8.3f\n", t.mesh);
    std::printf("  warping (CPLEX) %8.3f\n", t.warping);
    std::printf("  rendering       %8.3f\n", t.rendering);
    std::printf("  total           %8.3f\n", t.total());
}

}  // namespace

int main(int argc, const char *argv[])
{
    using namespace retargeting;

    cv::Mat source = cv::imread(kSourcePath);
    if (source.empty()) {
        std::cerr << "Failed to load input image: " << kSourcePath << std::endl;
        return -1;
    }
    cv::Mat saliency = cv::imread(kSaliencyPath);
    if (saliency.empty()) {
        std::cerr << "Failed to load saliency map: " << kSaliencyPath << std::endl;
        return -1;
    }

    // A gentler target ratio than the default square leaves the optimizer
    // more slack to hide the resize in genuinely low-significance regions
    // instead of visibly warping content that fills the whole frame.
    cv::Size target_size(source.rows, source.rows);
    if (argc > 2) {
        try {
            target_size = cv::Size(std::stoi(argv[1]), std::stoi(argv[2]));
        } catch (const std::exception &) {
            std::cerr << "Usage: " << argv[0] << " [target_width target_height]" << std::endl;
            return -1;
        }
    }

    RetargetingOutput out;
    try {
        out = retarget(source, saliency, target_size);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    print_summary(out, target_size);
    save_images(out.images, kResultDir);
    show_images(out.images);
    return 0;
}
