// Reproduces the experiments of Section IV of the paper on our own images.
//
// Usage: ./experiments [experiment ...] [--target W H]
//   experiment: timing | dlt | alpha | grid | overseg | aspect | all (default)
//   --target:   target size of the ablations (default: 60% width, same height)
//
// Reads res/gallery.jpg and res/gs.jpeg like the main program and writes one
// figure (and, where useful, a CSV table) per experiment to result/experiments/.

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "retargeting/retarget.h"
#include "retargeting/visualization.h"

using namespace retargeting;

namespace {

const char *const kSourcePath = "res/gallery.jpg";
const char *const kSaliencyPath = "res/gs.jpeg";
const std::string kOutputDir = "result/experiments";

struct Input {
    cv::Mat source;
    cv::Mat saliency;
    cv::Size target;  // target size of the ablation experiments
};

std::string format(const char *fmt, double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, value);
    return buffer;
}

void save_figure(const std::string &name, const cv::Mat &figure)
{
    const std::string path = kOutputDir + "/" + name;
    cv::imwrite(path, figure);
    std::printf("  -> %s\n", path.c_str());
}

// Paper Section IV: average time of each stage (paper: 1024x768 image,
// warping 0.063 s, segmentation 0.06 s, saliency 10.01 s).
void run_timing(const Input &in)
{
    const int runs = 5;
    StageTimes sum;
    for (int i = 0; i < runs; i++) {
        StageTimes t = retarget(in.source, in.saliency, in.target).times;
        sum.segmentation += t.segmentation;
        sum.significance += t.significance;
        sum.mesh += t.mesh;
        sum.warping += t.warping;
        sum.rendering += t.rendering;
    }

    const std::vector<std::pair<std::string, double>> rows = {
        {"segmentation", sum.segmentation / runs},
        {"significance", sum.significance / runs},
        {"mesh", sum.mesh / runs},
        {"warping (CPLEX)", sum.warping / runs},
        {"rendering", sum.rendering / runs},
        {"total", sum.total() / runs},
    };
    std::printf("  image %d x %d -> %d x %d, average of %d runs\n", in.source.cols, in.source.rows,
                in.target.width, in.target.height, runs);
    std::ofstream csv(kOutputDir + "/timing.csv");
    csv << "stage,seconds\n";
    for (const auto &row : rows) {
        std::printf("  %-16s %8.3f s\n", row.first.c_str(), row.second);
        csv << row.first << "," << row.second << "\n";
    }
    std::printf("  -> %s/timing.csv (saliency detection is timed by pipeline.sh)\n", kOutputDir.c_str());
}

// Paper Fig. 4: without the linear scaling term D_LT, low-significance
// regions get over-squeezed.
void run_dlt(const Input &in)
{
    RetargetingParams without_dlt;
    without_dlt.warp.dlt_weight = 0.0;
    RetargetingOutput a = retarget(in.source, in.saliency, in.target, without_dlt);
    RetargetingOutput b = retarget(in.source, in.saliency, in.target);

    save_figure("ablation_dlt.png", compose_row({
        {in.source, "(a) Source"},
        {a.images.deformed_mesh, "(b) w/o DLT: mesh"},
        {a.images.result, "(c) w/o DLT"},
        {b.images.deformed_mesh, "(d) with DLT: mesh"},
        {b.images.result, "(e) with DLT"},
    }));
}

// Paper Fig. 10: small alpha lets salient objects follow the linear scaling,
// large alpha keeps them rigid.
void run_alpha(const Input &in)
{
    std::vector<FigurePanel> results, meshes;
    for (double alpha : {0.0, 0.2, 0.5, 0.8, 1.0}) {
        RetargetingParams params;
        params.warp.alpha = alpha;
        RetargetingOutput out = retarget(in.source, in.saliency, in.target, params);
        const std::string label = format("alpha = %.1f", alpha);
        results.push_back({out.images.result, label});
        meshes.push_back({out.images.deformed_mesh, label});
    }
    save_figure("ablation_alpha.png", stack_rows({compose_row(results), compose_row(meshes)}));
}

// Paper Fig. 9: finer grids give better quality at a higher cost.
void run_grid(const Input &in)
{
    std::vector<FigurePanel> results, meshes;
    std::ofstream csv(kOutputDir + "/grid_timing.csv");
    csv << "grid_px,quads,warping_seconds,total_seconds\n";
    for (int grid : {10, 20, 30, 40}) {
        RetargetingParams params;
        params.mesh.grid_size = (float) grid;
        RetargetingOutput out = retarget(in.source, in.saliency, in.target, params);

        std::printf("  %2dx%-2d px: %5zu quads, warping %7.3f s, total %7.3f s\n", grid, grid,
                    out.mesh.quads.size(), out.times.warping, out.times.total());
        csv << grid << "," << out.mesh.quads.size() << "," << out.times.warping << "," << out.times.total() << "\n";

        const std::string label = std::to_string(grid) + "x" + std::to_string(grid) + format(", %.2f s", out.times.warping);
        results.push_back({out.images.result, label});
        meshes.push_back({out.images.deformed_mesh, label});
    }
    save_figure("ablation_grid.png", stack_rows({compose_row(meshes), compose_row(results)}));
    std::printf("  -> %s/grid_timing.csv\n", kOutputDir.c_str());
}

// Paper Fig. 7: skipping the patch merging (extreme over-segmentation)
// should barely change the result.
void run_overseg(const Input &in)
{
    RetargetingParams raw;
    raw.segmentation.merge_patches = false;
    RetargetingOutput a = retarget(in.source, in.saliency, in.target, raw);
    RetargetingOutput b = retarget(in.source, in.saliency, in.target);

    const std::string a_patches = std::to_string(a.segmentation.patches.size()) + " patches";
    const std::string b_patches = std::to_string(b.segmentation.patches.size()) + " patches";
    save_figure("ablation_oversegmentation.png", compose_row({
        {a.images.segmentation, "(a) No merging, " + a_patches},
        {a.images.result, "(b) Result of (a)"},
        {b.images.segmentation, "(c) Merged, " + b_patches},
        {b.images.result, "(d) Result of (c)"},
    }));
}

// Paper Fig. 8: resizing to 4:3 and 1:1 horizontally and vertically,
// compared with linear scaling.
void run_aspect(const Input &in)
{
    const int w = in.source.cols, h = in.source.rows;
    const std::vector<std::pair<cv::Size, std::string>> targets = {
        {cv::Size(h * 4 / 3, h), "width -> 4:3"},
        {cv::Size(h, h), "width -> 1:1"},
        {cv::Size(w, w * 3 / 4), "height -> 4:3"},
        {cv::Size(w, w), "height -> 1:1"},
    };

    std::vector<FigurePanel> linear, ours;
    for (const auto &target : targets) {
        if (target.first == in.source.size()) continue;  // source already has this ratio
        const std::string size = std::to_string(target.first.width) + "x" + std::to_string(target.first.height);
        cv::Mat scaled;
        cv::resize(in.source, scaled, target.first, 0, 0, cv::INTER_LINEAR);
        linear.push_back({scaled, "Linear, " + target.second + " (" + size + ")"});
        ours.push_back({retarget(in.source, in.saliency, target.first).images.result,
                        "Ours, " + target.second + " (" + size + ")"});
    }
    save_figure("aspect_ratios.png", stack_rows({
        compose_row({{in.source, "Source " + std::to_string(w) + "x" + std::to_string(h)}}),
        compose_row(linear),
        compose_row(ours),
    }));
}

void usage(const char *program)
{
    std::cerr << "Usage: " << program << " [timing|dlt|alpha|grid|overseg|aspect|all ...] [--target W H]\n";
}

}  // namespace

int main(int argc, const char *argv[])
{
    const std::map<std::string, std::function<void(const Input &)>> experiments = {
        {"timing", run_timing}, {"dlt", run_dlt}, {"alpha", run_alpha},
        {"grid", run_grid}, {"overseg", run_overseg}, {"aspect", run_aspect},
    };
    const std::vector<std::string> all = {"timing", "dlt", "alpha", "grid", "overseg", "aspect"};

    Input in;
    in.source = cv::imread(kSourcePath);
    in.saliency = cv::imread(kSaliencyPath);
    if (in.source.empty() || in.saliency.empty()) {
        std::cerr << "Failed to load " << kSourcePath << " / " << kSaliencyPath << std::endl;
        return -1;
    }
    in.target = cv::Size(cvRound(in.source.cols * 0.6), in.source.rows);

    std::vector<std::string> selected;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--target" && i + 2 < argc) {
            try {
                in.target = cv::Size(std::stoi(argv[i + 1]), std::stoi(argv[i + 2]));
            } catch (const std::exception &) {
                usage(argv[0]);
                return -1;
            }
            i += 2;
        } else if (arg == "all") {
            selected.insert(selected.end(), all.begin(), all.end());
        } else if (experiments.count(arg)) {
            selected.push_back(arg);
        } else {
            usage(argv[0]);
            return -1;
        }
    }
    if (selected.empty()) selected = all;

    std::filesystem::create_directories(kOutputDir);
    std::printf("Source %d x %d, ablation target %d x %d\n", in.source.cols, in.source.rows,
                in.target.width, in.target.height);
    for (const std::string &name : selected) {
        std::printf("== %s\n", name.c_str());
        try {
            experiments.at(name)(in);
        } catch (const std::exception &e) {
            std::cerr << "  failed: " << e.what() << std::endl;
        }
    }
    return 0;
}
