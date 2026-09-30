#include "retargeting/retarget.h"

#include "retargeting/mesh.h"
#include "retargeting/renderer.h"
#include "retargeting/segmentation.h"
#include "retargeting/significance.h"
#include "retargeting/timer.h"
#include "retargeting/warping.h"

namespace retargeting {

RetargetingOutput retarget(const cv::Mat &source, const cv::Mat &saliency, cv::Size target_size,
                           const RetargetingParams &params)
{
    RetargetingOutput out;
    Stopwatch stopwatch;

    out.segmentation = segment_image(source, params.segmentation);
    out.times.segmentation = stopwatch.lap();

    compute_significance(saliency, out.segmentation);
    out.times.significance = stopwatch.lap();

    out.mesh = build_mesh(source.size(), params.mesh);
    out.times.mesh = stopwatch.lap();

    if (params.warp.line_weight > 0) out.lines = detect_lines(source, params.warp.line_min_length);
    out.target_vertices = solve_warp(out.mesh, out.segmentation, target_size, params.warp, out.lines);
    out.times.warping = stopwatch.lap();

    WarpedImage warped = render_warp(source, out.mesh, out.target_vertices, target_size);
    out.times.rendering = stopwatch.lap();

    RetargetingImages &images = out.images;
    images.source = source;
    images.saliency = saliency;
    images.segmentation = paint_patches(out.segmentation, &Patch::segment_color);
    images.significance = paint_patches(out.segmentation, &Patch::significance_color);
    images.result = warped.result;
    images.source_coverage = warped.source_coverage;
    images.result_with_grid = draw_mesh(images.result, out.mesh, out.target_vertices);

    // A semi-transparent grid over a slightly faded result, so both the grid
    // and the content stay readable.
    const cv::Mat mesh_base = fade_to_white(images.result, 0.3);
    cv::addWeighted(mesh_base, 0.4,
                    draw_mesh(mesh_base, out.mesh, out.target_vertices, cv::Scalar(140, 40, 0, 255)), 0.6,
                    0, images.deformed_mesh);

    return out;
}

}  // namespace retargeting
