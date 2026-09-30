#include "retargeting/renderer.h"

#include <opencv2/imgproc.hpp>

namespace retargeting {

WarpedImage render_warp(const cv::Mat &source, const Mesh &mesh,
                        const std::vector<cv::Vec2f> &target_vertices, cv::Size target_size)
{
    WarpedImage output;
    output.result = cv::Mat(target_size, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    output.source_coverage = cv::Mat(source.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

    for (const Quad &quad : mesh.quads) {
        cv::Point2f src[4], dst[4];
        std::vector<std::vector<cv::Point>> contour(1);
        for (int j = 0; j < 4; j++) {
            src[j] = cv::Point2f(mesh.vertices[quad[j]][0], mesh.vertices[quad[j]][1]);
            dst[j] = cv::Point2f(target_vertices[quad[j]][0], target_vertices[quad[j]][1]);
            contour[0].push_back(cv::Point((int) src[j].x, (int) src[j].y));
        }

        // Cut the quad out of the source as an opaque BGRA patch.
        cv::Mat mask(source.size(), CV_8UC1, cv::Scalar::all(0));
        cv::drawContours(mask, contour, 0, cv::Scalar(255, 255, 255), cv::FILLED);
        cv::Mat quad_image(source.size(), source.type(), cv::Scalar(0, 0, 0));
        source.copyTo(quad_image, mask);
        cv::cvtColor(quad_image, quad_image, cv::COLOR_BGR2BGRA);

        // INTER_NEAREST, not INTER_LINEAR: bilinear-warping a quad that's
        // opaque inside and (0,0,0,0) just outside blends the two right at
        // the quad's own edge, baking a band of darkened, partly-transparent
        // pixels into every single quad boundary. Nearest-neighbor resampling
        // never mixes a pixel with the empty border, so there's nothing to
        // composite a seam from.
        cv::Mat transform = cv::getPerspectiveTransform(src, dst);
        cv::Mat warped_quad, warped_mask;
        cv::warpPerspective(quad_image, warped_quad, transform, target_size,
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

        // Warp the quad's own mask the same way and composite with copyTo
        // (overwrite) instead of cv::add: adjacent quads' rasterized masks
        // can overlap by a pixel at shared edges, and cv::add() there would
        // saturate-sum both quads' colors into a bright seam.
        cv::warpPerspective(mask, warped_mask, transform, target_size,
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // The optimizer may squeeze a quad to almost zero width; its
        // perspective transform is then so ill-conditioned that the warped
        // mask spills over the whole image and overwrites every quad drawn
        // before it. Clip the mask to the quad's target bounding box.
        cv::Rect bounds = cv::boundingRect(std::vector<cv::Point2f>(dst, dst + 4));
        bounds = cv::Rect(bounds.x - 1, bounds.y - 1, bounds.width + 2, bounds.height + 2) &
                 cv::Rect(cv::Point(0, 0), target_size);
        cv::Mat clipped_mask = cv::Mat::zeros(target_size, CV_8UC1);
        warped_mask(bounds).copyTo(clipped_mask(bounds));

        warped_quad.copyTo(output.result, clipped_mask);
        quad_image.copyTo(output.source_coverage, mask);
    }

    return output;
}

cv::Mat draw_mesh(const cv::Mat &image, const Mesh &mesh, const std::vector<cv::Vec2f> &vertices,
                  const cv::Scalar &color)
{
    cv::Mat canvas = image.clone();
    for (const Quad &quad : mesh.quads) {
        for (int j = 0; j < 4; j++) {
            const cv::Vec2f &a = vertices[quad[j]];
            const cv::Vec2f &b = vertices[quad[(j + 1) % 4]];
            cv::line(canvas, cv::Point((int) a[0], (int) a[1]), cv::Point((int) b[0], (int) b[1]),
                     color, 1, cv::LINE_AA);
        }
    }
    return canvas;
}

}  // namespace retargeting
