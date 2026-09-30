#include "retargeting/visualization.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace retargeting {
namespace {

// Converts to 3-channel BGR; transparent pixels become white.
cv::Mat to_bgr_on_white(const cv::Mat &image)
{
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        bgr = cv::Mat(image.size(), CV_8UC3);
        for (int i = 0; i < image.rows; i++) {
            const cv::Vec4b *in = image.ptr<cv::Vec4b>(i);
            cv::Vec3b *out = bgr.ptr<cv::Vec3b>(i);
            for (int j = 0; j < image.cols; j++) {
                double alpha = in[j][3] / 255.0;
                for (int c = 0; c < 3; c++) out[j][c] = cv::saturate_cast<uchar>(in[j][c] * alpha + 255 * (1 - alpha));
            }
        }
    } else {
        bgr = image.clone();
    }
    return bgr;
}

}  // namespace

cv::Mat fade_to_white(const cv::Mat &image, double amount)
{
    cv::Mat white(image.size(), image.type(), cv::Scalar::all(255));
    cv::Mat faded;
    cv::addWeighted(image, 1 - amount, white, amount, 0, faded);
    if (image.channels() == 4) {
        // keep the original alpha so transparent pixels stay transparent
        std::vector<cv::Mat> faded_channels, channels;
        cv::split(faded, faded_channels);
        cv::split(image, channels);
        faded_channels[3] = channels[3];
        cv::merge(faded_channels, faded);
    }
    return faded;
}

cv::Mat heatmap(const cv::Mat &bgr)
{
    cv::Mat gray, colored;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::applyColorMap(gray, colored, cv::COLORMAP_JET);
    return colored;
}

cv::Mat compose_row(const std::vector<FigurePanel> &panels, bool arrows)
{
    // Every spacing scales with the tallest panel.
    int panel_height = 0, panels_width = 0;
    for (const FigurePanel &panel : panels) {
        panel_height = std::max(panel_height, panel.image.rows);
        panels_width += panel.image.cols;
    }
    const double unit = panel_height / 630.0;
    const int margin = cvRound(24 * unit);
    const int gap = cvRound((arrows ? 70 : 30) * unit);
    const int label_height = cvRound(70 * unit);
    const int font = cv::FONT_HERSHEY_DUPLEX;
    const int thickness = std::max(1, cvRound(2 * unit));

    cv::Mat canvas(margin * 2 + panel_height + label_height,
                   margin * 2 + panels_width + gap * ((int) panels.size() - 1),
                   CV_8UC3, cv::Scalar::all(255));

    int x = margin;
    for (size_t i = 0; i < panels.size(); i++) {
        cv::Mat panel = to_bgr_on_white(panels[i].image);
        int y = margin + (panel_height - panel.rows) / 2;
        panel.copyTo(canvas(cv::Rect(x, y, panel.cols, panel.rows)));

        // Shrink the label if it would not fit under a narrow panel.
        double font_scale = 1.0 * unit;
        int baseline = 0;
        cv::Size text = cv::getTextSize(panels[i].label, font, font_scale, thickness, &baseline);
        if (text.width > panel.cols + gap) {
            font_scale *= (double) (panel.cols + gap) / text.width;
            text = cv::getTextSize(panels[i].label, font, font_scale, thickness, &baseline);
        }
        cv::Point text_origin(x + (panel.cols - text.width) / 2,
                              margin + panel_height + (label_height + text.height) / 2);
        cv::putText(canvas, panels[i].label, text_origin, font, font_scale, cv::Scalar::all(0), thickness,
                    cv::LINE_AA);

        x += panel.cols;
        if (i + 1 < panels.size()) {
            if (arrows) {
                int arrow_y = margin + panel_height / 2;
                cv::arrowedLine(canvas, cv::Point(x + gap / 5, arrow_y), cv::Point(x + gap * 4 / 5, arrow_y),
                                cv::Scalar::all(60), thickness + 1, cv::LINE_AA, 0, 0.35);
            }
            x += gap;
        }
    }
    return canvas;
}

cv::Mat stack_rows(const std::vector<cv::Mat> &rows)
{
    int width = 0, height = 0;
    for (const cv::Mat &row : rows) {
        width = std::max(width, row.cols);
        height += row.rows;
    }
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar::all(255));
    int y = 0;
    for (const cv::Mat &row : rows) {
        row.copyTo(canvas(cv::Rect(0, y, row.cols, row.rows)));
        y += row.rows;
    }
    return canvas;
}

cv::Mat make_overview(const RetargetingImages &images)
{
    return compose_row({
        {images.source, "(a) Source"},
        {images.segmentation, "(b) Segmentation"},
        {heatmap(images.saliency), "(c) Saliency"},
        {heatmap(images.significance), "(d) Significance"},
        {images.deformed_mesh, "(e) Deformed mesh"},
        {images.result, "(f) Result"},
    }, true);
}

cv::Mat paint_patches(const Segmentation &segmentation, cv::Scalar Patch::*color)
{
    const cv::Mat &labels = segmentation.labels;
    cv::Mat result = cv::Mat::zeros(labels.rows, labels.cols, CV_8UC3);
    for (int i = 0; i < labels.rows; i++) {
        const int *label = labels.ptr<int>(i);
        uchar *pixel = result.ptr<uchar>(i);
        for (int j = 0; j < labels.cols; j++) {
            const cv::Scalar &c = segmentation.patches[label[j]].*color;
            pixel[j * 3] = (uchar) c[0];
            pixel[j * 3 + 1] = (uchar) c[1];
            pixel[j * 3 + 2] = (uchar) c[2];
        }
    }
    return result;
}

void save_images(const RetargetingImages &images, const std::string &directory)
{
    const std::vector<std::pair<std::string, cv::Mat>> files = {
        {"segmentation.png", images.segmentation},
        {"saliency_heatmap.png", heatmap(images.saliency)},
        {"significance.png", images.significance},
        {"significance_heatmap.png", heatmap(images.significance)},
        {"source.png", images.source_coverage},
        {"result_gs.png", images.result},
        {"result_gs_with_grid.png", images.result_with_grid},
        {"deformed_mesh.png", images.deformed_mesh},
        {"pipeline_overview.png", make_overview(images)},
    };
    for (const auto &file : files) {
        cv::imwrite(directory + "/" + file.first, file.second);
    }
}

void show_images(const RetargetingImages &images)
{
    const std::vector<std::pair<std::string, cv::Mat>> windows = {
        {"Source", images.source},
        {"Segmentation", images.segmentation},
        {"Saliency", images.saliency},
        {"Significance Map", images.significance},
        {"GridSource", images.source_coverage},
        {"Result", images.result},
    };
    try {
        for (const auto &window : windows) {
            cv::namedWindow(window.first);
            cv::imshow(window.first, window.second);
        }
    } catch (const cv::Exception &e) {
        std::cerr << "No GUI backend available, skipping window display (results are still saved to result/): "
                  << e.what() << std::endl;
        return;
    }

    while (cv::waitKey(0) != 'q') {
    }
    cv::destroyAllWindows();
}

}  // namespace retargeting
