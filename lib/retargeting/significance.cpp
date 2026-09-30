#include "retargeting/significance.h"

#include <algorithm>

namespace retargeting {

void compute_significance(const cv::Mat &saliency, Segmentation &segmentation)
{
    std::vector<Patch> &patches = segmentation.patches;

    // Mean saliency color, accumulated as pixel / size to keep the original rounding.
    for (int i = 0; i < saliency.rows; i++) {
        const uchar *pixel = saliency.ptr<uchar>(i);
        const int *label = segmentation.labels.ptr<int>(i);
        for (int j = 0; j < saliency.cols; j++) {
            Patch &patch = patches[label[j]];
            for (int c = 0; c < 3; c++) patch.significance_color[c] += pixel[j * 3 + c] / (double) patch.size;
        }
    }

    // Pack B, G, R into a single scalar, then normalize to [0, 1].
    double min_saliency = 2e9;
    double max_saliency = -2e9;
    for (Patch &patch : patches) {
        patch.saliency_value = 0;
        patch.saliency_value += patch.significance_color[0];
        patch.saliency_value += patch.significance_color[1] * 256;
        patch.saliency_value += patch.significance_color[2] * 65536;
        min_saliency = std::min(min_saliency, patch.saliency_value);
        max_saliency = std::max(max_saliency, patch.saliency_value);
    }
    for (Patch &patch : patches) {
        patch.saliency_value = (patch.saliency_value - min_saliency) / (max_saliency - min_saliency);
    }
}

}  // namespace retargeting
