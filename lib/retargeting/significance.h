#ifndef RETARGETING_SIGNIFICANCE_H
#define RETARGETING_SIGNIFICANCE_H

#include <opencv2/core.hpp>

#include "retargeting/types.h"

namespace retargeting {

// Averages a BGR saliency map over each patch and fills significance_color
// and the normalized saliency_value of every patch.
void compute_significance(const cv::Mat &saliency, Segmentation &segmentation);

}  // namespace retargeting

#endif  // RETARGETING_SIGNIFICANCE_H
