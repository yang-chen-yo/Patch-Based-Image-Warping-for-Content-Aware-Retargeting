#ifndef RETARGETING_SEGMENTATION_H
#define RETARGETING_SEGMENTATION_H

#include <opencv2/core.hpp>

#include "retargeting/config.h"
#include "retargeting/types.h"

namespace retargeting {

// Splits a BGR image into patches: graph-based segmentation followed by the
// paper's merge of tiny and similarly colored neighboring patches. Fills each
// patch's id, size and segment_color.
Segmentation segment_image(const cv::Mat &source, const SegmentationParams &params);

}  // namespace retargeting

#endif  // RETARGETING_SEGMENTATION_H
