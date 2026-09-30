// Tunable parameters of each pipeline stage, gathered in one place.
#ifndef RETARGETING_CONFIG_H
#define RETARGETING_CONFIG_H

namespace retargeting {

struct SegmentationParams {
    // Felzenszwalb & Huttenlocher graph segmentation.
    double sigma = 1.5;
    float k = 200;
    int min_size = 50;

    // Post-processing from the paper. min_size above is a fixed pixel count,
    // not the paper's area-relative threshold, so large images (e.g. a
    // 2560x1600 photo) still come out of it badly over-segmented - the merge
    // pass is what actually enforces the "0.01% of image area" rule.
    double min_area_ratio = 0.0001;        // patches smaller than this are absorbed
    double color_merge_threshold = 20.0;   // RGB distance below which neighbors merge
    bool merge_patches = true;             // false keeps the raw over-segmentation
};

struct MeshParams {
    float grid_size = 20.0f;  // approximate cell size in pixels
};

// Energy of the warping optimization. The defaults follow the paper:
// eq. (4) weights D_ST and D_LT only by alpha and (1 - alpha), and eq. (6)
// is the plain sum D_TF + D_OR. tuned() gives our earlier hand-tuned setup.
struct WarpParams {
    double alpha = 0.8;  // paper: "a large value is assigned to this parameter (alpha = 0.8)"

    double dst_weight = 1.0;          // patch similarity transformation (DST)
    double dlt_weight = 1.0;          // patch linear scaling (DLT)
    double orientation_weight = 1.0;  // grid orientation (DOR)

    bool patch_size_weight = false;          // extra sqrt(1 / #edges) weight per patch (not in the paper)
    bool center_representative_edge = true;  // paper: "the edge closest to the center of the patch"
    bool prevent_foldover = false;           // min_vertex_spacing constraints (not in the paper)
    double min_vertex_spacing = 1e-4;        // used when prevent_foldover is on

    // Extension beyond the paper: keeps detected straight lines straight by
    // fixing each line's direction to its linearly scaled direction. 0 = off.
    double line_weight = 0.0;
    float line_min_length = 40.0f;  // lines shorter than ~2 quads cannot bend

    // Our earlier hand-tuned setup, favoring rigid salient content. DOR was
    // raised from 12 because on wide frames with tall vertical structures
    // (e.g. a skyscraper next to open sky) the grid columns bowed.
    static WarpParams tuned()
    {
        WarpParams p;
        p.alpha = 0.8f;
        p.dst_weight = 5.5f;
        p.dlt_weight = 0.5f;
        p.orientation_weight = 24.0f;
        p.patch_size_weight = true;
        p.center_representative_edge = false;
        p.prevent_foldover = true;
        return p;
    }
};

}  // namespace retargeting

#endif  // RETARGETING_CONFIG_H
