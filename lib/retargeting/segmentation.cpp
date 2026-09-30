#include "retargeting/segmentation.h"

#include <cmath>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include <opencv2/ximgproc/segmentation.hpp>

namespace retargeting {
namespace {

// Union-find over raw segment ids that also tracks the size and color sum
// of every merged set, so the mean color of a set is always available.
class PatchUnionFind {
public:
    PatchUnionFind(std::vector<unsigned int> sizes, std::vector<cv::Vec3d> color_sums)
        : parent_(sizes.size()), size_(std::move(sizes)), color_sum_(std::move(color_sums))
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x)
    {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size_[a] < size_[b]) std::swap(a, b);
        parent_[b] = a;
        size_[a] += size_[b];
        for (int c = 0; c < 3; c++) color_sum_[a][c] += color_sum_[b][c];
    }

    unsigned int size(int root) const { return size_[root]; }

    cv::Scalar mean_color(int root) const
    {
        return cv::Scalar(color_sum_[root][0] / size_[root],
                          color_sum_[root][1] / size_[root],
                          color_sum_[root][2] / size_[root]);
    }

private:
    std::vector<int> parent_;
    std::vector<unsigned int> size_;
    std::vector<cv::Vec3d> color_sum_;
};

double color_distance(const cv::Scalar &c1, const cv::Scalar &c2)
{
    double dr = c1[0] - c2[0], dg = c1[1] - c2[1], db = c1[2] - c2[2];
    return std::sqrt(dr * dr + dg * dg + db * db);
}

// Neighboring segment ids of every segment (4-connectivity).
std::vector<std::set<int>> build_adjacency(const cv::Mat &labels, int segment_count)
{
    std::vector<std::set<int>> adjacency(segment_count);
    for (int i = 0; i < labels.rows; i++) {
        const int *row = labels.ptr<int>(i);
        const int *row_below = (i + 1 < labels.rows) ? labels.ptr<int>(i + 1) : nullptr;
        for (int j = 0; j < labels.cols; j++) {
            int id = row[j];
            if (j + 1 < labels.cols && row[j + 1] != id) {
                adjacency[id].insert(row[j + 1]);
                adjacency[row[j + 1]].insert(id);
            }
            if (row_below && row_below[j] != id) {
                adjacency[id].insert(row_below[j]);
                adjacency[row_below[j]].insert(id);
            }
        }
    }
    return adjacency;
}

// Merges raw segments as in the paper's post-processing:
// (1) patches smaller than min_area_ratio of the image get absorbed into
//     their closest-color neighbor - regular graph segmentation always
//     leaves slivers that are noise, not real objects;
// (2) any adjacent pair of patches whose average color is within
//     color_merge_threshold gets merged too (same surface, just lightly
//     split by the segmentation threshold).
// Relabels `labels` in place with contiguous ids and returns the patch count.
int merge_segments(const cv::Mat &source, cv::Mat &labels, int segment_count,
                   const SegmentationParams &params)
{
    std::vector<unsigned int> sizes(segment_count, 0);
    std::vector<cv::Vec3d> color_sums(segment_count, cv::Vec3d(0, 0, 0));
    for (int i = 0; i < labels.rows; i++) {
        const int *label = labels.ptr<int>(i);
        const uchar *pixel = source.ptr<uchar>(i);
        for (int j = 0; j < labels.cols; j++) {
            sizes[label[j]] += 1;
            for (int c = 0; c < 3; c++) color_sums[label[j]][c] += pixel[j * 3 + c];
        }
    }

    const std::vector<std::set<int>> adjacency = build_adjacency(labels, segment_count);
    PatchUnionFind sets(std::move(sizes), std::move(color_sums));

    // Pass 1: small patches -> their closest-color neighbor.
    const double area_threshold = params.min_area_ratio * (double) (labels.rows * labels.cols);
    for (int i = 0; i < segment_count; i++) {
        int root = sets.find(i);
        if (sets.size(root) >= area_threshold) continue;

        double best_distance = -1;
        int best_neighbor = -1;
        for (int neighbor : adjacency[i]) {
            int neighbor_root = sets.find(neighbor);
            if (neighbor_root == root) continue;
            double distance = color_distance(sets.mean_color(root), sets.mean_color(neighbor_root));
            if (best_distance < 0 || distance < best_distance) {
                best_distance = distance;
                best_neighbor = neighbor_root;
            }
        }
        if (best_neighbor >= 0) sets.unite(root, best_neighbor);
    }

    // Pass 2: any adjacent pair with close average color.
    for (int i = 0; i < segment_count; i++) {
        for (int neighbor : adjacency[i]) {
            if (neighbor <= i) continue;  // visit each raw pair once
            int root_a = sets.find(i);
            int root_b = sets.find(neighbor);
            if (root_a == root_b) continue;
            if (color_distance(sets.mean_color(root_a), sets.mean_color(root_b)) < params.color_merge_threshold) {
                sets.unite(root_a, root_b);
            }
        }
    }

    // Compact the surviving roots into a contiguous id range.
    std::vector<int> new_id(segment_count, -1);
    int next_id = 0;
    for (int i = 0; i < labels.rows; i++) {
        int *label = labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; j++) {
            int root = sets.find(label[j]);
            if (new_id[root] < 0) new_id[root] = next_id++;
            label[j] = new_id[root];
        }
    }
    return next_id;
}

}  // namespace

Segmentation segment_image(const cv::Mat &source, const SegmentationParams &params)
{
    Segmentation result;

    cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> segmentator =
        cv::ximgproc::segmentation::createGraphSegmentation(params.sigma, params.k, params.min_size);
    segmentator->processImage(source, result.labels);

    double max_label;
    cv::minMaxLoc(result.labels, nullptr, &max_label);
    result.raw_segment_count = (int) max_label + 1;

    int patch_count = params.merge_patches
        ? merge_segments(source, result.labels, result.raw_segment_count, params)
        : result.raw_segment_count;

    std::vector<Patch> &patches = result.patches;
    patches.resize(patch_count);
    for (int i = 0; i < patch_count; i++) patches[i].id = i;

    for (int i = 0; i < result.labels.rows; i++) {
        const int *label = result.labels.ptr<int>(i);
        for (int j = 0; j < result.labels.cols; j++) patches[label[j]].size += 1;
    }

    // Mean color, accumulated as pixel / size to keep the original rounding.
    for (int i = 0; i < source.rows; i++) {
        const uchar *pixel = source.ptr<uchar>(i);
        const int *label = result.labels.ptr<int>(i);
        for (int j = 0; j < source.cols; j++) {
            Patch &patch = patches[label[j]];
            for (int c = 0; c < 3; c++) patch.segment_color[c] += pixel[j * 3 + c] / (double) patch.size;
        }
    }

    return result;
}

}  // namespace retargeting
