#include "retargeting/warping.h"

#include "retargeting/mesh.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ilcplex/ilocplex.h>

namespace retargeting {
namespace {

// Mesh edges touching each patch. An edge whose endpoints fall in two
// different patches belongs to both.
std::vector<std::vector<unsigned int>> edges_per_patch(const Mesh &mesh, const Segmentation &segmentation)
{
    std::vector<std::vector<unsigned int>> result(segmentation.patches.size());
    for (unsigned int edge_index = 0; edge_index < mesh.edges.size(); edge_index++) {
        const cv::Vec2f &v1 = mesh.vertices[mesh.edges[edge_index].from];
        const cv::Vec2f &v2 = mesh.vertices[mesh.edges[edge_index].to];
        int patch1 = segmentation.labels.at<int>((int) v1[1], (int) v1[0]);
        int patch2 = segmentation.labels.at<int>((int) v2[1], (int) v2[0]);

        result[patch1].push_back(edge_index);
        if (patch2 != patch1) result[patch2].push_back(edge_index);
    }
    return result;
}

// Paper Sec. III-C: "The edge closest to the center of the patch is selected
// as the representative edge." Distance is measured from the patch centroid
// to the edge's midpoint.
std::vector<unsigned int> center_edges(const Mesh &mesh, const Segmentation &segmentation,
                                       const std::vector<std::vector<unsigned int>> &patch_edges)
{
    const cv::Mat &labels = segmentation.labels;
    std::vector<cv::Vec2d> sum(segmentation.patches.size(), cv::Vec2d(0, 0));
    for (int y = 0; y < labels.rows; y++) {
        const int *label = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; x++) sum[label[x]] += cv::Vec2d(x, y);
    }

    std::vector<unsigned int> result(patch_edges.size(), 0);
    for (size_t p = 0; p < patch_edges.size(); p++) {
        if (patch_edges[p].empty()) continue;
        const double size = std::max(1u, segmentation.patches[p].size);
        const cv::Vec2d centroid(sum[p][0] / size, sum[p][1] / size);
        double best = -1;
        for (unsigned int edge_index : patch_edges[p]) {
            const Edge &edge = mesh.edges[edge_index];
            const cv::Vec2f mid = (mesh.vertices[edge.from] + mesh.vertices[edge.to]) * 0.5f;
            const double distance = cv::norm(cv::Vec2d(mid[0], mid[1]) - centroid);
            if (best < 0 || distance < best) {
                best = distance;
                result[p] = edge_index;
            }
        }
    }
    return result;
}

}  // namespace

std::vector<cv::Vec2f> solve_warp(const Mesh &mesh, const Segmentation &segmentation,
                                  cv::Size target_size, const WarpParams &params,
                                  const std::vector<LineSegment> &lines)
{
    if (target_size.width <= 0 || target_size.height <= 0) {
        throw std::invalid_argument("Wrong target image size");
    }
    const unsigned int target_width = target_size.width;
    const unsigned int target_height = target_size.height;
    const cv::Mat &labels = segmentation.labels;

    IloEnv env;
    IloNumVarArray vp(env);  // target (x, y) of every mesh vertex, interleaved
    for (unsigned int i = 0; i < mesh.vertices.size(); i++) {
        vp.add(IloNumVar(env, -IloInfinity, IloInfinity));  // x
        vp.add(IloNumVar(env, -IloInfinity, IloInfinity));  // y
    }
    auto dx = [&](const Edge &e) { return vp[e.from * 2] - vp[e.to * 2]; };
    auto dy = [&](const Edge &e) { return vp[e.from * 2 + 1] - vp[e.to * 2 + 1]; };

    IloExpr energy(env);

    // Patch transformation constraints DST + DLT: every edge of a patch
    // should follow the similarity transform of the patch's reference edge.
    const double width_ratio = (double) target_width / (labels.cols - 1);
    const double height_ratio = (double) target_height / (labels.rows - 1);
    const std::vector<std::vector<unsigned int>> patch_edges = edges_per_patch(mesh, segmentation);
    const std::vector<unsigned int> centers =
        params.center_representative_edge ? center_edges(mesh, segmentation, patch_edges) : std::vector<unsigned int>();

    for (unsigned int patch_index = 0; patch_index < patch_edges.size(); patch_index++) {
        const std::vector<unsigned int> &edge_list = patch_edges[patch_index];
        if (edge_list.empty()) continue;

        const double saliency = segmentation.patches[patch_index].saliency_value;
        const double patch_size_weight = params.patch_size_weight ? std::sqrt(1.0 / (double) edge_list.size()) : 1.0;

        // Express each edge e in the basis of the reference edge c:
        // e = t_s * c + t_r * perp(c), by inverting [[c_x, c_y], [c_y, -c_x]].
        const Edge &center = mesh.edges[params.center_representative_edge ? centers[patch_index] : edge_list[0]];
        const cv::Vec2f c = mesh.vertices[center.from] - mesh.vertices[center.to];
        const double c_x = c[0];
        const double c_y = c[1];

        double det = c_x * -c_x - c_y * c_y;
        if (std::fabs(det) <= 1e-9) det = (det > 0 ? 1 : -1) * 1e-9;
        const double inv_a = -c_x / det;
        const double inv_b = -c_y / det;
        const double inv_c = -c_y / det;
        const double inv_d = c_x / det;

        for (unsigned int edge_index : edge_list) {
            const Edge &edge = mesh.edges[edge_index];
            const cv::Vec2f e = mesh.vertices[edge.from] - mesh.vertices[edge.to];
            const double e_x = e[0];
            const double e_y = e[1];

            const double t_s = inv_a * e_x + inv_b * e_y;
            const double t_r = inv_c * e_x + inv_d * e_y;

            // DST: salient patches keep their shape (similarity transform).
            energy += params.dst_weight * patch_size_weight * params.alpha * saliency *
                (IloPower(dx(edge) - (t_s * dx(center) + t_r * dy(center)), 2) +
                 IloPower(dy(edge) - (-t_r * dx(center) + t_s * dy(center)), 2));

            // DLT: non-salient patches follow the global linear scaling.
            energy += params.dlt_weight * patch_size_weight * (1 - params.alpha) * (1 - saliency) *
                (IloPower(dx(edge) - width_ratio * (t_s * dx(center) + t_r * dy(center)), 2) +
                 IloPower(dy(edge) - height_ratio * (-t_r * dx(center) + t_s * dy(center)), 2));
        }
    }

    // Grid orientation constraint DOR: horizontal edges stay horizontal,
    // vertical edges stay vertical.
    for (const Edge &edge : mesh.edges) {
        const cv::Vec2f delta = mesh.vertices[edge.from] - mesh.vertices[edge.to];
        if (std::abs(delta[0]) > std::abs(delta[1])) {
            energy += params.orientation_weight * IloPower(dy(edge), 2);
        } else {
            energy += params.orientation_weight * IloPower(dx(edge), 2);
        }
    }

    // Line preservation (extension, not in the paper): every detected line is
    // sampled, each sample expressed bilinearly in its quad, and each piece
    // between consecutive samples must keep the line's linearly scaled
    // direction. Only the direction is fixed, so the line can still be
    // compressed non-uniformly along its length.
    if (params.line_weight > 0) {
        const float spacing = std::min(mesh.cell_width, mesh.cell_height) / 2;
        auto position = [&](cv::Point2f point, int axis, double scale, IloExpr &expr) {
            const MeshCoordinate at = locate(mesh, point);
            const Quad &quad = mesh.quads[at.quad];
            for (int k = 0; k < 4; k++) expr += scale * at.weights[k] * vp[quad[k] * 2 + axis];
        };
        for (const LineSegment &line : lines) {
            const cv::Point2d scaled(width_ratio * (line.b.x - line.a.x), height_ratio * (line.b.y - line.a.y));
            const double norm = std::hypot(scaled.x, scaled.y);
            if (norm < 1e-6) continue;
            const cv::Point2d normal(-scaled.y / norm, scaled.x / norm);

            const std::vector<cv::Point2f> samples = sample_line(line, spacing);
            for (size_t i = 0; i + 1 < samples.size(); i++) {
                IloExpr across(env);  // normal . (p[i+1]' - p[i]')
                position(samples[i + 1], 0, normal.x, across);
                position(samples[i + 1], 1, normal.y, across);
                position(samples[i], 0, -normal.x, across);
                position(samples[i], 1, -normal.y, across);
                energy += params.line_weight * IloPower(across, 2);
                across.end();
            }
        }
    }

    IloModel model(env);
    model.add(IloMinimize(env, energy));

    // Boundary constraints: the mesh border is pinned to the target frame.
    IloRangeArray constraints(env);
    const cv::Vec2f &origin = mesh.vertices[0];
    for (unsigned int r = 0; r < mesh.rows; r++) {
        constraints.add(vp[mesh.index(r, 0) * 2] == origin[0]);
        constraints.add(vp[mesh.index(r, mesh.cols - 1) * 2] == origin[0] + target_width);
    }
    for (unsigned int c = 0; c < mesh.cols; c++) {
        constraints.add(vp[mesh.index(0, c) * 2 + 1] == origin[1]);
        constraints.add(vp[mesh.index(mesh.rows - 1, c) * 2 + 1] == origin[1] + target_height);
    }

    // Ordering constraints: vertices may not flip past their left / upper neighbor.
    for (unsigned int r = 0; params.prevent_foldover && r < mesh.rows; r++) {
        for (unsigned int c = 1; c < mesh.cols; c++) {
            constraints.add((vp[mesh.index(r, c) * 2] - vp[mesh.index(r, c - 1) * 2]) >= params.min_vertex_spacing);
        }
    }
    for (unsigned int r = 1; params.prevent_foldover && r < mesh.rows; r++) {
        for (unsigned int c = 0; c < mesh.cols; c++) {
            constraints.add((vp[mesh.index(r, c) * 2 + 1] - vp[mesh.index(r - 1, c) * 2 + 1]) >= params.min_vertex_spacing);
        }
    }
    model.add(constraints);

    IloCplex cplex(model);
    cplex.setOut(env.getNullStream());
    if (!cplex.solve()) {
        env.end();
        throw std::runtime_error("Failed to optimize");
    }

    IloNumArray solution(env);
    cplex.getValues(solution, vp);

    std::vector<cv::Vec2f> target_vertices(mesh.vertices.size());
    for (unsigned int i = 0; i < target_vertices.size(); i++) {
        target_vertices[i] = cv::Vec2f(solution[i * 2], solution[i * 2 + 1]);
    }

    env.end();
    return target_vertices;
}

}  // namespace retargeting
