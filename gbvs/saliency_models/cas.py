"""Context-Aware Saliency Detection (Goferman, Zelnik-Manor & Tal, CVPR 2010).

This is the algorithm the original retargeting paper actually specifies,
distinct from the GBVS implementation in gbvs.py. Core idea: a pixel is
salient if its local patch looks unlike most other patches in the image
(appearance distinctiveness), with three refinements: (1) multi-scale
comparison so both fine detail and coarse structure count, (2) an
"immediate context" pass that spreads saliency from the most distinctive
points into the whole object/region they belong to, and (3) a hard face
prior. Parameters below (K=64, patch 7x7 at 50% overlap, working size 250,
scales {100%,80%,50%,30%}, attended threshold 0.8, c=3) follow the paper.
"""
import os
import cv2
import numpy as np
from scipy.spatial import cKDTree

# cv2.CascadeClassifier (classic Haar cascades, what the paper uses via
# Viola-Jones) was removed from the objdetect Python bindings in OpenCV 5.x,
# so face detection here uses the DNN-based YuNet detector instead, via a
# model file from the official OpenCV Zoo - same role, different detector.
_FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'face_detection_yunet_2023mar.onnx')


def _extract_patches(image, patch_size):
    """One flattened (patch_size*patch_size*channels) patch per pixel, edge-padded."""
    pad = patch_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w, c = image.shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size, c))
    return windows.reshape(h, w, patch_size * patch_size * c)


def _single_scale_saliency(lab_image, patch_size, k, c):
    h, w = lab_image.shape[:2]
    dense = _extract_patches(lab_image, patch_size)

    # Paper: patches sampled at 50% overlap, not one per pixel. Sparser
    # sampling also makes exhaustive K-NN (below) tractable without an
    # approximate/random candidate pool.
    stride = max(1, patch_size // 2)
    sparse = dense[::stride, ::stride]
    sparse_h, sparse_w = sparse.shape[:2]
    patches = sparse.reshape(-1, sparse.shape[-1]).astype(np.float32) / 255.0

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    positions = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    norm_dim = float(max(h, w))

    # Exhaustive pairwise distance via the a^2+b^2-2ab expansion, computed as
    # one matmul instead of an (N, N, patch_dim) tensor.
    color_dist = (
        np.sum(patches ** 2, axis=1, keepdims=True)
        + np.sum(patches ** 2, axis=1)[None, :]
        - 2.0 * patches @ patches.T
    )
    np.maximum(color_dist, 0, out=color_dist)
    np.sqrt(color_dist, out=color_dist)

    pos_dist = (
        np.sum(positions ** 2, axis=1, keepdims=True)
        + np.sum(positions ** 2, axis=1)[None, :]
        - 2.0 * positions @ positions.T
    )
    np.maximum(pos_dist, 0, out=pos_dist)
    np.sqrt(pos_dist, out=pos_dist)
    pos_dist /= norm_dim

    # Goferman et al. eq. (1): appearance distance discounted by spatial
    # proximity, so two similar patches that are also near each other barely
    # count as "explaining" one another away - that's ordinary local texture,
    # not evidence the pixel is common/background.
    dissimilarity = color_dist / (1.0 + c * pos_dist)
    np.fill_diagonal(dissimilarity, np.inf)  # a patch is never its own neighbor

    k = min(k, patches.shape[0] - 1)
    nearest = np.partition(dissimilarity, k - 1, axis=1)[:, :k]
    saliency = 1.0 - np.exp(-nearest.mean(axis=1))
    saliency = saliency.reshape(sparse_h, sparse_w)

    # Patches were sampled sparsely; interpolate back up to per-pixel.
    return cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_context(saliency, attended_threshold, iterations):
    h, w = saliency.shape
    ys, xs = np.mgrid[0:h, 0:w]
    positions = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    norm_dim = float(max(h, w))

    # Goferman et al. eq. (5): pixels above a fixed absolute threshold are the
    # "attended" foci; everything else gets pulled down by its distance to
    # the nearest one - this simulates attention concentrating near what's
    # already been identified as a focus, not just wherever colors differ.
    # The paper applies this once; a single pass with the paper's own
    # threshold (0.8) only lights up the very peak of a large uniform object
    # (a building facade, a face) instead of the whole thing, since nothing
    # nearby is above that bar yet. Iterating re-seeds from the pass before,
    # letting a strong peak progressively pull its whole surrounding surface
    # up with it - a deliberate, evidence-based deviation from the paper's
    # single-pass description, kept because it demonstrably fills in large
    # objects instead of leaving them dark except at their most extreme point.
    current = saliency
    for _ in range(iterations):
        attended_mask = (current > attended_threshold).ravel()
        if not np.any(attended_mask):
            break

        tree = cKDTree(positions[attended_mask])
        d_foci, _ = tree.query(positions)
        d_foci = d_foci.reshape(h, w) / norm_dim

        current = current * (1.0 - d_foci)
        current = cv2.normalize(current, None, 0.0, 1.0, cv2.NORM_MINMAX)

    return current


def _boost_faces(saliency, full_res_image):
    """Goferman et al. principle 4 (high-level factors) / eq. (6): a hard
    binary face mask, max'd in - a face is fully salient regardless of how
    low-level-distinctive it looks, no partial credit for being near one."""
    if not os.path.exists(_FACE_MODEL_PATH):
        return saliency

    img_h, img_w = full_res_image.shape[:2]
    detector = cv2.FaceDetectorYN_create(_FACE_MODEL_PATH, "", (img_w, img_h))
    _, faces = detector.detect(full_res_image)
    if faces is None or len(faces) == 0:
        return saliency

    h, w = saliency.shape
    face_mask = np.zeros_like(saliency)
    for face in faces:
        x, y, fw, fh = face[:4]
        x0 = max(0, int(round(x * w / img_w)))
        y0 = max(0, int(round(y * h / img_h)))
        x1 = min(w, int(round((x + fw) * w / img_w)))
        y1 = min(h, int(round((y + fh) * h / img_h)))
        face_mask[y0:y1, x0:x1] = 1.0

    return np.maximum(saliency, face_mask)


def run(image, params):
    h0, w0 = image.shape[:2]
    scale_factor = params['working_size'] / float(max(h0, w0))
    work_h = max(1, int(round(h0 * scale_factor)))
    work_w = max(1, int(round(w0 * scale_factor)))
    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)

    scale_maps = []
    for r in params['scales']:
        rh = max(params['patch_size'], int(round(work_h * r)))
        rw = max(params['patch_size'], int(round(work_w * r)))
        scaled = cv2.resize(lab, (rw, rh), interpolation=cv2.INTER_AREA)
        sal = _single_scale_saliency(scaled, params['patch_size'], params['k'], params['c'])
        scale_maps.append(cv2.resize(sal, (work_w, work_h), interpolation=cv2.INTER_LINEAR))

    saliency = np.mean(scale_maps, axis=0)
    saliency = cv2.normalize(saliency, None, 0.0, 1.0, cv2.NORM_MINMAX)
    saliency = _apply_context(saliency, params['attended_threshold'], params['context_iterations'])

    if params['face_boost']:
        saliency = _boost_faces(saliency, image)

    return cv2.resize(saliency, (w0, h0), interpolation=cv2.INTER_LINEAR)


def setupParams():
    return {
        'working_size': 250,             # paper: images scaled to a maximum dimension of 250px
        'scales': [1.0, 0.8, 0.5, 0.3],  # paper: R = {100%, 80%, 50%, 30%}
        'patch_size': 7,                 # paper: 7x7 patches
        'k': 64,                         # paper: K = 64 nearest-neighbor patches
        'c': 3.0,                        # paper: position-distance weight in eq. (1)
        'attended_threshold': 0.8,       # paper: 0.8
        'context_iterations': 1,         # paper applies eq. (5) once. Iterating sounded like it
                                          # should spread coverage further each pass, but tested
                                          # worse in practice: each pass renormalizes, so it
                                          # compounds into a MORE polarized map (thin bright rim,
                                          # everything else crushed toward zero), not a fuller one.
        'face_boost': True,              # paper eq. (6): max(S, face_binary_map)
    }


def compute_saliency(input_image):
    if type(input_image) is str:
        input_image = cv2.imread(input_image)

    params = setupParams()
    saliency = run(image=input_image, params=params)
    return saliency * 255.0
