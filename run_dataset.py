#!/usr/bin/env python3
"""Run the whole retargeting pipeline on every image of a dataset.

Usage (from the repository root, after `make`):
    env/bin/python run_dataset.py images-20100824.zip
    env/bin/python run_dataset.py images-20100824.zip --algo cas --out result/retargetme.zip
    env/bin/python run_dataset.py some_folder/ --target 400 300

For each image: compute the saliency map in Python, then run ./output in a
temporary working directory (so res/ and result/ of the repo are untouched).
Everything is collected into one zip:

    <name>/summary.csv   one row per image: sizes, patch counts, timings
    <name>/results/      retargeted images
    <name>/overviews/    pipeline figures (source -> ... -> result), JPEG
    <name>/saliency/     saliency maps
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'gbvs'))
from saliency_models import cas, gbvs, ittikochneibur  # noqa: E402

ALGORITHMS = {
    'gbvs': gbvs.compute_saliency,
    'ikn': ittikochneibur.compute_saliency,
    'cas': cas.compute_saliency,
}
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

# Lines printed by main.cpp's print_summary().
SUMMARY_PATTERNS = {
    'raw_segments': r'Patches\s*:\s*(\d+) raw',
    'patches': r'-> (\d+) after merging',
    'quads': r'vertices, (\d+) quads',
    'target': r'Target size\s*:\s*(\d+ x \d+)',
    'segmentation_s': r'segmentation\s+([\d.]+)',
    'significance_s': r'significance\s+([\d.]+)',
    'mesh_s': r'mesh\s+([\d.]+)',
    'warping_s': r'warping \(CPLEX\)\s+([\d.]+)',
    'rendering_s': r'rendering\s+([\d.]+)',
    'retarget_total_s': r'total\s+([\d.]+)',
}


def collect_images(source, workdir):
    """Returns sorted image paths from a zip file or a folder."""
    if zipfile.is_zipfile(source):
        folder = os.path.join(workdir, 'dataset')
        with zipfile.ZipFile(source) as archive:
            archive.extractall(folder)
    else:
        folder = source
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        paths += [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(IMAGE_EXTENSIONS)]
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def retarget_one(image_path, algo, target, outdir, workdir):
    name = os.path.splitext(os.path.basename(image_path))[0]
    row = {'image': name}
    image = cv2.imread(image_path)
    if image is None:
        row['status'] = 'failed to read image'
        return row
    row['width'], row['height'] = image.shape[1], image.shape[0]

    # Same layout ./output expects: res/gallery.jpg + res/gs.jpeg, writes result/.
    run_dir = os.path.join(workdir, 'run')
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(os.path.join(run_dir, 'res'))
    os.makedirs(os.path.join(run_dir, 'result'))

    start = time.time()
    saliency = ALGORITHMS[algo](image)
    row['saliency_s'] = round(time.time() - start, 3)
    # Written like pipeline.sh does it (demo.py saves JPEG).
    cv2.imwrite(os.path.join(run_dir, 'res', 'gs.jpeg'), saliency)
    cv2.imwrite(os.path.join(run_dir, 'res', 'gallery.jpg'), image)
    shutil.copy(os.path.join(run_dir, 'res', 'gs.jpeg'), os.path.join(outdir, 'saliency', name + '.jpg'))

    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(ROOT, 'opencv_local', 'lib'))
    env.pop('DISPLAY', None)  # never open result windows in batch mode
    command = [os.path.join(ROOT, 'output')] + ([str(target[0]), str(target[1])] if target else [])
    proc = subprocess.run(command, cwd=run_dir, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        row['status'] = 'retargeting failed: ' + (proc.stderr.strip().splitlines() or ['?'])[-1]
        return row

    for key, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, proc.stdout)
        row[key] = match.group(1) if match else ''
    shutil.copy(os.path.join(run_dir, 'result', 'result_gs.png'), os.path.join(outdir, 'results', name + '.png'))
    # Overviews are only for viewing: JPEG keeps the zip ~10x smaller.
    overview = cv2.imread(os.path.join(run_dir, 'result', 'pipeline_overview.png'))
    if overview is None:
        row['status'] = 'pipeline_overview.png missing'
        return row
    cv2.imwrite(os.path.join(outdir, 'overviews', name + '.jpg'), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    row['status'] = 'ok'
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help='zip file or folder of images')
    parser.add_argument('--algo', default='cas', choices=ALGORITHMS, help='saliency algorithm (default: cas)')
    parser.add_argument('--target', nargs=2, type=int, metavar=('W', 'H'),
                        help="target size for every image (default: ./output's own default, a square)")
    parser.add_argument('--out', help='output zip (default: result/<dataset name>_<algo>.zip)')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(ROOT, 'output')):
        sys.exit('./output not found - run `make` first')
    dataset_name = os.path.splitext(os.path.basename(os.path.normpath(args.dataset)))[0]
    out_zip = args.out or os.path.join(ROOT, 'result', '{}_{}.zip'.format(dataset_name, args.algo))
    folder_name = os.path.splitext(os.path.basename(out_zip))[0]

    with tempfile.TemporaryDirectory() as workdir:
        images = collect_images(args.dataset, workdir)
        if not images:
            sys.exit('no images found in ' + args.dataset)
        outdir = os.path.join(workdir, folder_name)
        for sub in ('results', 'overviews', 'saliency'):
            os.makedirs(os.path.join(outdir, sub))

        rows = []
        start = time.time()
        for i, path in enumerate(images, 1):
            row = retarget_one(path, args.algo, args.target, outdir, workdir)
            rows.append(row)
            print('[{:2d}/{}] {:<24} {}'.format(i, len(images), row['image'], row['status']), flush=True)

        columns = ['image', 'status', 'width', 'height', 'target', 'raw_segments', 'patches', 'quads',
                   'saliency_s', 'segmentation_s', 'significance_s', 'mesh_s', 'warping_s', 'rendering_s',
                   'retarget_total_s']
        with open(os.path.join(outdir, 'summary.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        os.makedirs(os.path.dirname(out_zip), exist_ok=True)
        shutil.make_archive(os.path.splitext(out_zip)[0], 'zip', workdir, folder_name)

    ok = sum(r['status'] == 'ok' for r in rows)
    print('{} / {} images succeeded in {:.0f} s -> {}'.format(ok, len(rows), time.time() - start, out_zip))


if __name__ == '__main__':
    main()
