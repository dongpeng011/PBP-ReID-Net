# encoding: utf-8
"""Create tiny synthetic datasets so that the full pipeline
(data loading -> training -> evaluation) can be exercised without downloading
the real benchmarks.

    # Market-1501 style tree
    python tools/make_dummy_dataset.py --layout market --root data/dummy_market --ids 12

    # MSMT17 style tree (train/ + test/ + list_*.txt)
    python tools/make_dummy_dataset.py --layout msmt --root data/dummy_msmt --ids 10

    # two datasets with different scene ids -> multi-scene regime
    python train.py --config_file configs/PBP-ReID-multiscene.yml \
        DATASETS.COMBINE_NAMES "['market1501','msmt17']" \
        DATASETS.ROOTS "['data/dummy_market','data/dummy_msmt']" \
        DATASETS.COMBINE_TYPE "[1,2]" ...

Each identity is painted with a distinct colour and carries a text label, so the
model receives real gradients.  The synthetic set is trivially separable: it is
a plumbing test, not a benchmark.
"""

import argparse
import os
import random

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description='generate a dummy ReID dataset')
    parser.add_argument('--layout', choices=['market', 'msmt'], default='market')
    parser.add_argument('--root', default=None)
    parser.add_argument('--ids', type=int, default=20, help='number of identities')
    parser.add_argument('--cameras', type=int, default=3, help='number of cameras')
    parser.add_argument('--train', type=int, default=4, help='train images per identity')
    parser.add_argument('--query', type=int, default=1, help='query images per identity')
    parser.add_argument('--gallery', type=int, default=2, help='gallery images per identity')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def identity_palette(pid):
    """Colours of the eight rectangles that identify one person."""
    random.seed(1000 + pid)
    return [tuple(random.randint(20, 235) for _ in range(3)) for _ in range(8)]


def make_image(path, pid, cam, width, height, variant=0):
    """Paint a pedestrian-like crop: a grey background plus a per-identity
    pattern of coloured rectangles.  The identity is *not* written as text, so
    the network has to learn the pattern rather than read a label."""
    image = Image.new('RGB', (width, height), (196, 196, 196))
    draw = ImageDraw.Draw(image)
    palette = identity_palette(pid)

    # fixed layout per identity, small jitter per image and camera
    random.seed(pid * 131 + variant * 7)
    for index, colour in enumerate(palette):
        x0 = 10 + (index % 2) * (width // 2 - 8) + random.randint(-6, 6)
        y0 = 20 + index * (height - 60) // len(palette) + random.randint(-5, 5)
        x1 = x0 + random.randint(width // 5, width // 3)
        y1 = y0 + random.randint(14, 30)
        shade = tuple(max(0, min(255, channel + random.randint(-25, 25))) for channel in colour)
        draw.rectangle([max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)],
                       fill=shade)

    # camera marker only (no identity text)
    draw.rectangle([0, height - 18, width, height], fill=(60, 60, 60))
    draw.text((4, height - 16), 'CAM{}'.format(cam + 1), fill=(255, 255, 255))
    image.save(path, quality=90)


def build_market(args, root):
    splits = {'bounding_box_train': args.train, 'query': args.query,
              'bounding_box_test': args.gallery}
    for split in splits:
        os.makedirs(os.path.join(root, split), exist_ok=True)
    counter = 1
    for pid in range(1, args.ids + 1):
        for split, count in splits.items():
            for index in range(count):
                cam = index % args.cameras
                name = '{:04d}_c{}s1_{:06d}_{:02d}.jpg'.format(pid, cam + 1, counter, index)
                counter += 1
                make_image(os.path.join(root, split, name), pid, cam,
                           args.width, args.height, variant=index)


def build_msmt(args, root):
    """MSMT17 layout: train/<name>.jpg + list_train.txt, test/ + list_query/gallery."""
    for split in ('train', 'test'):
        os.makedirs(os.path.join(root, split), exist_ok=True)

    def write(split, count, list_name, start_pid=0):
        records = []
        for pid in range(start_pid, start_pid + args.ids):
            for index in range(count):
                cam = index % args.cameras
                # img_path.split('_')[2] must be the camera id (see datasets/msmt17.py)
                name = '{:04d}_{:02d}_{:02d}_{:06d}.jpg'.format(pid, 0, cam, index)
                make_image(os.path.join(root, split, name), pid, cam,
                           args.width, args.height, variant=index)
                records.append('{} {}'.format(name, pid - start_pid))
        with open(os.path.join(root, list_name), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(records) + '\n')

    write('train', args.train, 'list_train.txt')
    # an empty validation list is accepted by the dataset class
    open(os.path.join(root, 'list_val.txt'), 'w', encoding='utf-8').close()
    write('test', args.query, 'list_query.txt')
    write('test', args.gallery, 'list_gallery.txt')


def main():
    args = parse_args()
    random.seed(args.seed)
    root = args.root or os.path.join('data', 'dummy_' + args.layout)
    if args.layout == 'market':
        build_market(args, root)
    else:
        build_msmt(args, root)
    images = sum(len(files) for _, _, files in os.walk(root))
    print('dummy {} dataset written to {} ({} identities, {} files)'.format(
        args.layout, root, args.ids, images))


if __name__ == '__main__':
    main()
