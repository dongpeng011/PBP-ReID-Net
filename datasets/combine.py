# encoding: utf-8
"""Combine several ReID training sets into one multi-scene training set.

Each subset keeps its own train/query/gallery lists; the training lists are
merged into a single list of ``(img_path, pid, camid, viewid, scene_id)`` tuples
with a unified label space when ``combine_pid`` is True.

Scene ids come from ``DATASETS.COMBINE_TYPE`` (0 = low resolution, 1 = general,
2 = clothing change, 3 = occlusion, 4 = infrared).  They do **not** need to be
contiguous or to start at 0 -- the statistics are accumulated per scene id and
only the sorted list of ids is used for the per-scene bookkeeping.
"""

from .bases import BaseImageDataset
from .market1501 import Market1501
from .occluded_duke import OccDukeMTMCreID
from .prcc import PRCC
from .msmt17 import MSMT17
from .celebrity import Celebrity
from .sysu_mm01 import SysuMM01
from .dslr_cuhk03 import DSLR_CUHK03
from .con_reid import ConReID

_CombineDataset__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'prcc': PRCC,
    'occluded_duke': OccDukeMTMCreID,
    'celebrity': Celebrity,
    'sysumm01': SysuMM01,
    'dslr_cuhk03': DSLR_CUHK03,
    'con_reid': ConReID
}


class CombineDataset(BaseImageDataset):
    def __init__(self, names, roots, subset_type, combine_pid=True):
        super(CombineDataset, self).__init__()

        if len(names) != len(roots) or len(names) != len(subset_type):
            raise ValueError('COMBINE_NAMES, ROOTS and COMBINE_TYPE must have the same length '
                             '(got {}, {}, {})'.format(len(names), len(roots), len(subset_type)))

        self.train_dict, train_statistic, train_statistic_scene = {}, {}, {}
        self.total_classes, self.len_train_subset, self.classes_train_subset = 0, [], []
        self.query_dict, query_statistic = {}, {}
        self.len_query_list = []
        self.gallery_dict, gallery_statistic = {}, {}

        scene_ids = sorted(set(subset_type))
        pid_begin_per_scene = {scene_id: 0 for scene_id in scene_ids}
        pid_begin = 0

        for index, dataset_name in enumerate(names):
            print('Loading dataset {}, path {}'.format(dataset_name, roots[index]))
            scene_id = subset_type[index]
            start_pid = pid_begin if combine_pid else pid_begin_per_scene[scene_id]
            dataset = _CombineDataset__factory[dataset_name](root=roots[index],
                                                             pid_begin=start_pid,
                                                             verbose=False)

            self.train_dict[dataset_name] = dataset.train
            train_statistic[dataset_name] = self.get_imagedata_info(dataset.train)
            self.total_classes += train_statistic[dataset_name][0]

            # accumulate the statistics of this scene
            current = list(self.get_imagedata_info(dataset.train))
            if scene_id not in train_statistic_scene:
                train_statistic_scene[scene_id] = current
            else:
                for position in range(len(train_statistic_scene[scene_id])):
                    train_statistic_scene[scene_id][position] += current[position]

            self.query_dict[dataset_name] = dataset.query
            query_statistic[dataset_name] = self.get_imagedata_info(dataset.query)

            self.gallery_dict[dataset_name] = dataset.gallery
            gallery_statistic[dataset_name] = self.get_imagedata_info(dataset.gallery)

            if combine_pid:
                pid_begin += train_statistic[dataset_name][0]
            else:
                pid_begin_per_scene[scene_id] += train_statistic[dataset_name][0]

        self.train = []
        # reorganise the training data and attach the scene id of every image
        for index, (name, train_dataset) in enumerate(self.train_dict.items()):
            scene_id = subset_type[index]
            for img_path, pid, cam_id, view_id in train_dataset:
                self.train.append((img_path, pid, cam_id, view_id, scene_id))

        print('--------------------------------------------')
        print('The combine dataset contains:')
        print(names)
        print('--------------------------------------------')
        print('Training set contains:')
        print('subset name   | # ids | # images | # cameras')
        for name in names:
            print((name + (14 - len(name)) * ' ' + '| '), end='')
            print('{:5d} | {:8d} | {:9d}'.format(train_statistic[name][0],
                                                 train_statistic[name][1],
                                                 train_statistic[name][2]))

        for scene_id in scene_ids:
            self.classes_train_subset.append(train_statistic_scene[scene_id][0])
            self.len_train_subset.append(train_statistic_scene[scene_id][1])

        if combine_pid:
            total_train_statistic = self.get_dataset_info(self.train)
            print('--------------------------------------------')
            print('The total training set contains:')
            print('# ids | # images | # cameras  | # scene')
            print('{:5d} | {:8d} | Don\'t care | {:8d}'.format(
                total_train_statistic[0], total_train_statistic[1], total_train_statistic[3]))
        else:
            print('W/O combining pid, the training subsets are separate.')
            print('subset type   | # ids | # images | # cameras')
            for scene_id in scene_ids:
                print((str(scene_id) + 13 * ' ' + '| '), end='')
                print('{:5d} | {:8d} | Don\'t care'.format(
                    train_statistic_scene[scene_id][0], train_statistic_scene[scene_id][1]))
        print('--------------------------------------------')

        print('Query set contains:')
        print('subset        | # ids | # images | # cameras')
        for name in names:
            print((name + (14 - len(name)) * ' ' + '| '), end='')
            print('{:5d} | {:8d} | {:9d}'.format(query_statistic[name][0],
                                                 query_statistic[name][1],
                                                 query_statistic[name][2]))
            self.len_query_list.append(query_statistic[name][1])
        print('--------------------------------------------')

        print('Gallery set contains:')
        print('subset        | # ids | # images | # cameras')
        for name in names:
            print((name + (14 - len(name)) * ' ' + '| '), end='')
            print('{:5d} | {:8d} | {:9d}'.format(gallery_statistic[name][0],
                                                 gallery_statistic[name][1],
                                                 gallery_statistic[name][2]))
        print('--------------------------------------------')

    @staticmethod
    def get_dataset_info(data):
        pids, cams, views, dataset_ids = [], [], [], []

        for _, pid, camid, view_id, dataset_id in data:
            pids += [pid]
            cams += [camid]
            views += [view_id]
            dataset_ids += [dataset_id]
        return (len(set(pids)), len(data), len(set(cams)), len(set(dataset_ids)))
