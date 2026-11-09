# encoding: utf-8
import glob
import re
import os.path as osp
from datasets.bases import BaseImageDataset  # 保持与原代码相同的基类


class ConReID(BaseImageDataset):
    dataset_dir = 'con-reid'  # 修改为您的数据集名称

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(ConReID, self).__init__()
        self.dataset_dir = root  # 根目录由用户指定
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')  # 训练集目录
        self.query_dir = osp.join(self.dataset_dir, 'query')  # 查询集目录
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')  # 测试集目录

        self._check_before_run()
        self.pid_begin = pid_begin  # PID偏移量

        # 加载数据集
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> ConReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 保存数据信息
        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            gallery)

    def _check_before_run(self):
        """验证数据集目录是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        for dir_path in [self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(dir_path):
                raise RuntimeError(f"'{dir_path}' is not available")

    def _process_dir(self, dir_path, relabel=False):
        """处理图像目录，返回数据列表"""
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # 假设图片格式为JPG
        pattern = re.compile(r'([-\d]+)_c(\d)')  # 匹配文件名格式（与Market1501一致）

        # 收集所有行人ID
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # 忽略junk图像
            pid_container.add(pid)

        # 重新分配标签（仅训练集需要）
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # 忽略junk图像

            # 确保ID和摄像头编号在合理范围内（根据您的数据集调整）
            assert 0 <= pid <= 1501, "Invalid PID"
            assert 1 <= camid <= 6, "Invalid camera ID"

            camid -= 1  # 转换为从0开始的索引
            if relabel:
                pid = pid2label[pid]  # 重新分配标签

            # 添加到数据列表（注意保持与原代码相同的元组格式）
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset


if __name__ == '__main__':
    # 示例用法
    data = ConReID(root='/root/autodl-tmp/data/con-reid')
    # print(f"训练集人数: {data.num_train_pids}")
    # print(f"查询集人数: {data.num_query_pids}")
    # print(f"测试集人数: {data.num_gallery_pids}")