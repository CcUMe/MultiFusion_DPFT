from __future__ import annotations  # noqa: F407

import os
import os.path as osp

from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

import torch
# dataset.py 顶部
from dprt.datasets.kradar.utils.lidar_bev_config import (
    BEV_X_RANGE as L_XR, BEV_Y_RANGE as L_YR,
    BEV_Z_RANGE as L_ZR, BEV_WIDTH as L_W, BEV_HEIGHT as L_H
)
from dprt.datasets.kradar.utils.radar_bev_config import (
    BEV_X_RANGE as R_XR, BEV_Y_RANGE as R_YR,
    BEV_Z_RANGE as R_ZR, BEV_WIDTH as R_W, BEV_HEIGHT as R_H
)
from dprt.datasets.kradar.utils import radar_info, lidar_bev_config
from dprt.datasets.weak_labels import convert_raw_labels


class KRadarDataset(Dataset):
    def __init__(self,
                 src: str,
                 version: str = '',
                 split: str = 'train',
                 camera: str = 'M',
                 camera_dropout: float = 0.0,
                 image_size: Union[int, Tuple[int, int]] = None,
                 radar: str = 'BF',
                 radar_dropout: float = 0.0,
                 lidar: int = 1,
                 lidar_dropout: float = 0.0,
                 label: str = 'detection',
                 num_classes: int = 10,
                 sequential: bool = False,
                 scale: bool = True,
                 label_mode: str = 'strict_3d',
                 weak_label: Dict[str, Any] = None,
                 fov: Dict[str, Tuple[float, float]] = None,
                 dtype: str = 'float32',
                 **kwargs):
        """Dataset class for the K-Radar dataset.

        Arguments:
            src: Source path to the pre-processed
                dataset folder.
            version: Dataset version. One of either
                mini or None (full dataset).
            split: Dataset split to load. One of
                either train or test.
            camera: Camera modalities to use. One of
                either 'M' (mono camera), 'S' (stereo camera)
                or None.
            camera_dropout: Camera modality dropout probability
                between 0 and 1.
            image_size: Image size to resize image data to.
                Either None (no resizing), int or tuple of two
                int specifying the height and width.
            radar: Radar modalities to use. Any combination
                of 'B' (BEV) and 'F' (Front) or None
            radar_dropout: Radar modality dropout probability
                between 0 and 1.
            lidar: Lidar modality to use. One of either
                0 (no lidar), 1 (OS1) or 2 (OS2).
            label: Type of label data to use. One of either
                'detection' (3D bounding boxes), 'occupancy'
                (3D occupancy grid) or None.
            num_classes: Number of object classes used for
                one hot encoding.
            sequential: Whether to consume sequneces of
                samples or single samples.
            scale: Whether to scale the radar data to
                a value range of [0, 255] or not.
            fov: Field of view to limit the lables to. Can
                contain values for x, y, z and azimuth.
        """
        # Initialize parent dataset class
        super().__init__()

        # 修改dropout检查
        assert camera_dropout + radar_dropout + lidar_dropout <= 1.0

        # Initialize instance attributes
        self.src = src
        self.version = version
        self.split = split
        self.camera = camera
        self.camera_dropout = camera_dropout
        self.image_size = image_size
        self.radar = radar
        self.radar_dropout = radar_dropout
        self.lidar = lidar
        self.lidar_dropout = lidar_dropout
        self.label = label
        self.num_classes = num_classes
        self.sequential = sequential
        self.scale = scale
        self.label_mode = label_mode
        self.weak_label = weak_label if weak_label is not None else {}
        self.fov = fov if fov is not None else {}
        self.dtype = dtype

        # Adjust split according to dataset version
        if self.version:
            self.split = f"{self.version}_{self.split}"

        # Initialize moality dropout attributes
        # Define the lottery pot to draw from (None, camera, radar)
        self.lottery = [
            {},
            {'camera_mono', 'camera_stereo'},
            {'radar_bev', 'radar_front'},
            {'lidar_bev'}  # 新增
        ]
        # print("---------------")
        # print(self.lottery)

        # Define dropout probabilities (sum of probabilities must be <= 1)

        self.dropout = [
            1 - (self.camera_dropout + self.radar_dropout+self.lidar_dropout),
            self.camera_dropout,
            self.radar_dropout,
            self.lidar_dropout  # 新增
        ]

        # Get dataset path
        self.dataset_paths = self.get_dataset_paths(self.src)
        # 在 __init__ 方法中，get_dataset_paths 后添加：
        # if self.split == 'test' and not self.sequential:
        #     filtered_paths = []
        #     seen_descriptions = {}  # 优化：序列共享描述，避免重复加载
        #     for path_dict in self.dataset_paths:
        #         desc_path = path_dict['description']
        #         seq_key = osp.dirname(des
        #         c_path)  # 使用序列目录作为键
        #         if seq_key not in seen_descriptions:
        #             desc = np.load(desc_path)
        #             seen_descriptions[seq_key] = desc[2]  # 仅存 time_zone
        #         weather_id = seen_descriptions[seq_key]
        #         if weather_id == 0:  # 保留夜间，过滤白天 (time_zone=0)
        #             filtered_paths.append(path_dict)
        #     self.dataset_paths = filtered_paths
        #     print(
        #         f"测试集过滤后样本数: {len(self.dataset_paths)} (原: {len(self.dataset_paths) + (len(self.dataset_paths) - len(filtered_paths))})")  # 可选日志

    def __len__(self):
        return len(self.dataset_paths)

    # def __getitem__(self, index) -> Dict[str, torch.Tensor]:
    #     """Returns an item of the dataset given its index.
    #
    #     Whether or not the retured Tensors include a time
    #     dimension depends on whether or not sequential is
    #     true or false.
    #
    #     Arguments:
    #         index: Index of the dataset item to return.
    #
    #     Returns:
    #         item: Dataset item as dictionary of tensors.
    #     """
    #     # Map index to sequence number for sequential data
    #     if self.sequential:
    #         index = list(sorted(self.dataset_paths.keys()))[index]
    #
    #     # Get item from dataset
    #     item = self._to_list(self.dataset_paths[index])
    #
    #     # Load data from file paths
    #     for sample in item:
    #         sample = self.load_sample_data(sample)
    #
    #     # Scale radar data
    #     if self.scale:
    #         sample = self.scale_radar_data(sample)
    #
    #     # Apply modality dropout
    #     sample = self.modality_dropout(sample)
    #
    #     # Get task specific label
    #     if self.label == 'detection':
    #         label = self.get_detection_label(sample.pop('label'))
    #
    #     # Add description to label
    #     label.update({'description': sample.pop('description')})
    #
    #     # Set sensor data transformations (transformations in cartesian space)
    #     sample = self._add_transformations(sample)
    #
    #     # Set sensor data projections (projections in sensor space)
    #     sample = self._add_projections(sample)
    #
    #     # Set sensor data input shape
    #     sample = self._add_shape(sample)
    #
    #     # Resize image (if required)
    #     if self.image_size is not None:
    #         sample = self.resize_image(sample)
    #
    #     # Convert list of dicts to dict of stacked tensors
    #     if self.sequential:
    #         # Stack tensors along the time dimension (use padding for variable
    #         # size inputs, e.g. label)
    #         # item = {key: default_collate([d[key] for d in item]) for key in sample}
    #         raise NotImplementedError()
    #     else:
    #         # There is just a single sample for non sequential data
    #         item = sample
    #
    #     return item, label

    def __getitem__(self, index) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # 顺序/非顺序逻辑保留
        if self.sequential:
            index = list(sorted(self.dataset_paths.keys()))[index]

        # 取到该样本的所有文件路径 dict
        sample_paths = self.dataset_paths[index]

        # 1. Load raw data (包括 camera, radar, lidar_top, label, description)
        sample = self.load_sample_data(sample_paths)
        # print('radar_bev shape:', sample['radar_bev'].shape)
        # print('lidar_bev shape:', sample['lidar_bev'].shape)
        # print('radar_front shape:', sample['radar_front'].shape)

        # 2. Scale radar
        if self.scale:
            sample = self.scale_radar_data(sample)

        # 3. Modality dropout
        sample = self.modality_dropout(sample)

        # # 4. 处理 LiDAR 点云 -> BEV  保证形状一致
        # if self.lidar > 0 and 'lidar_top' in sample:
        #     # 4.1 过滤
        #     pts = sample.pop('lidar_top')
        #     filtered = self.filter_lidar_points(pts)
        #     # 4.2 投影成 BEV
        #     bev = self.points_to_bev(filtered)
        #     sample['lidar_bev'] = bev  # 现在一定是 [4,H,W]


        # 5. 生成 label
        if self.label == 'detection':
            label = self.get_detection_label(sample.pop('label'), sample)
            label.update({'description': sample.pop('description')})
        else:
            label = {}

        # 6. 其余变换／投影／shape 信息添加
        sample = self._add_transformations(sample)
        sample = self._add_projections(sample)

        # print("---->", sample.keys())

        sample = self._add_shape(sample)

        # 7. Resize 图像（若配置）
        if self.image_size is not None:
            sample = self.resize_image(sample)

        return sample, label

    @classmethod
    def from_config(cls, config: Dict, *args, **kwargs) -> KRadarDataset:  # noqa: F821
        return cls(*args, **dict(config['computing'] | config['data']), **kwargs)

    @staticmethod
    def _to_list(item: Any) -> List[Any]:
        if not isinstance(item, (list, tuple, set)):
            return [item]
        return item

    #把标签投影到各个传感器的坐标系
    def _add_transformations(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the transformation matirces to the sample.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        if 'M' in self.camera:
            sample['label_to_camera_mono_t'] = torch.zeros_like(sample['label_to_camera_mono'])
        if 'S' in self.camera:
            sample['label_to_camera_stereo_t'] = torch.zeros_like(sample['label_to_camera_stereo'])
        if 'B' in self.radar:
            sample['label_to_radar_bev_t'] = sample.pop('label_to_radar_bev')
        if 'F' in self.radar:
            sample['label_to_radar_front_t'] = sample.pop('label_to_radar_front')
        if self.lidar > 0 and 'lidar_bev' in sample:
            # LiDAR BEV使用单位矩阵作为变换矩阵（已经在笛卡尔坐标系中）
            sample['label_to_lidar_bev_t'] = torch.eye(4).type(getattr(torch, self.dtype))

        return sample

    # 添加从3D世界坐标到各传感器2D特征图像素的投影矩阵
    def _add_projections(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the projection matrices to the sample.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        if 'M' in self.camera:
            sample['label_to_camera_mono_p'] = sample.pop('label_to_camera_mono')
        if 'S' in self.camera:
            sample['label_to_camera_stereo_p'] = sample.pop('label_to_camera_stereo')
        if 'B' in self.radar:
            sample['label_to_radar_bev_p'] = self._get_radar_ra_projection()
        if 'F' in self.radar:
            sample['label_to_radar_front_p'] = self._get_radar_ea_projection()
        if self.lidar > 0 and 'lidar_bev' in sample:
            sample['label_to_lidar_bev_p'] = self._get_lidar_bev_projection()

        return sample

    def _add_shape(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the original input data shape to the sample.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        if 'M' in self.camera:
            sample['camera_mono_shape'] = torch.as_tensor(sample['camera_mono'].shape)
        if 'S' in self.camera:
            sample['camera_stereo_shape'] = torch.as_tensor(sample['camera_stereo'].shape)
        if 'B' in self.radar:
            sample['radar_bev_shape'] = torch.as_tensor(sample['radar_bev'].shape)
        if 'F' in self.radar:
            sample['radar_front_shape'] = torch.as_tensor(sample['radar_front'].shape)
        if self.lidar > 0 and 'lidar_bev' in sample:
            sample['lidar_bev_shape'] = torch.as_tensor(sample['lidar_bev'].shape)

        return sample

    def _get_radar_ea_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the elevation-azimuth projection.

        The projection matrix P is given that
        [u]
        [v] = P [r, phi, roh, 1]
        [1]

        with range (r), azimuth (phi) and elevation (roh) in spherical
        coordinates. So that, u and v represent the indices of the radar
        grid (raster).
        """
        return torch.Tensor([
            [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
            [0, 0, 1, (len(radar_info.elevation_raster) - 1) / 2],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def _get_radar_ra_projection(self) -> torch.Tensor:
        # Radar BEV Range-Azimuth
        scale_r = len(radar_info.range_raster) / max(radar_info.range_raster)
        center_az = (len(radar_info.azimuth_raster) - 1) / 2
        return torch.Tensor([
            [0, -1, 0, center_az],
            [scale_r, 0, 0, 0],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def scale_radar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scales radar and lidar data to a range of 0 to 255"""

        # 缩放雷达数据
        for k, v in sample.items():
            if k in {'radar_bev', 'radar_front'}:
                sample[k] = (v - radar_info.min_power) / \
                            (radar_info.max_power - radar_info.min_power) * 255
                sample[k] = torch.clip(sample[k], 0, 255)

        # 缩放LiDAR数据
        sample = self.scale_lidar_data(sample)

        return sample

    def resize_image(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Resizes the images in the sample.

        Resizes the images to a Tensor of shape (C, self.image_size[0], self.image_size[1])
        if image size if given as a tuple of interger values, otherwise it resizes it to a
        tensor with shape (C, self.image_size, self.image_size * width / height).

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier data tensors.
        """
        if 'M' in self.camera:
            sample['camera_mono'] = \
                resize(sample['camera_mono'].movedim(-1, 0), self.image_size).movedim(0, -1)
        if 'S' in self.camera:
            sample['camera_stereo'] = \
                resize(sample['camera_stereo'].movedim(-1, 0), self.image_size).movedim(0, -1)

        return sample

    def get_detection_label(self, raw_label: Dict[str, torch.Tensor],
                            sample: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Get detection task label data.

        Splits the K-Radar dataset label, given as bounding box of
        [x, y, z, theta, l, w, h, category index, object id], into
        its individual components.

        Arguments:
            raw_label: Collection of sample data.

        Returns:
            label: Modified collecton of sample data.
        """
        if self.label_mode in {'weak', 'weak_2d', 'weak_2d_bev', 'weak_center_lh'}:
            raw_label = self._filter_raw_label_fov(raw_label)
            sample = sample or {}
            projection = sample.get('label_to_camera_mono')
            if projection is not None:
                projection = projection.detach().cpu().numpy()

            image_shape = None
            if 'camera_mono' in sample:
                image_shape = tuple(sample['camera_mono'].shape[:2])

            weak_label = convert_raw_labels(
                raw_label.detach().cpu().numpy(),
                projection=projection,
                image_shape=image_shape,
                num_classes=self.num_classes,
                config=self.weak_label
            )

            label = {}
            for key, value in weak_label.items():
                tensor = torch.from_numpy(value)
                if value.dtype.kind == 'f':
                    tensor = tensor.type(getattr(torch, self.dtype))
                label[key] = tensor
            return label

        # Initialize label
        label = {}

        # Split label data into its components
        label['gt_center'] = raw_label[:, (0, 1, 2)]
        label['gt_size'] = raw_label[:, (4, 5, 6)]

        # Encode angle by its sin and cos part
        label['gt_angle'] = torch.cat(
            (torch.sin(raw_label[:, (3, )]), torch.cos(raw_label[:, (3, )])),
            dim=-1
        )


        # One hot encode class labels (+1 for ignore class)
        label['gt_class'] = torch.nn.functional.one_hot(
            raw_label[:, 7].long() + 1,
            self.num_classes
        ).type(getattr(torch, self.dtype))


        # Get configured field of view
        x_min, x_max = self.fov.get('x', torch.tensor([-float('inf'), float('inf')]))
        y_min, y_max = self.fov.get('y', torch.tensor([-float('inf'), float('inf')]))
        z_min, z_max = self.fov.get('z', torch.tensor([-float('inf'), float('inf')]))
        a_min, a_max = self.fov.get('azimuth', torch.tensor([-float('inf'), float('inf')]))

        # Get azimuth angle of the center points
        azimuth = torch.rad2deg(torch.arctan2(label['gt_center'][:, 1], label['gt_center'][:, 0]))

        # Limit lables to configured field of view (FoV)
        x_mask = (x_min < label['gt_center'][:, 0]) & (label['gt_center'][:, 0] < x_max)
        y_mask = (y_min < label['gt_center'][:, 1]) & (label['gt_center'][:, 1] < y_max)
        z_mask = (z_min < label['gt_center'][:, 2]) & (label['gt_center'][:, 2] < z_max)
        a_mask = (a_min < azimuth) & (azimuth < a_max)

        fov_mask = x_mask & y_mask & z_mask & a_mask

        # Mask lables according to the field of view
        label = {k: v[fov_mask] for k, v in label.items()}

        return label

    def _filter_raw_label_fov(self, raw_label: torch.Tensor) -> torch.Tensor:
        if raw_label.numel() == 0:
            return raw_label

        def bounds(name: str):
            value = self.fov.get(name, [-float('inf'), float('inf')])
            value = torch.as_tensor(value, dtype=raw_label.dtype, device=raw_label.device)
            return value[0], value[1]

        x_min, x_max = bounds('x')
        y_min, y_max = bounds('y')
        z_min, z_max = bounds('z')
        a_min, a_max = bounds('azimuth')

        center = raw_label[:, (0, 1, 2)]
        azimuth = torch.rad2deg(torch.arctan2(center[:, 1], center[:, 0]))
        fov_mask = (
            (x_min < center[:, 0]) & (center[:, 0] < x_max) &
            (y_min < center[:, 1]) & (center[:, 1] < y_max) &
            (z_min < center[:, 2]) & (center[:, 2] < z_max) &
            (a_min < azimuth) & (azimuth < a_max)
        )
        return raw_label[fov_mask]

    def get_sample_path(self, src: str) -> Dict[str, str]:
        """Returns all data paths of a given dataset sample.

        Arguments:
            src: Sourcce path to the data files
                of a single dataset sample.

        Returns:
            sample_batch: Dictionary mapping the sample
                items to filenames.
        """
        # Initialize sample paths
        sample_path = {}

        # Get sensor data and calibration information
        if 'M' in self.camera:
            sample_path['camera_mono'] = osp.join(src, 'mono.jpg')
            sample_path['label_to_camera_mono'] = osp.join(src, 'mono_info.npy')

        if 'S' in self.camera:
            sample_path['camera_stereo'] = osp.join(src, 'stereo.jpg')
            sample_path['label_to_camera_stereo'] = osp.join(src, 'stereo_info.npy')

        if 'B' in self.radar:
            sample_path['radar_bev'] = osp.join(src, 'ra.npy')
            sample_path['label_to_radar_bev'] = osp.join(src, 'ra_info.npy')

        if 'F' in self.radar:
            sample_path['radar_front'] = osp.join(src, 'ea.npy')
            sample_path['label_to_radar_front'] = osp.join(src, 'ea_info.npy')

        if self.lidar == 1:
            sample_path['lidar_top'] = osp.join(src, 'os1.npy')

        if self.lidar == 2:
            sample_path['lidar_top'] = osp.join(src, 'os2.npy')


        # Get annotation data
        if self.label == 'detection':
            sample_path['label'] = osp.join(src, 'labels.npy')

        # Get description data
        sample_path['description'] = osp.join(src, 'description.npy')

        return sample_path

    def get_dataset_paths(
        self,
        src: str
    ) -> Union[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        """Returns the paths of all dataset items.

        The return type is either a list of dictionaries (each representing
        a single sample) or a dictionary of lists (each representing a
        single sequence), where each list holds the dictionaries of the
        single samples.

            sequential: Dict[sequence number, List[sample dicts]]
            non sequential: List[sample dicts]

        Arguments:
            src: Source path to the pre-processed
                dataset folder.

        Returns:
            dataset_paths: File paths of all dataset
                items (either sequences or samples).
        """
        # Initialize dataset paths
        dataset_paths = {}

        # List all sequences in the dataset
        for sequence in os.listdir(osp.join(src, self.split)):
            # Set sequence path
            sequence_path = osp.join(src, self.split, sequence)

            # List all samples in the sequence
            samples = sorted(os.listdir(sequence_path))

            # Disolve all sample data paths
            dataset_paths[sequence] = [
                self.get_sample_path(osp.join(sequence_path, sample)) for sample in samples
            ]

        # Concatenate all sequences for non sequential data
        if not self.sequential:
            dataset_paths = list(chain.from_iterable(dataset_paths.values()))

        return dataset_paths

    # def get_dataset_paths(
    #         self,
    #         src: str
    # ) -> Union[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    #     """Returns the paths of all dataset items."""
    #
    #     # Initialize dataset paths
    #     dataset_paths = {}
    #
    #     # List all sequences in the dataset
    #     for sequence in os.listdir(osp.join(src, self.split)):
    #         # Set sequence path
    #         sequence_path = osp.join(src, self.split, sequence)
    #
    #         # List all samples in the sequence
    #         samples = sorted(os.listdir(sequence_path))
    #
    #         # 收集有效样本路径
    #         valid_samples = []
    #         for sample in samples:
    #             sample_path_dict = self.get_sample_path(osp.join(sequence_path, sample))
    #
    #             # 检查标签文件是否存在
    #             if self.label == 'detection':
    #                 label_file = sample_path_dict.get('label')
    #                 if label_file is None or not osp.exists(label_file):
    #                     # print(f"⚠️  跳过样本 (标签文件不存在): {sample}")
    #                     continue
    #
    #             valid_samples.append(sample_path_dict)
    #
    #         if valid_samples:  # 只添加非空序列
    #             dataset_paths[sequence] = valid_samples
    #
    #     # Concatenate all sequences for non sequential data
    #     if not self.sequential:
    #         dataset_paths = list(chain.from_iterable(dataset_paths.values()))
    #
    #     print(f"✅ 数据集加载完成: {len(dataset_paths)} 个有效样本")
    #     return dataset_paths

    def load_sample_data(self, sample_path: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Returns the actual sample data given their paths."""
        sample = {}

        # Load sample data
        for key, path in sample_path.items():
            if osp.splitext(path)[-1] in {'.png', '.jpg'}:
                img: torch.Tensor = read_image(path).type(getattr(torch, self.dtype))
                sample[key] = img.movedim(0, -1)
            elif osp.splitext(path)[-1] in {'.npy'}:
                sample[key] = torch.from_numpy(np.load(path)).type(getattr(torch, self.dtype))

        # 2. LiDAR -> BEV
        if self.lidar > 0:
            try:
                # 如果没有 lidar_top，会 KeyError 跳到 except
                pts = sample['lidar_top']
                filtered = self.filter_lidar_points(pts)
                bev = self.points_to_bev(filtered)

                # # 确保通道维度在第 0 维
                # dims = bev.shape
                # chan_dims = [i for i, s in enumerate(dims) if s == 4]
                # if len(chan_dims) != 1:
                #     raise RuntimeError(f"无法确定 BEV 通道维度: shape={dims}")
                # c_dim = chan_dims[0]
                # if c_dim != 0:
                #     bev = bev.movedim(c_dim, 0)
                #
                # # 兜底断言
                # assert bev.dim() == 3 and bev.shape[0] == 4, \
                #     f"BEV tensor 维度异常: {bev.shape}"

                sample['lidar_bev'] = bev

            except Exception:
                # 任何异常都使用零 BEV 兜底
                try:
                    from dprt.datasets.kradar.utils import lidar_bev_config
                    H, W = lidar_bev_config.BEV_HEIGHT, lidar_bev_config.BEV_WIDTH
                except ImportError:
                    H, W = 64, 360
                sample['lidar_bev'] = torch.zeros((4, H, W), dtype=getattr(torch, self.dtype))
            # **删除原始点云，防止 batch 时尺寸不一**
            sample.pop('lidar_top', None)
        # for key in ('radar_bev', 'radar_front'):
        #     if key in sample:
        #         v = sample[key]
        #         # v.dim()==3 且 shape==(H,W,C)
        #         sample[key] = v.permute(2, 0, 1) # -> [C, H, W]

        #转换 lidar_bev 形状 [4,H,W] -> [W,H,4] 因为resnet.py那里会换回去
            if 'lidar_bev' in sample:
                sample['lidar_bev'] = sample['lidar_bev'].permute(2, 1, 0)
        return sample

    def modality_dropout(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies modality dropout to the sample data.

        Randomly drops one input modality by setting all input
        values to zero. The drop ratio of each modality is given
        by their individual dropout propabilities.

        Note: It is ensured that not all modalities are dropped
        at the same time but at least one modality remains.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample items to
                thier data tensors with applied dropout.
        """
        # Draw of lots (select a modality based on their probabilities)
        drawing = self.lottery[np.random.choice(4, replace=True, p=self.dropout)]

        # Apply dropout (replace selected input modality with zeros)
        for modality in drawing:
            if modality in sample:
                sample[modality] = torch.zeros_like(sample[modality])

        return sample

    #新增lidar bev投影矩阵计算方法
    def _get_lidar_bev_projection(self) -> torch.Tensor:
        # LiDAR BEV 投影矩阵
        scale_x = L_W / (L_XR[1] - L_XR[0])
        scale_y = L_H / (L_YR[1] - L_YR[0])
        offset_x = -L_XR[0] * scale_x
        offset_y = -L_YR[0] * scale_y
        return torch.Tensor([
            [scale_x, 0, 0, offset_x],
            [0, scale_y, 0, offset_y],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def scale_lidar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scales LiDAR BEV data to [0, 255] using lidar_bev_config."""
        if 'lidar_bev' in sample:
            bev = sample['lidar_bev']  # 形状: [4, H, W]

            # 高度通道 (0)：基于配置的 Z 范围
            z_min, z_max = lidar_bev_config.BEV_Z_RANGE
            if bev[0].max() > bev[0].min():
                bev[0] = (bev[0] - z_min) / (z_max - z_min) * 255

            # 强度通道 (1)：基于配置的 intensity 范围
            i_min, i_max = lidar_bev_config.min_intensity, lidar_bev_config.max_intensity
            if bev[1].max() > bev[1].min():
                bev[1] = (bev[1] - i_min) / (i_max - i_min) * 255

            # 密度通道 (2)：基于配置的 point density 范围
            d_min, d_max = lidar_bev_config.min_point_density, lidar_bev_config.max_point_density
            if bev[2].max() > d_min:
                bev[2] = (bev[2] - d_min) / (d_max - d_min) * 255

            # 平均高度通道 (3)：同通道0
            if bev[3].max() > bev[3].min():
                bev[3] = (bev[3] - z_min) / (z_max - z_min) * 255

            # 限制到 [0,255]
            sample['lidar_bev'] = torch.clamp(bev, 0, 255)

        return sample

    #点云到BEV
    def points_to_bev(self, points: torch.Tensor) -> torch.Tensor:
        """
        将 LiDAR 点云投影到 BEV，输出 3 通道:
          0: 点数密度
          1: 最大高度
          2: 最大反射强度
        """
        # 1. 空输入
        if points.numel() == 0:
            return torch.zeros((3, L_H, L_W), dtype=getattr(torch, self.dtype))

        # 2. 拆分坐标与强度
        pts_xyz = points[:, :3]
        intensity = points[:, 3] if points.shape[1] >= 4 \
            else torch.full((pts_xyz.size(0),), 128.0, dtype=getattr(torch, self.dtype))

        # 3. 过滤 FOV
        mask = (
                (pts_xyz[:, 0] >= L_XR[0]) & (pts_xyz[:, 0] <= L_XR[1]) &
                (pts_xyz[:, 1] >= L_YR[0]) & (pts_xyz[:, 1] <= L_YR[1]) &
                (pts_xyz[:, 2] >= L_ZR[0]) & (pts_xyz[:, 2] <= L_ZR[1])
        )
        pts, ints = pts_xyz[mask], intensity[mask]
        if pts.numel() == 0:
            return torch.zeros((3, L_H, L_W), dtype=getattr(torch, self.dtype))

        # 4. 坐标映射
        xpix = ((pts[:, 0] - L_XR[0]) / (L_XR[1] - L_XR[0]) * L_W).long().clamp(0, L_W - 1)
        ypix = ((pts[:, 1] - L_YR[0]) / (L_YR[1] - L_YR[0]) * L_H).long().clamp(0, L_H - 1)

        # 5. 初始化 BEV
        bev = torch.zeros((3, L_H, L_W), dtype=getattr(torch, self.dtype))

        # 6. 填充通道
        for i in range(pts.size(0)):
            x, y = xpix[i].item(), ypix[i].item()
            z_val, i_val = pts[i, 2].item(), ints[i].item()
            bev[0, y, x] += 1  # density
            bev[1, y, x] = max(bev[1, y, x].item(), z_val)  # max-height
            bev[2, y, x] = max(bev[2, y, x].item(), i_val)  # max-intensity

        return bev

    def filter_lidar_points(self, lidar_points: torch.Tensor) -> torch.Tensor:
        """
        过滤和清洗原始LiDAR点云数据

        Args:
            lidar_points: 原始LiDAR点云数据 [N, 9]

        Returns:
            filtered_points: 过滤后的LiDAR点云数据 [M, 9] (M <= N)
        """
        if lidar_points.numel() == 0:
            return lidar_points

        #print(f"🔧 开始过滤LiDAR数据: 原始点数 {len(lidar_points)}")

        # 1. 基本数据完整性检查
        if lidar_points.shape[1] < 4:
            #print("⚠️ LiDAR数据通道数不足4，跳过过滤")
            return lidar_points

        # 提取基本坐标和强度信息
        points_xyz = lidar_points[:, :3]  # x, y, z
        intensity = lidar_points[:, 3]  # 反射强度

        # 2. 异常值检测和清理
        #print("  📋 执行异常值过滤...")

        # 强度值合理性检查
        valid_intensity_mask = (intensity > 0) & (intensity < 1000000)

        # 坐标值合理性检查 (移除极端异常值)
        valid_xyz_mask = (torch.abs(points_xyz).max(dim=1)[0] < 1000)

        # 3. 距离过滤 (OS1-128最大120米)
        distances = torch.sqrt(points_xyz[:, 0] ** 2 + points_xyz[:, 1] ** 2)
        valid_distance_mask = (distances >= 0.1) & (distances <= 120.0)  # 0.1m-120m

        # 4. FOV范围过滤 (基于配置)
        try:
            from dprt.datasets.kradar.utils import lidar_bev_config

            # 使用稍微扩展的FOV范围，避免过度过滤
            extended_x_range = [lidar_bev_config.BEV_X_RANGE[0] - 5.0, lidar_bev_config.BEV_X_RANGE[1] + 5.0]
            extended_y_range = [lidar_bev_config.BEV_Y_RANGE[0] - 2.0, lidar_bev_config.BEV_Y_RANGE[1] + 2.0]
            extended_z_range = [lidar_bev_config.BEV_Z_RANGE[0] - 1.0, lidar_bev_config.BEV_Z_RANGE[1] + 2.0]

            fov_x_mask = (points_xyz[:, 0] >= extended_x_range[0]) & (points_xyz[:, 0] <= extended_x_range[1])
            fov_y_mask = (points_xyz[:, 1] >= extended_y_range[0]) & (points_xyz[:, 1] <= extended_y_range[1])
            fov_z_mask = (points_xyz[:, 2] >= extended_z_range[0]) & (points_xyz[:, 2] <= extended_z_range[1])

            fov_mask = fov_x_mask & fov_y_mask & fov_z_mask

        except ImportError:
            print("  ⚠️ 无法导入bev_config，跳过FOV过滤")
            fov_mask = torch.ones(len(points_xyz), dtype=torch.bool)

        # 5. 综合所有过滤条件
        comprehensive_mask = valid_intensity_mask & valid_xyz_mask & valid_distance_mask & fov_mask

        # 应用过滤
        filtered_lidar_points = lidar_points[comprehensive_mask]

        # 6. 输出过滤统计信息
        n_original = len(lidar_points)
        n_filtered = len(filtered_lidar_points)
        retention_rate = n_filtered / n_original * 100

        # print(f"  📊 过滤统计:")
        # print(f"    - 强度异常: {(~valid_intensity_mask).sum()} 点")
        # print(f"    - 坐标异常: {(~valid_xyz_mask).sum()} 点")
        # print(f"    - 距离超限: {(~valid_distance_mask).sum()} 点")
        # print(f"    - FOV范围外: {(~fov_mask).sum()} 点")
        # print(f"    - 保留点数: {n_filtered}/{n_original} ({retention_rate:.1f}%)")

        if n_filtered == 0:
            #print("  ⚠️ 过滤后无有效点云数据，返回原始数据")
            return lidar_points

        # 7. 验证过滤后数据的基本属性
        if n_filtered > 0:
            filtered_xyz = filtered_lidar_points[:, :3]
            filtered_distances = torch.sqrt(filtered_xyz[:, 0] ** 2 + filtered_xyz[:, 1] ** 2)

            # print(f"  ✅ 过滤后数据验证:")
            # print(f"    - X范围: [{filtered_xyz[:, 0].min():.2f}, {filtered_xyz[:, 0].max():.2f}]")
            # print(f"    - Y范围: [{filtered_xyz[:, 1].min():.2f}, {filtered_xyz[:, 1].max():.2f}]")
            # print(f"    - Z范围: [{filtered_xyz[:, 2].min():.2f}, {filtered_xyz[:, 2].max():.2f}]")
            # print(f"    - 距离范围: [{filtered_distances.min():.2f}, {filtered_distances.max():.2f}]")

        return filtered_lidar_points


def initialize_kradar(*args, **kwargs):
    return KRadarDataset.from_config(*args, **kwargs)
