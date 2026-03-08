import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 split_ratios=None, scale_mode='train', noise_std=0.0,
                 max_rows=None):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        # 允许逗号分隔的多目标（如 MM264,MM256）
        self.target = target
        self.target_cols = [t.strip() for t in target.split(',')]
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.split_ratios = split_ratios
        self.scale_mode = scale_mode
        self.noise_std = noise_std
        self.max_rows = max_rows

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # 分开 scaler：特征/目标各自标准化，避免目标量纲被其他特征放缩
        self.scaler_feat = StandardScaler()
        self.scaler_target = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.max_rows is not None:
            df_raw = df_raw.head(int(self.max_rows))

        # 兼容不同时间列命名（date 或 datetime）
        time_col = 'date'
        if time_col not in df_raw.columns:
            if 'datetime' in df_raw.columns:
                time_col = 'datetime'
            else:
                raise ValueError('Expect a time column named date or datetime in the CSV file')

        # 重排列：time_col + 非目标列 + 目标列（可多列）
        cols = [c for c in df_raw.columns if c != time_col]
        non_target_cols = [c for c in cols if c not in self.target_cols]
        df_raw = df_raw[[time_col] + non_target_cols + self.target_cols]

        if self.split_ratios is None:
            train_ratio, val_ratio = 0.7, 0.1
        else:
            train_ratio, val_ratio = self.split_ratios

        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError('Invalid split ratios, expect train>0, val>=0 and train+val<1')

        num_train = int(len(df_raw) * train_ratio)
        num_vali = int(len(df_raw) * val_ratio)
        num_test = len(df_raw) - num_train - num_vali
        if num_test <= 0:
            raise ValueError('Split ratios leave no samples for test set')

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 选择输入列
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]  # 除去 date
        else:  # 'S'
            cols_data = self.target_cols

        df_data = df_raw[cols_data]

        if self.scale:
            fit_slice = df_data if self.scale_mode == 'global' else df_data.iloc[border1s[0]:border2s[0]]
            feat_cols = [c for c in cols_data if c not in self.target_cols]
            tgt_cols = [c for c in cols_data if c in self.target_cols]

            if feat_cols:
                self.scaler_feat.fit(fit_slice[feat_cols].values)
                feat_scaled = self.scaler_feat.transform(df_data[feat_cols].values)
            else:
                feat_scaled = np.empty((len(df_data), 0))

            self.scaler_target.fit(fit_slice[tgt_cols].values)
            tgt_scaled = self.scaler_target.transform(df_data[tgt_cols].values)

            data = np.concatenate([feat_scaled, tgt_scaled], axis=1)
            self.data_cols_order = feat_cols + tgt_cols
        else:
            data = df_data.values
            self.data_cols_order = list(cols_data)

        # 记录目标列在当前顺序中的索引，便于仅对目标计算损失/输出
        self.target_indices = [self.data_cols_order.index(c) for c in self.target_cols if c in self.data_cols_order]

        df_stamp = df_raw[[time_col]][border1:border2]
        df_stamp[time_col] = pd.to_datetime(df_stamp[time_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[time_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[time_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[time_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[time_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([time_col], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp[time_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.noise_std > 0 and self.set_type == 0:
            seq_x = seq_x + np.random.normal(0.0, self.noise_std, seq_x.shape)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, only_target=True):
        """
        默认只对目标列做反归一化，保持与分开 scaler 的逻辑一致。
        若传入的 data 只包含目标列，则直接 inverse；
        若包含特征列，请先切片再调用。
        """
        if not self.scale:
            return data
        if only_target:
            return self.scaler_target.inverse_transform(data)
        # 对包含特征+目标的数组，按记录的列顺序分别反归一化。
        feat_cols = [c for c in self.data_cols_order if c not in self.target_cols]
        tgt_cols = [c for c in self.data_cols_order if c in self.target_cols]
        feat_dim = len(feat_cols)
        feat_part = data[:, :feat_dim] if feat_dim > 0 else np.empty((len(data), 0))
        tgt_part = data[:, feat_dim:]
        if feat_dim > 0:
            feat_part = self.scaler_feat.inverse_transform(feat_part)
        tgt_part = self.scaler_target.inverse_transform(tgt_part)
        return np.concatenate([feat_part, tgt_part], axis=1)


class Dataset_PCA_Custom(Dataset):
    """
    读取预先做好的 PCA 数据（特征已标准化+PCA，目标已标准化）
    期望文件结构：
      root_path/
        gas_pca_all.csv 或单独 train/val/test（但这里读取 all 按比例再切）
        scaler_target.pkl, scaler_feat.pkl, pca.pkl, pca_meta.json
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='gas_pca_all.csv',
                 target='MM264,MM256', scale=False, timeenc=0, freq='h',
                 split_ratios=None, pca_dir=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.target_cols = [t.strip() for t in target.split(',')]
        self.scale = False  # 已经标准化过，不再二次缩放
        self.timeenc = timeenc
        self.freq = freq
        self.split_ratios = split_ratios

        self.root_path = root_path
        self.data_path = data_path
        self.pca_dir = pca_dir or root_path
        self.__read_data__()

    def __read_data__(self):
        from pathlib import Path
        import joblib
        meta_path = Path(self.pca_dir) / 'pca_meta.json'
        scaler_tgt_path = Path(self.pca_dir) / 'scaler_target.pkl'
        if not meta_path.exists() or not scaler_tgt_path.exists():
            raise FileNotFoundError('pca_meta.json/scaler_target.pkl 不存在，请先运行 scripts/pca_preprocess.py')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.time_col = meta.get('time_col', 'datetime')
        self.scaler_target = joblib.load(scaler_tgt_path)

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.time_col not in df_raw.columns:
            raise ValueError(f'PCA 文件缺少时间列 {self.time_col}')

        # 切分比例：优先用 meta 中记录，其次用传入参数，默认 0.7/0.1/0.2
        if self.split_ratios is None:
            train_ratio = meta.get('train_ratio', 0.7)
            val_ratio = meta.get('val_ratio', 0.1)
        else:
            train_ratio, val_ratio = self.split_ratios
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError('Invalid split ratios for PCA dataset')

        num_train = int(len(df_raw) * train_ratio)
        num_vali = int(len(df_raw) * val_ratio)
        num_test = len(df_raw) - num_train - num_vali
        if num_test <= 0:
            raise ValueError('Split ratios leave no samples for test set')

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 列顺序：pc* 在前，目标列在后，时间列最后
        cols = [c for c in df_raw.columns if c != self.time_col]
        non_target_cols = [c for c in cols if c not in self.target_cols]
        df_raw = df_raw[non_target_cols + self.target_cols + [self.time_col]]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[:-1]  # 除去时间列
        else:  # 'S'
            cols_data = self.target_cols

        df_data = df_raw[cols_data]

        data = df_data.values  # 已经标准化+PCA，直接使用
        self.data_cols_order = list(cols_data)

        df_stamp = df_raw[[self.time_col]][border1:border2]
        df_stamp[self.time_col] = pd.to_datetime(df_stamp[self.time_col])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.target_indices = [self.data_cols_order.index(c) for c in self.target_cols if c in self.data_cols_order]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, only_target=True):
        if not only_target:
            return data
        return self.scaler_target.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
