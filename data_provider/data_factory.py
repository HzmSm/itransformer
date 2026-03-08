from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, Dataset_Pred, Dataset_PCA_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'pca_custom': Dataset_PCA_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # when probing distribution shift, reuse test split as train/val dataset
    dataset_flag = 'test' if getattr(args, 'train_on_test', False) and flag in ['train', 'val'] else flag

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=dataset_flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        scale=args.scale,
        max_rows=getattr(args, 'max_rows', None),
    )

    # extra controls for custom dataset: split ratios, scaling mode, and optional input noise
    if Data.__name__ == 'Dataset_Custom':
        dataset_kwargs.update(
            split_ratios=(getattr(args, 'train_ratio', 0.7), getattr(args, 'val_ratio', 0.1)),
            scale_mode=getattr(args, 'scale_mode', 'train'),
            noise_std=getattr(args, 'input_noise', 0.0),
        )
    if Data.__name__ == 'Dataset_PCA_Custom':
        dataset_kwargs.update(
            split_ratios=(getattr(args, 'train_ratio', 0.7), getattr(args, 'val_ratio', 0.1)),
            pca_dir=getattr(args, 'pca_dir', None),
        )

    data_set = Data(**dataset_kwargs)
    print(flag, len(data_set))

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
    if args.num_workers and args.persistent_workers:
        loader_kwargs['persistent_workers'] = True
    if args.num_workers and args.prefetch_factor:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    data_loader = DataLoader(
        data_set,
        **loader_kwargs,
    )
    return data_set, data_loader
