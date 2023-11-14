
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, default_collate
import pytorch_lightning as pl
import numpy as np
from dataclasses import dataclass, fields, is_dataclass


@dataclass
class Coords:
    station: str
    warmup_start_date: str
    start_date: str
    end_date: str
    warmup_size: int


@dataclass
class BatchPattern:
    """Class defining MachFlowData return pattern."""
    dfeatures: Tensor | np.ndarray
    dtargets: Tensor | np.ndarray
    coords: Coords
    sfeatures: Tensor | np.ndarray | None = None


def collate_dataclass(batch):
    elem_type = type(batch[0])

    r = []
    for field in fields(elem_type):
        el = [getattr(b, field.name) for b in batch]

        if is_dataclass(el[0]):
            r.append(collate_dataclass(el))
        else:
            if el[0] is None:
                r.append(None)
            else:
                r.append(default_collate(el))

    return elem_type(*r)


class MachFlowData(Dataset):
    """Describes a MACH-Flow dataset. See `MachFlowDataModule` for more details."""
    def __init__(
            self,
            ds: xr.Dataset,
            features: list[str],
            targets: list[str],
            stat_features: list[str],
            window_size: int = -1,
            window_min_count: int = -1,
            warmup_size: int = 0,
            num_samples_per_epoch: int = 1,
            load_ds: bool = True,
            seed: int = 19):

        if load_ds:
            ds = ds.load()

        self.check_vars_in_ds(ds=ds, vars=features + targets)
        self.features = features
        self.targets = targets
        self.ds = self.add_static_vars(ds=ds, vars=self.features)
        self.stat_features = [] if stat_features is None else stat_features
        self.check_vars_in_ds(ds=ds, vars=self.stat_features)

        self.window_size = window_size
        self.window_min_count = window_min_count
        self.warmup_size = warmup_size

        if self.window_size > 0:
            self.target_rolling_mask = self.compute_target_rolling_mask()
        else:
            self.target_rolling_mask = None

        self.num_samples_per_epoch = num_samples_per_epoch

        self.rs = np.random.RandomState(seed=seed)

    def __len__(self) -> int:
        return len(self.ds.station) * self.num_samples_per_epoch

    def __getitem__(self, ind: int) -> BatchPattern:

        station = ind // self.num_samples_per_epoch

        data_f = self.get_features(station=station)
        data_s = self.get_stat_features(station=station)
        data_t = self.get_targets(station=station)

        target_rolling_mask = self.get_target_rolling_mask(station=station)

        if target_rolling_mask is not None:
            indices = np.argwhere(target_rolling_mask.values)[:, 0]
            if len(indices) < 1:
                raise RuntimeError(
                    f'Station #{station} has no window with window_size={self.window_size} with at least '
                    f'window_min_count={self.window_min_count} valid time steps.'
                )

            window_end = self.rs.choice(indices) + 1
            window_start = window_end - self.window_size
            warmup_start = window_end - self.window_size - self.warmup_size

            data_f = data_f.isel(time=slice(warmup_start, window_end))
            data_t = data_t.isel(time=slice(warmup_start, window_end))

        else:
            warmup_start = 0
            window_start = self.warmup_size
            window_end = -1

        return_data = BatchPattern(
            dfeatures=data_f.values.astype('float32'),
            sfeatures=None if data_s is None else data_s.values.astype('float32'),
            dtargets= data_t.values.astype('float32'),
            coords=Coords(
                station=self.ds.station.isel(station=station).item(),
                warmup_start_date=self.ds.time[warmup_start].dt.strftime('%Y-%m-%d').item(),
                start_date=self.ds.time[window_start].dt.strftime('%Y-%m-%d').item(),
                end_date=self.ds.time[window_end - 1].dt.strftime('%Y-%m-%d').item(),
                warmup_size=self.warmup_size
            )
        )

        return return_data

    def compute_target_mask(self, **isel) -> xr.DataArray:
        return self.get_targets(**isel).notnull().any('variable')

    def compute_target_rolling_mask(self, **isel) -> xr.DataArray:
        target_mask = self.compute_target_mask(**isel).load()
        target_rolling_mask = target_mask.rolling(time=self.window_size).sum() >= self.window_min_count
        return target_rolling_mask.compute()

    def get_target_rolling_mask(self, **isel) -> xr.DataArray | None:
        if self.target_rolling_mask is None:
            return None
        else:
            return self.target_rolling_mask.isel(**isel)

    def get_features(self, **isel) -> xr.DataArray:
        return self.ds[self.features].isel(**isel).to_array('variable')

    def get_stat_features(self, **isel) -> xr.DataArray | None:
        if self.stat_features is None:
            return None
        else:
            return self.ds[self.stat_features].isel(**isel).to_array('variable')

    def get_targets(self, **isel) -> xr.DataArray:
        return self.ds[self.targets].isel(**isel).to_array('variable')

    def check_vars_in_ds(self, ds: xr.Dataset, vars: str | list[str]) -> None:
        vars = [vars] if isinstance(vars, str) else vars

        missing_vars = []
        for var in vars:
            if var not in ds.data_vars:
                missing_vars.append(var)

        if len(missing_vars) > 0:
            raise KeyError(
                f'variable(s) `{"`, `".join(missing_vars)}` not found in the provided dataset `ds`. '
                f'Valid variables are: `{"`, `".join(list(ds.data_vars))}`.'
            )

    def add_static_vars(self, ds: xr.Dataset, vars: str | list[str]) -> xr.Dataset:
        vars = [vars] if isinstance(vars, str) else vars
        
        for var in vars:
            da = ds[var]

            ds[f'{var}_mon_std'] = da.groupby('time.month').mean().std('month').compute()
            ds[f'{var}_std'] = da.std('time').compute()
            ds[f'{var}_mean'] = da.mean('time').compute()

        return ds



class MachFlowDataModule(pl.LightningDataModule):
    """Describes a MACH-Flow data modukle.

    This MachFlowDataModule builds upon the pl.LightningDataModule. It can be used to define and retrieve the
    training, validation, test, and prediction dataloaders and integrates with the pl.Trainer framework.

    For training, a random subset of the time-series of length `train_window_size` is sampled from the basins, but
    only for windows with at least `window_min_count` observations of a target variable present. For each basin,
    `train_num_samples_per_epoch` random windows are loaded in an epoch.

    A warmup period of length `warmup_size` is added for model spin-up. Turn off this behavior by setting
    `warmup_size=0`. If `drop_all_nan_stations` is set to True, sites with all NaN targets are dropped entirely
    from training, validation, and test.

    Shapes:
        The dataloaders return a `BatchPattern` with elements/shapes:
        - Dynamic features (dfeatures): (batch_size, num_dfeatures, seq_length,)
        - Static features (sfeatures, optional): (batch_size, num_stargets,)
        - Dynamic targets (dtargets): (batch_size, num_dtargets, seq_length,)
        - Coords (coords):
            - Station ID (station): (batch_size,)
            - Window start date including warmup (warmup_start_date): (batch_size,)
            - Window start date without warmup (start_date): (batch_size,)
            - Window end date (end_date): (batch_size,)
            - Number of warmup steps (warmup_size): (batch_size,)

    """
    def __init__(
            self,
            machflow_data_path: str,
            features: list[str],
            targets: list[str],
            stat_features: list[str] | None = None,
            train_window_size: int = 1000,
            window_min_count: int = 1,
            train_num_samples_per_epoch: int = 1,
            warmup_size=365,
            drop_all_nan_stations: bool = True,
            num_cv_folds: int = 6,
            fold_nr: int = 0,
            train_date_slice: slice = slice(None, None),
            valid_date_slice: slice = slice(None, None),
            test_date_slice: slice = slice(None, None),
            predict_date_slice: slice = slice(None, None),
            batch_size: int = 10,
            num_workers: int = 10,
            seed: int = 19):
        """Initialize MachFlowDataModule.

        Args:
            machflow_data_path (str): Path to machflow zarr file.
            features (list[str]): A list of features.
            targets (list[str]): A list of targets.
            stat_features (list[str]): A list of static features.
            train_window_size (int, optional): The training window number of time steps. Defaults to 1000.
            window_min_count (int, optional): Minimum number of target observations in time window. Defaults to 1.
            train_num_samples_per_epoch (int, optional): Number of samples of size `train_window_size` to draw from
                each station during training. If 10, for example, 10 random windows will be used in training for each
                station. Defaults to 1.
            warmup_size (int, optional): The number of warmup staps to use for model spinup. Defaults to 365.
            drop_all_nan_stations (bool, optional): If True, only stations with at least one observation are used.
                Applies for all modes but 'prediction'. Defaults to True.
            num_cv_folds (int): The number of cross-validation sets (folds). Defaults to 6.
            fold_nr (int): The fold to use, a value in the range [0, num_cv_folds). Defaults to 0.
            [train/valid/test/predict]_date_slice: (slice): Slice to subset the respective set in time. Defaults 
                to no subsetting.
            batch_size (int, optional): The minibatch batch size. Defaults to 10.
            num_workers (int, optional): The number of workers. Defaults to 10.
                to ()
            seed (int, optional): The random seed. Defaults to 19.
        """

        self.machflow_data_path = machflow_data_path
        self.features = features
        self.targets = targets
        self.stat_features = stat_features
        self.train_window_size = train_window_size
        self.window_min_count = window_min_count
        self.train_num_samples_per_epoch = train_num_samples_per_epoch
        self.warmup_size = warmup_size
        self.drop_all_nan_stations = drop_all_nan_stations
        self.num_cv_folds = num_cv_folds
        self.fold_nr = fold_nr
        self.train_date_slice = train_date_slice
        self.valid_date_slice = valid_date_slice
        self.test_date_slice = test_date_slice
        self.predict_date_slice = predict_date_slice
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Lazy-load zarr dataset.
        self.ds = xr.open_zarr('/data/basil/harmonized_basins.zarr/')

        # Drop all stations with all-NaN targets.
        if drop_all_nan_stations:
            mask = self.ds[self.targets].to_array('variable').notnull().any('variable')
            stations = mask.station.values[mask.any('time')]
        else:
            stations = self.ds.station.values

        # Create random state with seed.
        self.rs = np.random.RandomState(seed)

        # Split basins into training, validation, and test set.
        self.train_basins, self.val_basins, self.test_basins = self.split_basins(basins=stations)
        self.predict_basins = self.ds.station.values

    def get_dataset(self, mode: str) -> Dataset:
        """Returns a PyTorch Dataset of type MachFlowData.

        Args:
            mode (str): The evaluation mode, one of 'train', 'val', 'test', or 'predict'.

        Raises:
            ValueError: non-valid mode.

        Returns:
            Dataset: A MachFlowData instance.
        """        
        if mode == 'train':
            window_size = self.train_window_size
            window_min_count = self.window_min_count
            num_samples_per_epoch = self.train_num_samples_per_epoch
            basins = self.train_basins
            time_sel = self.train_date_slice
        elif mode == 'val':
            window_size = -1
            window_min_count = self.window_min_count
            num_samples_per_epoch = 1
            basins = self.val_basins
            time_sel = self.valid_date_slice
        elif mode == 'test':
            window_size = -1
            window_min_count = self.window_min_count
            num_samples_per_epoch = 1
            basins = self.test_basins
            time_sel = self.test_date_slice
        elif mode == 'predict':
            window_size = -1
            window_min_count = -1
            num_samples_per_epoch = 1
            basins = self.predict_basins
            time_sel = self.predict_date_slice
        else:
            raise ValueError(
                f'mode \'{mode}\' not valid, must be one of \'train\', \'val\', \'test\', or \'predict\'.'
            )

        dataset = MachFlowData(
            ds=self.ds.sel(station=basins, time=time_sel),
            features=self.features,
            targets=self.targets,
            stat_features=self.stat_features,
            window_size=window_size,
            window_min_count=window_min_count,
            num_samples_per_epoch=num_samples_per_epoch,
            seed=self.seed
        )

        return dataset

    def common_dataloader(self, mode: str) -> DataLoader:
        """Returns a PyTorch DataLoader.

        Args:
            mode (str): The evaluation mode, one of 'train', 'val', 'test', or 'predict'.

        Returns:
            DataLoader: the data loader.
        """        
        data = self.get_dataset(mode=mode)

        dataloader = DataLoader(
            dataset=data,
            shuffle=mode=='train',
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_dataclass
        )

        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader.

        Returns:
            DataLoader: The training dataloader.
        """        
        return self.common_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader.

        Returns:
            DataLoader: The validation dataloader.
        """  
        return self.common_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader.

        Returns:
            DataLoader: The test dataloader.
        """  
        return self.common_dataloader('test')

    def predict_dataloader(self) -> DataLoader:
        """Get the prediction data loader.

        Returns:
            DataLoader: The prediction dataloader.
        """  
        return self.common_dataloader('predict')

    def split_basins(
            self,
            basins: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Split and return the basins/statinos.

        Args:
            basins (list[str]): A list of basin IDs.
            train_frac (float): Training set size relative to validation and test set.
            val_frac (float): Validation set size relative to training and test set.
            test_frac (float): Test set size relative to training and test validation.

        Returns:
            tuple[list[str], list[str], list[str]]: The training, validation, and test basin IDs.
        """

        if self.fold_nr not in range(self.num_cv_folds):
            raise ValueError(
                f'\'fold_nr\' must be in range [0, {self.num_cv_folds}), is {self.fold_nr}.'
            )

        basins = list(self.rs.permutation(basins))
        groups = np.array_split(basins, self.num_cv_folds)

        folds = {i for i in range(self.num_cv_folds)}
        valid_folds = {self.fold_nr % (self.num_cv_folds - 1)}
        test_folds = {(self.fold_nr + 1) % (self.num_cv_folds - 1)}
        train_folds = folds - valid_folds - test_folds

        train_basins = []
        valid_basins = []
        test_basins = []
        for group_i, group in enumerate(groups):
            if group_i in train_folds: train_basins.extend(group)
            if group_i in valid_folds: valid_basins.extend(group)
            if group_i in test_folds: test_basins.extend(group)


        return train_basins, valid_basins, test_basins
