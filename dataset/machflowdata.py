
import xarray as xr
from torch.utils.data import Dataset, DataLoader, default_collate
import lightning.pytorch as pl
import numpy as np
from dataclasses import fields, is_dataclass
import os
import math

from utils.torch_modules import Normalize, RobustStandardize
from utils.types import SamplePattern, SampleCoords


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
            stat_features: list[str] | None = None,
            window_size: int = -1,
            window_min_count: int = -1,
            warmup_size: int = 0,
            num_samples_per_epoch: int = 1,
            load_ds: bool = True,
            seed: int = 19):

        if load_ds:
            ds = ds.load()

        self.check_vars_in_ds(ds=ds, variables=features + targets)
        self.features = features
        self.targets = targets
        self.ds = ds # = self.add_static_vars(ds=ds, variables=self.features)
        self.stat_features = [] if stat_features is None else stat_features
        self.check_vars_in_ds(ds=ds, variables=self.stat_features)

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

    def __getitem__(self, ind: int) -> SamplePattern:

        station = ind // self.num_samples_per_epoch

        data_f = self.get_features(station=station)
        data_s = self.get_stat_features(station=station)
        data_t = self.get_targets(station=station)

        target_rolling_mask = self.get_target_rolling_mask(station=station)

        if target_rolling_mask is not None:
            indices = np.argwhere(target_rolling_mask.values)[:, 0]
            if len(indices) < 1:
                raise RuntimeError(
                    f'Station #{station} ({data_f.station.item()}) has no window with window_size={self.window_size} '
                    f'with at least window_min_count={self.window_min_count} valid time steps.'
                )

            slice_end = self.rs.choice(indices) + 1  # for slicing we need end index + 1
            slice_start = slice_end - self.window_size - self.warmup_size

            warmup_start_index = slice_start
            window_start_index = slice_start + self.warmup_size
            window_end_index = slice_end - 1

            data_f = data_f.isel(time=slice(slice_start, slice_end))
            data_t = data_t.isel(time=slice(slice_start, slice_end))

        else:
            slice_start = None
            slice_end = None

            warmup_start_index = 0
            window_start_index = self.warmup_size
            window_end_index = -1

        return_data = SamplePattern(
            dfeatures=data_f.values.astype('float32'),
            sfeatures=None if data_s is None else data_s.values.astype('float32'),
            dtargets=data_t.values.astype('float32'),
            coords=SampleCoords(
                station=self.ds.station.isel(station=station).item(),
                warmup_start_date=self.ds.time[warmup_start_index].dt.strftime('%Y-%m-%d').item(),
                start_date=self.ds.time[window_start_index].dt.strftime('%Y-%m-%d').item(),
                end_date=self.ds.time[window_end_index].dt.strftime('%Y-%m-%d').item(),
                warmup_size=self.warmup_size
            )
        )

        return return_data

    def compute_target_mask(self, **isel) -> xr.DataArray:
        return self.get_targets(**isel).notnull().any('variable')

    def compute_target_rolling_mask(self, **isel) -> xr.DataArray:
        target_mask = self.compute_target_mask(**isel).load()
        target_rolling_mask = target_mask.rolling(time=self.window_size).sum() >= self.window_min_count
        target_rolling_mask = target_rolling_mask.compute()
        target_rolling_mask[{'time': slice(0, self.warmup_size + self.window_size - 1)}] = False

        return target_rolling_mask

    def get_target_rolling_mask(self, **isel) -> xr.DataArray | None:
        if self.target_rolling_mask is None:
            return None
        else:
            return self.target_rolling_mask.isel(**isel)

    def get_features(self, **isel) -> xr.DataArray:
        return self.ds[self.features].isel(**isel).to_array('variable')

    def get_stat_features(self, **isel) -> xr.DataArray | None:
        if len(self.stat_features) == 0:
            return None
        else:
            return self.ds[self.stat_features].isel(**isel).to_array('variable')

    def get_targets(self, **isel) -> xr.DataArray:
        return self.ds[self.targets].isel(**isel).to_array('variable')
    

    def check_vars_in_ds(self, ds: xr.Dataset, variables: str | list[str]) -> None:
        variables = [variables] if isinstance(variables, str) else variables

        missing_vars = []
        for var in variables:
            if var not in ds.data_vars:
                missing_vars.append(var)

        if len(missing_vars) > 0:
            raise KeyError(
                f'variable(s) `{"`, `".join(missing_vars)}` not found in the provided dataset `ds`. '
                f'Valid variables are: `{"`, `".join(list(ds.data_vars))}`.'
            )

    def add_static_vars(self, ds: xr.Dataset, variables: str | list[str]) -> xr.Dataset:
        variables = [variables] if isinstance(variables, str) else variables
        
        for var in variables:
            da = ds[var]

            ds[f'{var}_mon_std'] = da.groupby('time.month').mean().std('month').compute()
            ds[f'{var}_std'] = da.std('time').compute()
            ds[f'{var}_mean'] = da.mean('time').compute()

        return ds



class MachFlowDataModule(pl.LightningDataModule):
    """MACH-Flow data modukle.

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
        - BatchCoords (coords):
            - Station ID (station): (batch_size,)
            - Window start date including warmup (warmup_start_date): (batch_size,)
            - Window start date without warmup (start_date): (batch_size,)
            - Window end date (end_date): (batch_size,)
            - Number of warmup steps (warmup_size): (batch_size,)

    """
    def __init__(
            self,
            machflow_data_path: str | list[str],
            features: list[str],
            targets: list[str],
            stat_features: list[str] | None = None,
            train_window_size: int = 1000,
            window_min_count: int = 1,
            train_num_samples_per_epoch: int = 1,
            warmup_size=365,
            drop_all_nan_stations: bool = True,
            num_cv_folds: int = 8,
            fold_nr: int = 0,
            use_additional_basins_as_training: bool = False,
            train_tranges: list[str] | None = None,
            valid_tranges: list[str] | None = None,
            test_tranges: list[str] | None = None,
            predict_tslice: list[str | None] = [None, None],
            norm_features: bool = True,
            norm_stat_features: bool = True,
            norm_targets: bool = False,
            batch_size: int = 10,
            num_workers: int = 0,
            seed: int = 19) -> None:
        """Initialize MachFlowDataModule.

        Args:
            machflow_data_path (str): Path to machflow zarr file. Can be a list of strings, then all paths are checked
                and the first on existing is taken. If none exists, an error is raised.
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
            use_additional_basins_as_training (bool): whether to use additional basins (such strongly affected by
                humans) for training. Default is False.
            [train/valid/test]_tranges: (list[str]): List of comma-separated slices to subset the respective
                set in time. For example ['2000,2001', '2010-01-01,2011-10'] to use 2000-01-01 up to 2001-12-31 and
                '2010-01-01' up to '2011-10-31'. Must be passed, if None (default), error is raised.
            predict_tslice: slice to define which period to predict after training, e.g., ['1970', '2020'].
            norm_features (bool): If True, dynamic input features are noramalized. Defaults to True.
            norm_stat_features (bool): If True, static input features are noramalized. Defaults to True.
            norm_targets (bool): If True, targets are standardized (by division). Defaults to False.
            batch_size (int, optional): The minibatch batch size. Defaults to 10.
            num_workers (int, optional): The number of workers. Defaults to 10.
            seed (int, optional): The random seed. Defaults to 19.
        """

        super().__init__()

        self.machflow_data_path = self.validate_paths(machflow_data_path)
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
        self.train_tranges = train_tranges
        self.valid_tranges = valid_tranges
        self.test_tranges = test_tranges
        self.predict_tslice = slice(*predict_tslice)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Lazy-load zarr dataset.
        self.ds = xr.open_zarr(self.machflow_data_path)

        # Drop all stations with all-NaN targets.
        if drop_all_nan_stations:
            # Mask True=valid.
            mask = self.ds[self.targets].to_array('variable').notnull().any('variable').compute()
            train_mask = self.mask_all_nan_basins(mask=mask, tranges=self.train_tranges)
            valid_mask = self.mask_all_nan_basins(mask=mask, tranges=self.valid_tranges)
            test_mask = self.mask_all_nan_basins(mask=mask, tranges=self.test_tranges)

            station_mask = train_mask & valid_mask & test_mask

            stations = mask.station.values[station_mask]
        else:
            stations = self.ds.station.values

        # Create random state with seed.
        self.rs = np.random.RandomState(seed)

        # Split basins into training, validation, and test set.
        self.train_basins, self.val_basins, self.test_basins = self.split_basins(basins=stations, folds=self.ds.folds)

        if use_additional_basins_as_training:
            additional_basins = list(self.ds.station.where(self.ds.folds.load() == -1, drop=True).values)
            additional_basins = [basin for basin in additional_basins if basin in stations]
            self.train_basins += additional_basins

        self.predict_basins = self.ds.station.values
        self.add_cv_set_ids()

        # Data normalization.
        train_data = self.get_dataset('train').ds

        if norm_features:
            self.norm_args_features = Normalize.make_kwargs(
                ds=train_data, norm_variables=self.features
            )
        else:
            self.norm_args_features = None

        if norm_stat_features and (self.stat_features is not None):
            self.norm_args_stat_features = Normalize.make_kwargs(
                ds=train_data, norm_variables=self.stat_features
            )
        else:
            self.norm_args_stat_features = None

        if norm_targets:
            self.norm_args_targets = RobustStandardize.make_kwargs(
                ds=train_data, norm_variables=self.targets
            )
        else:
            self.norm_args_targets = None

        self.num_steps_per_epoch = math.ceil(
            self.train_num_samples_per_epoch * len(self.train_basins) / self.batch_size
        )

    def get_dataset(self, mode: str) -> MachFlowData:
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
            tranges = self.train_tranges
        elif mode == 'val':
            window_size = -1
            window_min_count = self.window_min_count
            num_samples_per_epoch = 1
            basins = self.val_basins
            tranges = self.valid_tranges
        elif mode == 'test':
            window_size = -1
            window_min_count = self.window_min_count
            num_samples_per_epoch = 1
            basins = self.test_basins
            tranges = self.test_tranges
        elif mode == 'predict':
            window_size = -1
            window_min_count = -1
            num_samples_per_epoch = 1
            basins = self.predict_basins
            tranges = None
        else:
            raise ValueError(
                f'mode \'{mode}\' not valid, must be one of \'train\', \'val\', \'test\', or \'predict\'.'
            )

        ds = self.ds.sel(station=basins)

        if tranges is None:
            ds = ds.sel(time=self.predict_tslice).compute()
        else:
            mask = xr.full_like(ds[self.targets[0]], True, dtype=bool)
            for target in self.targets:
                tmask = self.mask_time_slices(mask, tranges)

                ds[target] = ds[target].where(tmask)

            # Cut dataset for validation and test as we set values outside of period to NaN, but they are still in 
            # the dataset. We don't want to run validation and test on the full time range.
            if mode in ['val', 'test']:
                mask_where = np.argwhere(tmask.any('station').compute().values)
                cut_start_time = tmask.time[
                        max(mask_where[0] - self.warmup_size, 0)
                    ].dt.strftime('').item()
                cut_end_time = tmask.time[
                        mask_where[-1]
                    ].dt.strftime('').item()

                ds = ds.sel(time=slice(cut_start_time, cut_end_time))

        dataset = MachFlowData(
            ds=ds.compute(),
            features=self.features,
            targets=self.targets,
            stat_features=self.stat_features,
            window_size=window_size,
            window_min_count=window_min_count,
            warmup_size=self.warmup_size,
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
            num_workers=0 if mode =='predict' else self.num_workers,
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

    @staticmethod
    def validate_paths(paths: str | list[str]) -> str:
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if os.path.isdir(path):
                return path

        raise FileNotFoundError(
            f'None of the provided paths exist:\n `{"`, `".join(paths)}`'
        )

    def add_cv_set_ids(self) -> None:
        cv_set = xr.full_like(self.ds.station, -1, dtype=np.int8)
        cv_set = cv_set.where(~cv_set.station.isin(self.train_basins), 0)
        cv_set = cv_set.where(~cv_set.station.isin(self.val_basins), 1)
        cv_set = cv_set.where(~cv_set.station.isin(self.test_basins), 2)
        cv_set.attrs['note'] = '-1=not used, 0=training, 1=validation, 2=test'
        self.ds['cv_set'] = cv_set.compute()

    def split_basins(
            self,
            basins: list[str],
            folds: xr.DataArray | None = None) -> tuple[list[str], list[str], list[str]]:
        """Split and return the basins/statinos.

        Args:
            basins (list[str]): A list of basin IDs.
            folds (xr.DataArray, optional): An xarray.DataArray containing fold IDs. If not passed, the stations
                are split randomly into `self.num_cv_folds` groups. The fold ID is -1 for `not valid`, and
                0 to self.num_cv_folds - 1 for the folds. Default is None.

        Returns:
            tuple[list[str], list[str], list[str]]: The training, validation, and test basin IDs.
        """

        if self.fold_nr not in range(self.num_cv_folds):
            raise ValueError(
                f'\'fold_nr\' must be in range [0, {self.num_cv_folds-1}], is {self.fold_nr}.'
            )

        if folds is None:
            basins = list(self.rs.permutation(basins))
            groups = np.array_split(basins, self.num_cv_folds)
        else:
            folds = folds.sel(station=basins).load()
            for fold in range(self.num_cv_folds):
                if (folds==fold).sum() < 1:
                    raise RuntimeError(
                        f'At least one fold ID ({fold}) is not present in the `folds` DataArray. '
                        f'For `num_cv_folds={self.num_cv_folds}`, folds from 0 to {self.num_cv_folds-1} are '
                        'expected. Note that some stations may have be dropped due to missing data.'
                    )
            groups = [folds.where(folds==fold, drop=True).station.values for fold in range(self.num_cv_folds)]

        folds = {i for i in range(self.num_cv_folds)}
        valid_folds = {self.fold_nr % self.num_cv_folds}
        test_folds = {(self.fold_nr + 1) % self.num_cv_folds}
        train_folds = folds - valid_folds - test_folds

        train_basins = []
        valid_basins = []
        test_basins = []
        for group_i, group in enumerate(groups):
            if group_i in train_folds:
                train_basins.extend(group)
            if group_i in valid_folds:
                valid_basins.extend(group)
            if group_i in test_folds:
                test_basins.extend(group)


        return train_basins, valid_basins, test_basins

    @staticmethod
    def decode_time_string(x: list[str]) -> list[slice]:
        """Decode time slice strings in form ['2000,2001', '2010-01-01,2011-10'] to list of slices."""

        if not isinstance(x, list):
            raise TypeError(
                f'`x` must be `None` or a `list`, is `{type(x).__name__}`.'
            )

        if len(x) == 0:
            raise ValueError(
                '`x` must not be empty.'
            )

        time_slices = []
        for el in x:
            if not isinstance(el, str):
                raise ValueError(
                    f'element \'{el}\' must be of type `str`, is `{type(el).__name__}`.'
                )

            if ',' not in el:
                raise ValueError(
                    'each time item must be a comma separated pair of start time, end time, e.g., \'2001,2003-05-01\', '
                    f'but no comma found in element \'{el}\'.'
                )

            if ' ' in el:
                raise ValueError(
                    'each time item must be a comma separated pair of start time, end time, e.g., \'2001,2003-05-01\', '
                    f'with no whitespaces, but whitespace found in element \'{el}\'.'
                )

            start_date, end_date = el.split(',')

            try:
                xr.cftime_range(start_date, end_date)

            except Exception as _:
                raise RuntimeError(
                    f'Could not parse element \'{el}\' to time stamps. Infered time start was \'{start_date}\' and '
                    f'time end was \'{end_date}\'.'
                )
            
            time_slices.append(slice(start_date, end_date))

        return time_slices

    def mask_time_slices(self, mask: xr.DataArray, tranges: list[str]) -> xr.DataArray:
        """Update mask with time_slices; values outside of slices get set to False."""

        time_slices = self.decode_time_string(x=tranges)

        time_mask = xr.full_like(mask, False)

        for time_slice in time_slices:
            time_mask.loc[{'time': time_slice}] = True

        mask = mask & time_mask

        return mask

    def mask_all_nan_basins(self, mask: xr.DataArray, tranges: list[str] | None) -> xr.DataArray:
        if tranges is None:
            raise ValueError('[set]_tranges must not be None.')

        return self.mask_time_slices(
                mask=mask.copy(),
                tranges=tranges).any('time').compute()

    @property
    def num_dfeatures(self) -> int:
        """The number of dynamic features."""
        return len(self.features)

    @property
    def num_sfeatures(self) -> int:
        """The number of static features."""
        if self.stat_features is None:
            return 0
        else:
            return len(self.stat_features)

    @property
    def num_dtargets(self) -> int:
        """The number of dynamic targets."""
        return len(self.targets)
