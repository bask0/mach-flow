
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, default_collate
import pytorch_lightning as pl
import numpy as np
from dataclasses import dataclass, fields, is_dataclass

from typing import Any


@dataclass
class Coords:
    station: str
    start_time: str
    end_time: str


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
            window_size: int = -1,
            window_min_count: int = -1,
            warmup_size: int = 0,
            num_samples_per_epoch: int = 1,
            seed: int = 19):

        self.ds = ds
        self.features = features
        self.targets = targets

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
        data_t = self.get_targets(station=station)

        target_rolling_mask = self.get_target_rolling_mask(station=station)

        if target_rolling_mask is not None:
            indices = np.argwhere(target_rolling_mask.values)[:, 0]
            if len(indices) < 1:
                raise RuntimeError(
                    f'Station #{station} has no window with window_size={self.window_size} with at least '
                    f'window_min_count={self.window_min_count} valid time steps.'
                )

            time_end = self.rs.choice(indices) + 1
            time_start = time_end - self.window_size - self.warmup_size

            data_f = data_f.isel(time=slice(time_start, time_end))
            data_t = data_t.isel(time=slice(time_start, time_end))

        else:
            time_start = 0
            time_end = -1

        return_data = BatchPattern(
            dfeatures=data_f.values.astype('float32'),
            dtargets= data_t.values.astype('float32'),
            coords=Coords(
                station=self.ds.station.isel(station=station).item(),
                start_time=self.ds.time[time_start].dt.strftime('%Y-%m-%d').item(),
                end_time=self.ds.time[time_start - 1].dt.strftime('%Y-%m-%d').item()
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

    def get_targets(self, **isel) -> xr.DataArray:
        return self.ds[self.targets].isel(**isel).to_array('variable')


class MachFlowDataModule(pl.LightningDataModule):
    """Describes a MACH-Flow data modukle.

    This MachFlowDataModule builds upon the pl.LightningDataModule. It can be used to define and retrieve the
    training, validation, test, and prediction dataloaders and integrates with the pl.Trainer framework.

    For training, a random subset of the time-series of length `train_window_size` is sampled from the basins, but
    only for windows with at least `window_min_count` observations of a target variable present. For each basin,
    `train_num_samples_per_epoch` random windows are loaded in an epoch.

    A warmup periode of length `warmup_size` is added for model spin-up. Turn off this behavior by setting
    `warmup_size=0`. If `drop_all_nan_stations` is set to True, sites with all NaN targets are dropped entirely
    from training, validation, and test.

    Shapes:
        The dataloaders return a `BatchPattern` with elements/shapes:
        - Dynamic features (dfeatures): (batch_size, num_dfeatures, seq_length,)
        - Static features (sfeatures, optional): (batch_size, num_stargets,)
        - Dynamic targets (dtargets): (batch_size, num_dtargets, seq_length,)
        - Coords (coords):
            - Station ID (station): (batch_size,)
            - Window start date including warmup (start_time): (batch_size,)
            - Window end date (end_time): (batch_size,)

    """
    def __init__(
            self,
            machflow_data_path: str,
            features: list[str],
            targets: list[str],
            train_window_size: int = 1000,
            window_min_count: int = 1,
            train_num_samples_per_epoch: int = 1,
            warmup_size=365,
            drop_all_nan_stations: bool = True,
            batch_size: int = 10,
            num_workers: int = 10,
            train_val_test_split: tuple[int, int, int] = (60, 20, 20),
            seed: int = 19):
        """Initialize MachFlowDataModule.

        Args:
            machflow_data_path (str): Path to machflow zarr file.
            features (list[str]): A list of features.
            targets (list[str]): A list of targets.
            train_window_size (int, optional): The training window number of time steps. Defaults to 1000.
            window_min_count (int, optional): Minimum number of target observations in time window. Defaults to 1.
            train_num_samples_per_epoch (int, optional): Number of samples of size `train_window_size` to draw from
                each station during training. If 10, for example, 10 random windows will be used in training for each
                station. Defaults to 1.
            warmup_size (int, optional): The number of warmup staps to use for model spinup. Defaults to 365.
            drop_all_nan_stations (bool, optional): If True, only stations with at least one observation are used.
                Applies for all modes but 'prediction'. Defaults to True.
            batch_size (int, optional): The minibatch batch size. Defaults to 10.
            num_workers (int, optional): The number of workers. Defaults to 10.
            train_val_test_split (tuple[int, int, int]): The training, validation, test proportions. Defaults
                to ()
            seed (int, optional): The random seed. Defaults to 19.
        """

        self.machflow_data_path = machflow_data_path
        self.features = features
        self.targets = targets
        self.train_window_size = train_window_size
        self.window_min_count = window_min_count
        self.train_num_samples_per_epoch = train_num_samples_per_epoch
        self.warmup_size = warmup_size
        self.drop_all_nan_stations = drop_all_nan_stations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Lazy-load zarr dataset.
        self.ds = xr.open_zarr('/data/basil/harmonized_basins.zarr/')

        # Save all basins for prediction mode.
        self.all_basins = self.ds.stations.values

        # Drop all stations with all-NaN targets.
        if drop_all_nan_stations:
            mask = self.ds[self.targets].to_array('variable').notnull().any('variable')
            stations = mask.station.values[mask.any('time')]
        else:
            stations = self.ds.station.values

        # Create random state with seed.
        self.rs = np.random.RandomState(seed)

        # Split basins into training, validation, and test set.
        self.train_basins, self.val_basins, self.test_basins = self.split_basins(
            basins=stations,
            rs=self.rs,
            train_frac=train_val_test_split[0],
            val_frac=train_val_test_split[1],
            test_frac=train_val_test_split[2])
        self.predict_basins = self.ds.stations.values

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
        elif mode == 'val':
            window_size = -1
            window_min_count = self.window_min_count
            num_samples_per_epoch = 1
            basins = self.val_basins
        elif mode == 'test':
            window_size = -1
            window_min_count = self.window_min_count
            num_samples_per_epoch = 1
            basins = self.test_basins
        elif mode == 'predict':
            window_size = -1
            window_min_count = -1
            num_samples_per_epoch = 1
            basins = self.predict_basins
        else:
            raise ValueError(
                f'mode \'{mode}\' not valid, must be one of \'train\', \'val\', \'test\', or \'predict\'.'
            )

        dataset = MachFlowData(
            ds=self.ds.sel(station=basins),
            features=self.features,
            targets=self.targets,
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
            basins: list[str],
            rs: np.random.RandomState,
            train_frac: float,
            val_frac: float,
            test_frac: float) -> tuple[list[str], list[str], list[str]]:
        """Split and return the basins/statinos.

        Args:
            basins (list[str]): A list of basin IDs.
            train_frac (float): Training set size relative to validation and test set.
            val_frac (float): Validation set size relative to training and test set.
            test_frac (float): Test set size relative to training and test validation.

        Returns:
            tuple[list[str], list[str], list[str]]: The training, validation, and test basin IDs.
        """
        frac_sum = train_frac + val_frac + test_frac
        train_frac /= frac_sum
        val_frac /= frac_sum
        test_frac /= frac_sum

        num_basins = len(basins)
        num_train_basins = int(num_basins * train_frac)
        num_val_basins = int(num_basins * val_frac)

        basins = list(rs.permutation(basins))

        train_basins = basins[:num_train_basins]
        val_basins = basins[num_train_basins:num_train_basins+num_val_basins]
        test_basins = basins[num_train_basins+num_val_basins:]

        return train_basins, val_basins, test_basins
