# from argparse import ArgumentParser
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer

# from project.machflowdata import MachFlowDataModule
# from lstm_regressor import LSTM


# def cli_main():
#     pl.seed_everything(19)

#     # ------------
#     # args
#     # ------------
#     parser = ArgumentParser()
#     parser.add_argument(
#         '--data_path', default='/data/basil/harmonized_basins.zarr/', type=str
#     )
#     parser.add_argument(
#         '--features', nargs='+', default=['P', 'T']
#     )
#     parser.add_argument(
#         '--stat_features', nargs='+', default=['P_mean', 'P_std', 'P_mon_std', 'T_mean', 'T_std', 'T_mon_std']
#     )
#     parser.add_argument(
#         '--targets', nargs='+', default=['Qmm']
#     )
#     parser.add_argument(
#         '--num_samples_per_epoch', default=5, type=int
#     )
#     parser.add_argument(
#         '--batch_size', default=32, type=int
#     )
#     parser.add_argument(
#         '--num_workers', default=0, type=int
#     )
#     parser = Trainer.add_argparse_args(parser)
#     parser = LSTM.add_model_specific_args(parser)
#     args = parser.parse_args()

#     # ------------
#     # data
#     # ------------
#     datamodule = MachFlowDataModule(
#         machflow_data_path=args.data_path,
#         features=args.features,
#         stat_features=args.stat_features,
#         targets=args.targets,
#         train_window_size=1000,
#         window_min_count=1,
#         train_num_samples_per_epoch=args.num_samples_per_epoch,
#         warmup_size=200,
#         train_date_slice=slice('1961-01-01', '1999-12-31'),
#         valid_date_slice=slice('2000-01-01', '2009-12-31'),
#         test_date_slice=slice('2010-01-01', '2023-04-30'),
#         predict_date_slice=slice(None, None),
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         seed=1
#     )

#     # ------------
#     # model
#     # ------------
#     model = LSTM.from_argparse_args(args)

#     # ------------
#     # training
#     # ------------
#     trainer = pl.Trainer.from_argparse_args(args)
#     #trainer.fit(model, train_loader, val_loader)

#     # ------------
#     # testing
#     # ------------
#     #trainer.test(test_dataloaders=test_loader)


# if __name__ == '__main__':
#     cli_main()



from pytorch_lightning.cli import LightningCLI

from project.machflowdata import MachFlowDataModule
from project.lstm_regressor import LSTM


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.num_sfeatures', 'model.num_static_in', apply_on='instantiate')
        parser.link_arguments('data.num_dfeatures', 'model.num_dynamic_in', apply_on='instantiate')
        parser.link_arguments('data.num_dtargets', 'model.num_outputs', apply_on='instantiate')
        parser.link_arguments('data.norm_args_features', 'model.norm_args_features', apply_on='instantiate')
        parser.link_arguments('data.norm_args_stat_features', 'model.norm_args_stat_features', apply_on='instantiate')


def cli_main():
    cli = MyLightningCLI(LSTM, MachFlowDataModule)


if __name__ == '__main__':
    cli_main()
