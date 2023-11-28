from argparse import ArgumentParser
from glob import glob
import os
import shutil

from utils.analysis import plot_model_comp
from utils.logging import get_logger

if __name__ == '__main__':
    logger = get_logger('main')

    parser = ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str, default='./runs'
    )
    parser.add_argument(
        '--target', type=str, default='Qmm'
    )
    parser.add_argument(
        '--benchmark', type=str, default='prevah'
    )

    args = parser.parse_args()

    for path in glob(os.path.join(args.log_dir, '*')):
        logger.info(f'Analyzing \'{path}\'.')

        out_dir = os.path.join(path, 'results')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        logger.info(f'  > Saving results to \'{out_dir}\'.')

        model_comp_plot_path = os.path.join(out_dir, 'model_comp.png')
        plot_model_comp(dir=path, target=args.target, ref=args.benchmark, save_path=model_comp_plot_path)

        logger.info(f'  > Done.')
