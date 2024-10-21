import optuna
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import load_config_xval_test_set
from utils.metrics import compute_metrics
from utils.plotting import load_default_mpl_config, savefig
from config import get_path_dict

load_default_mpl_config()

paths = get_path_dict()

def config_to_tune_path(df: pd.DataFrame) -> pd.DataFrame:

    for i in range(len(df)):
        el = df.iloc[i, :]

        s = []
        if el['static'] == 'all':
            s += ['staticall']
        elif el['static'] == 'dred':
            s += ['staticdred']

        if el['allbasins']:
            s += ['allbasins']

        if el['sqrttrans']:
            s += ['sqrttrans']

        if len(s) == 0:
            s = ['default']

        config = "_".join(s)
        model = el['model']

        optuna_path = f'sqlite:///{paths["runs"]}{config}/{model}/tune/optuna.db'
        study = optuna.load_study(study_name=model, storage=optuna_path)

        for k, v in study.best_params.items():
            if k not in df.columns:
                df[k] = ''

            df.loc[i, k] = v

        summary_path = f'{paths["runs"]}{config}/{model}/xval/fold_000/model_summary.txt'

        if 'num_params' not in df.columns:
            df['num_params'] = ''

        with open(summary_path) as f:
            for line in f:
                if 'Trainable params' in line:
                    num_params = line.split('Trainable params')[0].strip()
                    if 'M' in num_params:
                        num_params = float(num_params.split('M')[0].strip()) * 1000
                    elif 'K' in num_params:
                        num_params = float(num_params.split('K')[0].strip())
                    else:
                        raise ValueError('value is neither K nor M.')

                    num_params = f'{num_params:6.0f} K'
                    df.loc[i, ['num_params']] = num_params

                    break

    return df


xval_ds = load_config_xval_test_set(
    path=paths['runs'],
    nonbool_kwords=['static'],
    time_slices=['1995,1999', '2016,2020']).drop_vars('tau')

met_ds = compute_metrics(
    obs=xval_ds.Qmm,
    mod=xval_ds.Qmm_mod,
    metrics='all',
    dim='time').median(dim='station').compute()

xval_df = met_ds.nse.to_dataframe().reset_index()
xval_df = config_to_tune_path(xval_df)
xval_df = xval_df.replace('none', 'area')

xval_df_nice = xval_df.sort_values(by='nse', ascending=False).reset_index().drop(columns='index').rename(
    columns={
            'lr': 'learning_rate',
            'nse': 'NSE',
        }
    )
xval_df_nice.index = xval_df_nice.index + 1
xval_df_nice.index.name = 'Rank'
xval_df_nice['NSE'] = [f'{val:0.2f}' for val in xval_df_nice['NSE']]

xval_df_lstm = xval_df_nice.loc[xval_df_nice.model=='LSTM',:].drop(columns='model')
xval_df_tcn = xval_df_nice.loc[xval_df_nice.model=='TCN',:].drop(columns='model')

xval_df_lstm = xval_df_lstm.rename(columns={'lstm_layers': 'temp_layers'})
xval_df_tcn = xval_df_tcn.rename(columns={'tcn_layers': 'temp_layers', 'tcn_kernel_size': 'kernel_size'})

nse_col = 'tab:gray'
setup_col = 'w'
mod_col = 'tab:olive'
opt_col = 'tab:pink'
param_col = 'tab:cyan'

col_props = {
    'NSE': nse_col,
    'allbasins': setup_col,
    'sqrttrans': setup_col,
    'static': setup_col,
    'model_dim': mod_col,
    'enc_dropout': mod_col,
    'fusion_method': mod_col,
    'temp_layers': mod_col,
    'kernel_size': mod_col,
    'learning_rate': opt_col,
    'weight_decay': opt_col,
    'num_params': param_col
}

col_props_tcn = col_props.copy()
col_props.pop('kernel_size')

fig, ax = plt.subplots(figsize=(8, 2))
ax.set_axis_off()

tab = pd.plotting.table(
    ax=ax,
    data=xval_df_lstm[col_props.keys()],
    loc='center',
    cellLoc='right',
    colColours=list(col_props.values()),
    # edges='horizontal'
)

w, h = tab[0, 1].get_width(), tab[0, 1].get_height()
tab.add_cell(0, -1, w, h, text=xval_df_lstm.index.name)

tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.auto_set_column_width(col=list(range(len(xval_df_lstm[col_props.keys()].columns))))


savefig(fig, paths['figures'] / 'tabA01.pdf')

fig, ax = plt.subplots(figsize=(8, 2))
ax.set_axis_off()

tab = pd.plotting.table(
    ax=ax,
    data=xval_df_tcn[col_props_tcn.keys()],
    loc='center',
    cellLoc='center',
    colColours=list(col_props_tcn.values()),
    # edges='horizontal'
)

w, h = tab[0, 1].get_width(), tab[0, 1].get_height()
tab.add_cell(0, -1, w, h, text=xval_df_tcn.index.name)

tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.auto_set_column_width(col=list(range(len(xval_df_tcn[col_props_tcn.keys()].columns))))

savefig(fig, paths['figures'] / 'tabA02.pdf')
