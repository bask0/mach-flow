import geopandas as gpd
import os
from pathlib import Path

from utils.data import load_xval

# BEST_MODEL_PATH = '/mydata/machflow/basil/runs/basin_level/staticall_allbasins_sqrttrans/LSTM/'
SHAPEFILE_PATH = Path('/mydata/machflow/shared/data/RawFromMichael/prevah_307/shapefile/')
BEST_MODEL_PATH = Path('/mydata/machflow/basil/runs/basin_level/staticall_allbasins_sqrttrans/LSTM/')
BEST_MODEL_XVAL_PATH = BEST_MODEL_PATH / 'xval/'
OUT_PATH = Path('/mydata/machflow/basil/runs/basin_level/CHRUN/')

CHRUN_OUT_PATH = OUT_PATH / 'chrun.nc'
CATCHMENTS_OUT_PATH = OUT_PATH / 'catchments/'
README_OUT_PATH = OUT_PATH / 'readme.txt'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

static_names = dict(
    water=1,
    urban_areas=2,
    coniferous_forests=3,
    deciduous_forests=4,
    mixed_forests=5,
    cereals=6,
    pasture=7,
    bush=8,
    unknown=10,
    firn=13,
    bare_ice=14,
    rock=15,
    fruits=18,
    vegetables=19,
    wheat=20,
    alpine_vegetation=21,
    wetlands=22,
    rough_pasture=23,
    sub_Alpine_meadow=24,
    alpine_meadow=25,
    bare_soil_vegetation=26,
    corn=28,
    grapes=29,
)

ds = load_xval(BEST_MODEL_XVAL_PATH, sources=0).median('cv').isel(tau=0).drop([
    'tau',
    'E',
    'Qmm',
    'Qm3s',
    'Qm3s_prevah',
    'c0',
    'c1',
    'c2',
    'c3',
    'c4',
    'cv_set',
    'folds'
]
).sel(time=slice('1962', '2023'))

ds = ds.load()

global_attrs = dict(
    title='CH-RUN: A data-driven spatially contiguous runoff monitoring product for Switzerland',
    institution='IAC, ETH Zurich',
    source='CH-RUN',
    comment='Modeled runoff product (CH-RUN): Qmm_mod, PREVAH runoff: Qmm_prevah.',
    references='INSERT HESS PAPER',
    version_number='1.0',
    contact='Basil Kraft (basilkraft@env.ethz.ch), Lukas Gudmundsson (lukas.gudmundsson@env.ethz.ch)'
)

default_attrs = dict(
    history='Aggregated to catchment level using average.',
    units='-',
)

variable_attrs = {
    'P': dict(
        title='Precipitation',
        source='MeteoSwiss',
        references='MeteoSwiss: Daily Precipitation (Final Analysis): RhiresD, https://www.meteoschweiz.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-5666327e63c/ProdDoc_RhiresD.pdf, last access: 1 April 2024.',
        long_name='Precipitation',
        units='mm d-1'
    ),
    'T': dict(
        title='Precipitation',
        source='MeteoSwiss',
        references='MeteoSwiss: Daily Mean, Minimum and Maximum Temperature: TabsD, TminD, TmaxD, https://www.meteoschweiz.admin.ch/dam/jcr:818a4d17-cb0c-4e8b-92c6-1a1bdf5348b7/ProdDoc_TabsD.pdf, last access: 1 April 2024.',
        long_name='Air temperature',
        units='degC'
    ),
    'Qmm_mod': dict(
        title='CH-RUN runoff',
        source='CH-RUN v1.0',
        references='INSERT HESS PAPER',
        long_name='Runoff (CH-RUN)',
        units='mm d-1'
    ),
    'Qmm_prevah': dict(
        title='PREVAH runoff',
        source='WSL',
        references='1) Viviroli, D., Zappa, M., Schwanbeck, J., Gurtz, J., and Weingartner, R.: Continuous Simulation for Flood Estimation in Ungauged Mesoscale Catchments of Switzerland – Part I: Modelling Framework and Calibration Results, Journal of Hydrology, 377, 191–207, https://doi.org/10.1016/j.jhydrol.2009.08.023, 2009. 2) Viviroli, D., Zappa, M., Gurtz, J., and Weingartner, R.: An Introduction to the Hydrological Modelling System PREVAH and Its Pre- and Post-Processing-Tools, Environmental Modelling & Software, 24, 1209–1222, https://doi.org/10.1016/j.envsoft.2009.04.001, 2009.',
        long_name='Runoff (PREVAH)',
        units='mm d-1'
    ),
    'dhm': dict(
        title='Digital elevation model',
        source='swisstopo',
        references='swisstopo: swissALTI3D, https://www.swisstopo.admin.ch/en/height-model-swissalti3d#Additional-information, last access: 1 April 2024, 2018.',
        long_name='Elevation',
        units='m a.s.l.'
    ),
    'abb': dict(
        title='Soil-topographic index',
        source='Derived',
        references='-',
        long_name='Soil-topographic index',
        units='-'
    ),
    'atb': dict(
        title='Hydraulic-topographic index',
        source='Derived',
        references='-',
        long_name='Hydraulic-topographic index',
        units='-'
    ),
    'slp': dict(
        title='Slope',
        source='Derived',
        references='-',
        long_name='Slope',
        units='deg'
    ),
    'btk': dict(
        title='Soil depth',
        source='1) Soilgrids, 2) Hochaufloesende Bodenkarten für den Schweizer Wald',
        references='1) Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: SoilGrids 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty, SOIL, 7, 217–240, https://doi.org/10.5194/soil-7-217-2021, 2021. 2) Baltensweiler, A., Walthert, L., Zimmermann, S., and Nussbaum, M.: Hochaufloesende Bodenkarten Für Den Schweizer Wald, Schweizerische Zeitschrift fur Forstwesen, 173, 288–291, https://doi.org/10.3188/szf.2022.0288, 2022.',
        long_name='Soil depth',
        units='m'
    ),
    'kwt': dict(
        title='Hydraulic conductivity',
        source='1) Soilgrids, 2) Hochaufloesende Bodenkarten für den Schweizer Wald',
        references='1) Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: SoilGrids 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty, SOIL, 7, 217–240, https://doi.org/10.5194/soil-7-217-2021, 2021. 2) Baltensweiler, A., Walthert, L., Zimmermann, S., and Nussbaum, M.: Hochaufloesende Bodenkarten Für Den Schweizer Wald, Schweizerische Zeitschrift fur Forstwesen, 173, 288–291, https://doi.org/10.3188/szf.2022.0288, 2022.',
        long_name='Hydraulic conductivity',
        units='mm h-1'
    ),
    'pfc': dict(
        title='Soil water storage capacity',
        source='1) Soilgrids, 2) Hochaufloesende Bodenkarten für den Schweizer Wald',
        references='1) Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: SoilGrids 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty, SOIL, 7, 217–240, https://doi.org/10.5194/soil-7-217-2021, 2021. 2) Baltensweiler, A., Walthert, L., Zimmermann, S., and Nussbaum, M.: Hochaufloesende Bodenkarten Für Den Schweizer Wald, Schweizerische Zeitschrift fur Forstwesen, 173, 288–291, https://doi.org/10.3188/szf.2022.0288, 2022.',
        long_name='Soil water storage capacity',
        units='%'
    ),
    'glm': dict(
        title='Glacier morphology',
        source='SGI2016',
        references='Linsbauer, A., Huss, M., Hodel, E., Bauder, A., Fischer, M., Weidmann, Y., Baertschi, H., and Schmassmann, E.: The New Swiss Glacier Inventory SGI2016: From a Topographical to a Glaciological Dataset, Frontiers in Earth Science, 9, https://doi.org/10.3389/feart.2021.704189, 2021.',
        long_name='Glacier morphology',
        units='-'
    ),
    'area': dict(
        title='Catchment area',
        history='',
        source='Derived',
        references='',
        long_name='Catchment area',
        units='m2'
    ),
}

ds.attrs = global_attrs
for var, attrs in variable_attrs.items():
    if not var.startswith('Qmm'):
        attrs = {**default_attrs, **attrs}
    ds[var].attrs = attrs

for k, v in static_names.items():
    pus_name = f'pus{v:02d}'
    name_nice = k.capitalize().replace('_', ' ') + ' fraction'

    attrs = dict(
        title=name_nice,
        source='1) Habitat Map of Switzerland 2) WSL',
        references='1) Price, B., Huber, N., Nussbaumer, A., and Ginzler, C.: The Habitat Map of Switzerland: A Remote Sensing, Composite Approach for a High Spatial and Thematic Resolution Product, Remote Sensing, 15, 643, https://doi.org/10.3390/rs15030643, 2023. 2) Viviroli, D., Zappa, M., Schwanbeck, J., Gurtz, J., and Weingartner, R.: Continuous Simulation for Flood Estimation in Ungauged Mesoscale Catchments of Switzerland – Part I: Modelling Framework and Calibration Results, Journal of Hydrology, 377, 191–207, https://doi.org/10.1016/j.jhydrol.2009.08.023, 2009.',
        long_name=name_nice,
        units='-'
    )

    if pus_name in ds.data_vars:
        pus_name_frac = f'frac_{k}'
        ds = ds.rename({pus_name: pus_name_frac})
        ds[pus_name_frac].attrs = attrs


df = gpd.read_file(SHAPEFILE_PATH)
df['station'] = [f'HSU_{obj:03d}' for obj in df['OBJECTID']]
df = df.drop(columns=['Shape_Leng', 'Shape_Area', 'OBJECTID']).set_index('station')

README = """
CH-RUN V1 --- Kraft et al. (2024)

Datasets
-----------
- catchments: ESRI shapefiles containing the 307 catchments.
- chrun: NetCDF file containing model features and simulated runoff rates by CH-RUN and PREVAH.

Usage
-----------
The catchments provided as shapefile can be linked with the NetCDF data via the 'station' field.

References
-----------
Add HESS  reference.

"""

df.to_file(CATCHMENTS_OUT_PATH)
ds.to_netcdf(CHRUN_OUT_PATH)
with open(README_OUT_PATH, 'w') as f:
    f.write(README)

