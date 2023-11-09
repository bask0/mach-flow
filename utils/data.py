import pyreadr
import pandas as pd


def read_rds(path: str) -> pd.DataFrame:
    return pyreadr.read_r(path)[None]
