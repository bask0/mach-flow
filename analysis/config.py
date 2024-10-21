
from pathlib import Path

def get_path_dict() -> dict[str, Path]:
    path_dict = dict(
        figures=Path('/net/argon/landclim/kraftb/machflow/mach-flow/analysis/figures/'),
        figures_alt=Path('/net/argon/landclim/kraftb/machflow/mach-flow/analysis/helper_figures/'),
        runs=Path('/net/argon/landclim/kraftb/machflow/runs/'),
        data=Path('/net/argon/landclim/kraftb/machflow/data/'),
    )

    return path_dict
