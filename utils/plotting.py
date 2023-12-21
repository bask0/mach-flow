
def load_default_mpl_config():
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 140
    plt.rcParams['savefig.dpi'] = 300

def savefig(fig, path, tight=True, **kwargs) -> None:
    fig.tight_layout()
    fig.savefig(path, **kwargs)
