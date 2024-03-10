
def load_default_mpl_config():
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 500

def savefig(fig, path, tight=True, transparent=True, **kwargs) -> None:
    if tight:
        fig.tight_layout()
        kwargs['bbox_inches'] = 'tight'

    if transparent:
        kwargs['transparent'] = True

    fig.savefig(path, **kwargs)
