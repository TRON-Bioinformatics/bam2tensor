import matplotlib.pyplot as plt
import seaborn as sns

FREQUENCY_INDICES = [
    'Frequency',
    'Base quality',
    'Mapping quality',
    'Coverage(-)',
    'Frequency (-)',
    'Coverage(+)',
    'Frequency (+)',
    'Edit distance',
    'Unique',
    'Frequency (Confident)',
    'Coverage (Confident)'
]


def visualize(tensor, impath=None):
    """Visualize a given tensor

    :param tensor: Tensor to visualize
    """
    fig = plot_paired(tensor, FREQUENCY_INDICES, 24, 0, 1, 'Blues')
    plt.subplots_adjust(hspace=0.5)
    if not impath:
        plt.show()
    else:
        fig.savefig(impath, dpi=300)


def plot_paired(tensor, indices, num_channels, vmin, vmax, cmap):
    fontsize = 8
    sns.set(font_scale=0.5)

    if tensor.shape[0] != num_channels:
        raise Exception(
            'Number of channels in the tensor: {} '
            'Expected number of channels: {}'.format(
                tensor.shape[0], num_channels
            ))

    paired_start = 2
    middle_point = int(num_channels / 2) + 1
    axis_norm1 = paired_start - 1
    axis_norm2 = middle_point - 1

    fig, ax = plt.subplots(
        int(num_channels / 2),
        2,
        sharex=True,
        sharey=True,
        figsize=(8, 11)
    )

    sns.heatmap(tensor[0, :, :], cmap=cmap, ax=ax[0, 0], vmin=vmin,
                vmax=vmax)
    ax[0, 0].set_title('Reference')

    sns.heatmap(tensor[1, :, :], cmap=cmap, ax=ax[0, 1], vmin=vmin,
                vmax=vmax)
    ax[0, 1].set_title('Position')


    for i in range(paired_start, middle_point):
        sns.heatmap(tensor[i, :, :], cmap=cmap, ax=ax[i - axis_norm1, 0],
                    vmin=vmin, vmax=vmax)
        ax[i - axis_norm1, 0].set_title(
            'Tumor-{}'.format(indices[i - paired_start]),
        )

    for i in range(middle_point, num_channels):
        sns.heatmap(tensor[i, :, :], cmap=cmap, ax=ax[i - axis_norm2, 1],
                    vmin=vmin, vmax=vmax)
        ax[i - axis_norm2, 1].set_title(
            'Normal-{}'.format(indices[i - middle_point]),
        )
        ax[i - axis_norm2, 1].set_yticks(
            range(5),
            labels=['-   ', ' A   ', ' C    ', ' G    ', 'T   ']
        )

    return fig
