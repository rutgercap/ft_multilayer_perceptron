from pathlib import Path
from sys import argv
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import dataset_from_path, normalize_data


def split(path: Path) -> None:
    df = dataset_from_path(path)
    df = normalize_data(df)
    features = df.columns[2:]
    fig, axes = plt.subplots(nrows=len(features), figsize=(8, 6 * len(features)))

    for i, feature in enumerate(features):
        sns.violinplot(x=feature, y='diagnosis', data=df, ax=axes[i])

    plt.tight_layout()
    fig.savefig('my_plot.pdf')


if __name__ == "__main__":
    split(Path("data.csv"))
