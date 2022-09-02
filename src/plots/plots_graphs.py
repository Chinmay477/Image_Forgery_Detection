import matplotlib.pyplot as plt
import pandas as pd


def plot_epochs(metric1, ylab):

    plt.plot(metric1)
    plt.ylabel(ylab)
    plt.xlabel("Epoch")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    df1 = pd.read_csv(filepath_or_buffer="../../src/SRM_accuracy.csv")
    df3 = pd.read_csv(filepath_or_buffer="../../src/SRM_loss.csv")
    plot_epochs(df1.iloc[:, 1], 'Training Accuracy')
    plot_epochs(df3.iloc[:, 1], 'Training Loss')
