import matplotlib.pyplot as plt
import numpy as np
import os
from config import OutputStore


def save_loss_per_epoch_chart(losses):
    # save the chart of the losses per epoch
    OutputStore.check_folder(OutputStore.chart_path_pascal)
    chart_path = os.path.join(OutputStore.chart_path_pascal, OutputStore.loss_filename)
    epochs = np.arange(1, len(losses) + 1)
    plt.title(f"Loss per epoch")
    plt.plot(epochs, losses)
    plt.ylabel('Epoch')
    plt.xlabel('Loss')
    plt.savefig(chart_path)
