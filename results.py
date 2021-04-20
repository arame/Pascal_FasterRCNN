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
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(chart_path)


def save_ave_mAP_per_epoch_chart(ave_mAP):
    # save the chart of the average mAP (mean average precision) per epoch
    OutputStore.check_folder(OutputStore.chart_path_pascal)
    chart_path = os.path.join(OutputStore.chart_path_pascal, OutputStore.ave_mAP_filename)
    epochs = np.arange(1, len(ave_mAP) + 1)
    plt.title(f"Average mAP per epoch")
    plt.plot(epochs, ave_mAP)
    plt.ylabel('Average mAP')
    plt.xlabel('Epoch')
    plt.savefig(chart_path)
