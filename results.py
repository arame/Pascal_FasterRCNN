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
    plt.clf()


def save_ave_MAP_per_epoch_chart(ave_MAP):
    # save the chart of the average MAP (mean average precision) per epoch
    OutputStore.check_folder(OutputStore.chart_path_pascal)
    chart_path = os.path.join(OutputStore.chart_path_pascal, OutputStore.ave_MAP_filename)
    epochs = np.arange(1, len(ave_MAP) + 1)
    plt.title(f"Average MAP per epoch")
    plt.plot(epochs, ave_MAP)
    plt.ylabel('Average MAP')
    plt.xlabel('Epoch')
    plt.savefig(chart_path)
    plt.clf()
    

def save_ave_overlaps_per_epoch_chart(ave_overlaps):
    # save the chart of the average MAP (mean average precision) per epoch
    OutputStore.check_folder(OutputStore.chart_path_pascal)
    chart_path = os.path.join(OutputStore.chart_path_pascal, OutputStore.ave_IOU_filename)
    epochs = np.arange(1, len(ave_overlaps) + 1)
    plt.title(f"Average IOU per epoch")
    plt.plot(epochs, ave_overlaps)
    plt.ylabel('Average IOU')
    plt.xlabel('Epoch')
    plt.savefig(chart_path)
    plt.clf()
