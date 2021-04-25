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


def save_class_metrics(class_name, precisions, recalls):
    # chart_class_path
    OutputStore.check_folder(OutputStore.chart_class_path)
    filename = class_name + OutputStore.class_chart_filename
    path = os.path.join(OutputStore.chart_class_path, filename)
    plt.title(f"{class_name} Precision-Recall Curve")
    plt.plot(recalls, precisions)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(path)
    plt.clf()

def save_combined_class_metrics(select_class_results_dict):
    # chart_class_path
    OutputStore.check_folder(OutputStore.chart_class_path)
    list = []
    for key in select_class_results_dict:
        list.append(key)
    names = ', '.join(list)
    filename = "combined_" + OutputStore.class_chart_filename
    path = os.path.join(OutputStore.chart_class_path, filename)
    plt.title(f"{names} Precision-Recall Curve")
    for key, value in select_class_results_dict.items():
        plt.plot(value.recalls, value.precisions, label = key)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend()
    plt.savefig(path)
    plt.clf()