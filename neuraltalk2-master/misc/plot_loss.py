import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import argparse

def main(**kwargs):
    checkpoint_path = kwargs.get('checkpoint_path')
    id = kwargs.get('id')
    output_json = checkpoint_path + 'model_id' + str(id) + '.json'
    checkpoint = json.load(open(output_json, 'r'))

    train_loss = checkpoint['loss_history']
    val_loss = checkpoint['val_loss_history']
    val_scores = checkpoint['val_lang_stats_history']

    plot_train_full(train_loss, id)
    plot_train_val(train_loss, val_loss, id)
    metrics = ['Bleu_4', 'CIDEr', 'METEOR']
    plot_scores(val_scores, metrics, id)

def plot_train_full(train_loss, id):
    img = '../plots/train_loss' + str(id) + '.jpg'
    keys = np.array(train_loss.keys(), dtype=int)
    vals = np.array(train_loss.values())
    indices = np.argsort(keys)
    keys, vals = keys[indices], vals[indices]
    fig = plt.figure(), plt.plot(keys, vals)
    plt.xlabel('Num. Iterations'), plt.ylabel('Training Loss')
    plt.title('Training Loss'), plt.savefig(img)

def plot_train_val(train_loss, val_loss, id):
    img = '../plots/train_val_loss' + str(id) + '.jpg'
    keys = np.array(val_loss.keys(), dtype=int)
    val_vals = np.array(val_loss.values())
    indices = np.argsort(keys)
    keys, val_vals = keys[indices], val_vals[indices]
    train_vals = [train_loss[str(key).decode("utf-8")] for key in keys]

    lines = []
    fig = plt.figure()
    line_val, = plt.plot(keys, val_vals, label='Validation')
    line_train, = plt.plot(keys, train_vals, label='Train')
    lines.append(line_train), lines.append(line_val)
    plt.legend(handles=lines, loc='upper right')
    plt.xlabel('Num. Iterations'), plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss'), plt.savefig(img)

def plot_scores(val_scores, metrics, id):
    img = '../plots/val_scores' + str(id) + '.jpg'
    keys = np.sort(np.array(val_scores.keys(), dtype=int)) # num. iteration

    lines = []
    fig = plt.figure()
    for metric in metrics:
        scores = [val_scores[str(key).decode("utf-8")][metric.decode("utf-8")] for key in keys]
        line, = plt.plot(keys, scores, label=metric)
        lines.append(line)
    plt.legend(handles=lines, loc='lower right', prop={'size': 13})
    plt.xlabel('Num. Iterations'), plt.ylabel('Score')
    plt.title('Validation Scores'), plt.savefig(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', nargs='?', default='/mnt0/data/img-cap/models/', type=str)
    parser.add_argument('-id', nargs='?', default=0, type=int)
    args = parser.parse_args()
    main(**vars(args))
