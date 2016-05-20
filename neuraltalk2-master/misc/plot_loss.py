import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import argparse

def main(**kwargs):
    checkpoint_path = kwargs.get('checkpoint_path')
    ids = kwargs.get('ids').split(',')
    output_jsons = [checkpoint_path + 'model_id' + id + '.json' for id in ids]
    checkpoints = [json.load(open(f, 'r')) for f in output_jsons]

    train_loss = [cp['loss_history'] for cp in checkpoints]
    val_loss = [cp['val_loss_history'] for cp in checkpoints]
    val_scores = [cp['val_lang_stats_history'] for cp in checkpoints]

    colors = ['g', 'r', 'c', 'm']
    linestyles = ['-', '--', ':']
    plot_train_full(train_loss, ids)
    plot_train_val(train_loss, val_loss, ids, colors)
    metrics = ['Bleu_4', 'CIDEr', 'METEOR']
    plot_scores(val_scores, metrics, ids, colors, linestyles)

def plot_train_full(train_loss, ids):
    img = '../plots/train_loss' + ''.join(ids) + '.jpg'
    lines = []
    fig = plt.figure()
    for loss in train_loss:
        keys = np.array(loss.keys(), dtype=int)
        vals = np.array(loss.values())
        indices = np.argsort(keys)
        keys, vals = keys[indices], vals[indices]
        id = ids[train_loss.index(loss)]
        line, = plt.plot(keys, vals, label='model'+id)
        lines.append(line)
    plt.legend(handles=lines, loc='upper right')
    plt.xlabel('Num. Iterations'), plt.ylabel('Training Loss')
    plt.title('Training Loss'), plt.savefig(img)

def plot_train_val(train_loss, val_loss, ids, c):
    img = '../plots/train_val_loss' + ''.join(ids) + '.jpg'
    lines = []
    fig = plt.figure()
    for id in ids:
        i = ids.index(id)
        keys = np.array(val_loss[i].keys(), dtype=int)
        val_vals = np.array(val_loss[i].values())
        indices = np.argsort(keys)
        keys, val_vals = keys[indices], val_vals[indices]
        train_vals = [train_loss[i][str(key).decode("utf-8")] for key in keys]
        line_val, = plt.plot(keys, val_vals, color=c[i], linestyle='--', label='model'+id+' Val')
        line_train, = plt.plot(keys, train_vals, color=c[i], label='model'+id+' Train')
        lines.append(line_val), lines.append(line_train)
    plt.legend(handles=lines, loc='upper right')
    plt.xlabel('Num. Iterations'), plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss'), plt.savefig(img)

def plot_scores(val_scores, metrics, ids, c, ls):
    img = '../plots/val_scores' + ''.join(ids) + '.jpg'
    lines = []
    fig = plt.figure()
    for id in ids:
        i = ids.index(id)
        keys = np.sort(np.array(val_scores[i].keys(), dtype=int)) # num. iteration
        for metric in metrics:
            j = metrics.index(metric)
            scores = [val_scores[i][str(key).decode("utf-8")][metric.decode("utf-8")] for key in keys]
            line, = plt.plot(keys, scores, color=c[i], linestyle=ls[j], label='model'+id+' '+metric)
            lines.append(line)
    m = len(metrics)
    l1 = plt.legend(handles=lines[:m], loc='center right', prop={'size': 12})
    ax = plt.gca().add_artist(l1)
    plt.legend(handles=lines[m:], loc='lower right', prop={'size': 12})
    plt.xlabel('Num. Iterations'), plt.ylabel('Score')
    plt.title('Validation Scores'), plt.savefig(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', nargs='?', default='/mnt0/data/img-cap/models/', type=str)
    parser.add_argument('-ids', nargs='?', default='0', type=str) # ids separated by ','
    args = parser.parse_args()
    main(**vars(args))
