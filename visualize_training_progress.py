import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('names', nargs='+')
args = parser.parse_args()

fig_idx = 0
for file_name in args.names:
    name = os.path.splitext(file_name)[0]
    if not os.path.isfile(file_name):
        print(f"file {file_name} does not exist")

    f = open(file_name, "r")
    lines = f.readlines()

    semseg_mious_eval = []
    semseg_mious_train = []
    normals_rmse_eval = []
    normals_rmse_train = []
    percentage_executed_layers = []

    model_loss = []
    efficiency_loss = []
    lines = [line.split() for line in lines]

    for i, line in enumerate(lines):
        if len(line) > 0 and line[0] == "Semantic":
            if lines[i-2][0] != "Training":
                semseg_mious_eval.append(float(line[-1]))
            else:
                semseg_mious_train.append(float(line[-1]))

        if len(line) > 0 and line[0] == "rmse":
            if lines[i-27][0] != "Training":
                normals_rmse_eval.append(float(line[-1]))
            else:
                normals_rmse_train.append(float(line[-1]))

        if len(line) > 5 and line[5] == "%" and line[6] == "Activated":
            percentage_executed_layers.append(float(line[-1][:-1]))

        if len(line) > 6 and line[5] == "model" and line[6] == "loss":
            model_loss.append(float(line[7]))
            efficiency_loss.append(float(line[11]))

    # visualize accuracy
    assert len(semseg_mious_eval) == len(normals_rmse_eval) and len(
        normals_rmse_eval) == len(percentage_executed_layers)
    assert len(semseg_mious_train) == len(normals_rmse_train)

    fig = plt.figure(fig_idx)
    fig_idx += 1

    ax = fig.add_subplot(111)

    xs_eval = np.arange(len(semseg_mious_eval))
    ax.plot(xs_eval, semseg_mious_eval,
            c='forestgreen', label="semseg mIoU (eval)")
    ax.plot(xs_eval, normals_rmse_eval, c='royalblue',
            label="normals rmse (eval)")

    xs_eval = np.arange(1, len(semseg_mious_eval))
    ax.plot(xs_eval, semseg_mious_train,
            c='lightgreen', label="semseg mIoU (train)")
    ax.plot(xs_eval, normals_rmse_train, c='lightskyblue',
            label="normals rmse (train)")

    ax2 = ax.twinx()
    xs_eval = np.arange(len(semseg_mious_eval))
    plt.plot(xs_eval, percentage_executed_layers, c='red', label="% FLOPS")

    ax2.set_ylabel(r"FLOPS (%)")

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.25), ncol=2, fontsize=10)
    ax2.legend(loc=1)
    plt.grid()
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{name}_accuracy.png")

    fig = plt.figure(fig_idx)
    fig_idx += 1

    xs = np.arange(len(model_loss))

    ax = fig.add_subplot(111)
    ax.plot(xs, model_loss, c="r", label="model loss")
    ax.legend(loc=1)

    # Creating Twin axes for dataset_1
    ax2 = ax.twinx()
    ax2.plot(xs, efficiency_loss, c="b", label="efficiency loss")

    ax2.legend(loc=2)

    ax.grid()
    plt.title(name)
    plt.savefig(f"{name}_loss.png")
