import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(csv_path, fig_path):

    history_df = pd.read_csv(csv_path)
    alpha = history_df['alpha'][0]
    alpha_q = history_df['q_alpha'][0]
    q_weight_seed = history_df['q_weight_seed'][0]

    fig, ax=plt.subplots(2, figsize=(12, 10))

    ax[0].plot(history_df['epoch'], history_df['train_acc'])
    ax[0].plot(history_df['epoch'], history_df['val_acc'])
    ax[0].legend(['train_acc', 'val_acc'])
    ax[0].grid()
    ax[0].set_title(f'Accuracies vs epoch number \n alpha:{alpha} | alpha_q:{alpha_q} | q_weight_seed: {q_weight_seed} ')


    ax[1].plot(history_df['epoch'], -history_df['train G'])
    ax[1].legend(['Train Loss'])
    ax[1].set_title(f'Training loss vs epoch number')
    ax[1].grid()

    plt.savefig(fig_path)
