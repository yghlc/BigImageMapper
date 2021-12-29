import basic_src.io_function as io_function
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# wall_time_to_relative time
def wall_time_to_relative_time(wall_time_list):
    diff_hours = [(wall_time_list[idx+1] - wall_time_list[idx])/3600 for idx in range(len(wall_time_list) - 1)]
    mean_diff = sum(diff_hours)/len(diff_hours)

    relative_time = [0,mean_diff]
    acc_time = mean_diff
    for idx in range(len(diff_hours)):
        acc_time += diff_hours[idx]
        relative_time.append(acc_time)

    return min(relative_time), max(relative_time)


def plot_miou_step_time(train_dict, val_dict):

    fig, ax = plt.subplots(2, 1, sharey='col', sharex='col', tight_layout=True, figsize=(6, 6), gridspec_kw={'hspace': 0})

    # ax1 = fig.add_subplot(212)
    ax[1].plot(val_dict['step'], val_dict['class_0'], linestyle='-', color='tab:red', label="Background", linewidth=0.8)
    ax[1].plot(val_dict['step'], val_dict['class_1'], linestyle='-', color='tab:blue', label="Rock glaciers", linewidth=0.8)
    ax[1].plot(val_dict['step'], val_dict['overall'], 'k-.', label="Overall", linewidth=0.8)
    ax[1].legend(fontsize=10, loc="lower right")
    ax[1].set_xlim([0, max(val_dict['step'])])
    ax[1].set_xlabel('Training iteration', fontsize=10)
    ax[1].set_ylabel('Validation IoU')
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].grid(axis='both', ls='--', alpha=0.5, lw=0.4, color='grey')

    # ax2 = fig.add_subplot(211, sharex=ax1, sharey=ax1)
    ax[0].plot(train_dict['step'], train_dict['class_0'], linestyle='-', color='tab:red', label="Background", linewidth=0.8)
    ax[0].plot(train_dict['step'], train_dict['class_1'], linestyle='-', color='tab:blue', label="Rock glaciers", linewidth=0.8)
    ax[0].plot(train_dict['step'], train_dict['overall'], 'k-.', label="Overall", linewidth=0.8)
    ax[0].grid(axis='both', ls='--', alpha=0.5, lw=0.4, color='grey')
    ax[0].set_ylabel('Training IoU')

    ax2 = ax[0].twiny()    #have another x-axis for time
    min_t, max_t = wall_time_to_relative_time(train_dict['wall_time'])
    ax2.set_xlim([min_t, max_t])
    ax2.set_xlabel("Training time (hours)", fontsize=10)

    plt.savefig('/Users/huyan/Data/WKL/Plots/IoU.png', dpi=300)
    # plt.show()


file_path = '/Users/huyan/Data/WKL/automapping/WestKunlun_Sentinel2_2018_westKunlun_beta_exp14_Area30k/'
train_txt_path = file_path + 'westKunlun_beta_exp14_training_miou.txt'
val_txt_path = file_path + 'westKunlun_beta_exp14_val_miou.txt'

train_dict_data = io_function.read_dict_from_txt_json(train_txt_path)
val_dict_data = io_function.read_dict_from_txt_json(val_txt_path)
plot_miou_step_time(train_dict_data, val_dict_data)