import winsound
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_compare_precictions(train_y, y_pred_train, test_y, y_pred_test):
    plt.figure(figsize=(15, 7))

    plt.subplot(2, 1, 1)
    plot_prog_vs_correct(y_pred_train, train_y, 'Train')

    plt.subplot(2, 1, 2)
    plot_prog_vs_correct(y_pred_test, test_y, 'Test')
    plt.show()


def plot_prog_vs_correct(predicted, correct, title=""):
    count = predicted.shape[0]
    os_x = range(count)

    plt.plot(os_x, correct, label='Right', color="green")
    plt.plot(os_x, predicted, label='Predicted', color="red")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)


def my_beep(start_freq=900, step_freq=300, n_steps=3):
    for i in range(n_steps):
        frequency = start_freq + i*step_freq       # Set Frequency To x Hertz
        duration = int(2000/n_steps)  # Set Duration To x ms
        winsound.Beep(frequency, duration)


def display_error_change_and_avg(error_in_iterations, iteration_cout, title=''):
    error_in_iterations = np.array(error_in_iterations)
    first_iteration = error_in_iterations[0,:]
    avg_version = np.average(error_in_iterations, axis=0)

    x = np.arange(1, iteration_cout + 1)
    plt.ylim(min(np.min(error_in_iterations),np.min(first_iteration)), max(np.max(error_in_iterations),(np.max(first_iteration))))
    plt.xlim(1.0, iteration_cout)
    plt.xlabel('epoka uczenia')
    plt.ylabel('Sredni blad bezwzgledny(po normalizacji)')
    plt.title(title)

    lineOne = plt.plot(x, first_iteration, color='red', label='Pojedynczy przebieg', linewidth=1.5)
    lineAvg = plt.plot(x, avg_version, color='blue', label='Sredni przebieg', linewidth=1.5)
    legend = plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    full_path = os.path.abspath('.//'+'results'+'//avg'+datetime.now().strftime('%d-%m %H_%M')+title+".png")
    plt.ylim(ymin=0.025, ymax=0.035)
    plt.savefig(full_path)
