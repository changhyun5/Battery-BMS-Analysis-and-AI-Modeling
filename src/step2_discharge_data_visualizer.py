import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_discharging_curve(csv_filepath):

    file_name = Path(csv_filepath).name


    df = pd.read_csv(csv_filepath)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])


    fig, ax1 = plt.subplots(figsize=(12, 6))


    color_volt = '#d62728'
    ax1.set_xlabel('Time (HH:MM)', fontsize=12)
    ax1.set_ylabel('Cell Voltage (V)', color=color_volt, fontsize=12, fontweight='bold')
    ax1.plot(df['TimeStamp'], df['Cell 1 Voltage'], color=color_volt, linewidth=2.5, label='Cell 1 Voltage')
    ax1.tick_params(axis='y', labelcolor=color_volt)


    ax2 = ax1.twinx()
    color_curr = '#2ca02c'
    ax2.set_ylabel('Current (A)', color=color_curr, fontsize=12, fontweight='bold')
    ax2.plot(df['TimeStamp'], df['Current'], color=color_curr, linewidth=2, linestyle='-', alpha=0.8,
             label='Discharge Current')
    ax2.tick_params(axis='y', labelcolor=color_curr)


    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    plt.title(f'4S1P Battery Discharging Curve (Voltage Drop Profile)\n[Source: {file_name}]',
              fontsize=16, fontweight='bold', pad=15)

    ax1.grid(True, linestyle='--', alpha=0.7)


    ax1.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1))
    ax2.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    file_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\discharge_Data_5_260413.csv'

    plot_discharging_curve(file_path)