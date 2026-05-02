import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_charging_curve(csv_filepath):

    file_name = Path(csv_filepath).name


    try:
        df = pd.read_csv(csv_filepath)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return


    fig, ax1 = plt.subplots(figsize=(12, 6))


    color_volt = '#1f77b4'  # Steel Blue
    ax1.set_xlabel('Time (HH:MM)', fontsize=12)
    ax1.set_ylabel('Cell Voltage (V)', color=color_volt, fontsize=12, fontweight='bold')
    ax1.plot(df['TimeStamp'], df['Cell 1 Voltage'], color=color_volt, linewidth=2.5, label='Cell 1 Voltage')
    ax1.tick_params(axis='y', labelcolor=color_volt)


    ax2 = ax1.twinx()

    color_curr = '#9467bd'  # Muted Purple
    ax2.set_ylabel('Current (A)', color=color_curr, fontsize=12, fontweight='bold')
    ax2.plot(df['TimeStamp'], df['Current'], color=color_curr, linewidth=2, linestyle='-', alpha=0.8,
             label='Charging Current')
    ax2.tick_params(axis='y', labelcolor=color_curr)


    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


    plt.title(f'4S1P Battery Charging Curve (CC-CV Profile)\n[Source: {file_name}]',
              fontsize=16, fontweight='bold', pad=15)

    ax1.grid(True, linestyle='--', alpha=0.6)


    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.90))

    fig.tight_layout()

    # 이미지 저장
    save_path = csv_filepath.replace('.csv', '_charging_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"그래프가 저장되었습니다: {save_path}")

    plt.show()


if __name__ == '__main__':

    file_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\charge_Data_5_260413.csv'


    if Path(file_path).exists():
        plot_charging_curve(file_path)
    else:
        print(f"파일 경로를 확인해주세요: {file_path}")