import pandas as pd
import seaborn as sns
import mne
import numpy as np
import matplotlib.pyplot as plt


sub_01_ses_01_task_hfo_run_01_events_tsv = pd.read_csv("..\\data\\ds003555-1.0.1\\derivatives\\sub-01\\ses-01\\eeg\\sub-01_ses-01_task-hfo_run-01_events.tsv", sep="\t")
print("sub-01_ses_01_task-hfo_run-01_events_tsv:", sub_01_ses_01_task_hfo_run_01_events_tsv)

""" 存疑点，这个HFO检测为什么会变成52通道，后续可能需要解释，但是不重要先继续分析 """

print(sub_01_ses_01_task_hfo_run_01_events_tsv.columns)

print(sub_01_ses_01_task_hfo_run_01_events_tsv.iloc[:, 2:6])

sub_01_ses_01_task_hfo_channels_tsv = pd.read_csv("..\\data\\ds003555-1.0.1\\sub-01\\ses-01\\eeg\\sub-01_ses-01_task-hfo_channels.tsv", sep="\t")
# print("sub_01_ses_01_task_hfo_channels_tsv:", sub_01_ses_01_task_hfo_channels_tsv)


""" 
nChannel
通道编号：表示数据采集时的物理通道序号（如EEG电极通道编号），用于区分不同信号来源。

strChannelName
通道名称：通道的字符串标识，可能包含更详细的描述（如"Frontal_Lobe_Ch1"）。

indStart
起始索引：事件在时间序列数据中开始的样本点位置（如数组索引值）。

indStop
结束索引：事件在时间序列数据中结束的样本点位置。

indDuration
持续时间（样本点数）：事件从indStart到indStop的样本点数量，即 indStop - indStart + 1。

Event_RMS
事件均方根值：事件窗口内信号的均方根（Root Mean Square），反映事件的平均能量强度。

Window_RMS
参考窗口均方根值：用于对比的背景窗口（如事件前后的静息期）的RMS值，常作为噪声基准。

EventPeak2Peak
峰峰值幅度：事件信号中最高峰与最低谷的差值，表征信号的最大动态范围。

SNR
信噪比：事件信号与背景噪声的强度比，通常由 Event_RMS / Window_RMS 计算。

Amplpp
峰峰值幅度（可能缩写）：与EventPeak2Peak类似，可能为另一种计算方式或单位下的峰峰值。

PowerTrough
功率谷值：事件在特定频段（如θ/α波）的最低功率值，可能用于检测抑制性活动。

Ftrough
谷值对应频率：PowerTrough所在的频率点（单位Hz），标识功率最低点的频率位置。

PowmaxFR
频率范围内最大功率：在指定频率范围（如高频振荡的80-200Hz）内，事件窗口的最大功率值。

fmax_FR
最大功率对应频率：PowmaxFR所在的频率点（单位Hz），反映主导振荡频率。

EvPassRejection
事件通过/拒绝标记：布尔值或分类标签，表示事件是否通过质量控制（如SNR、幅度阈值等）。

"""

raw = mne.io.read_raw_edf("..\\data\\ds003555-1.0.1\\derivatives\\sub-01\\ses-01\\eeg\\sub-01_ses-01_task-hfo_run-01_eeg.edf", preload=True)
# raw = mne.io.read_raw_edf("..\\data\\ds003555-1.0.1\\sub-01\\ses-01\\eeg\\sub-01_ses-01_task-hfo_eeg.edf", preload=True)
preload=(True)

ch_labels = raw.ch_names

# print(ch_labels)

data = raw.get_data()

sampling_rate=raw.info['sfreq']
start_time=0
end_time=8

n_samples = data.shape[1]

def plot_eeg(sample, title, ch_labels, sampling_rate, start_time=None, end_time=None, target_channels=0):
    """
    Plots EEG signals over a specified time range.
    
    Parameters:
    - sample: 2D array (n_channels × n_samples) of EEG data.
    - title: Title of the plot.
    - ch_labels: List of channel names for y-axis labels.
    - sampling_rate: Sampling rate of the data (Hz) [[4]].
    - start_time: Start time (in seconds) for plotting (optional).
    - end_time: End time (in seconds) for plotting (optional).
    """
    n_samples = sample.shape[1]
    # total_time = n_samples / sampling_rate
    
    # 默认绘制全部数据
    start_idx = 0 if start_time is None else max(0, int(start_time * sampling_rate))
    end_idx = n_samples if end_time is None else min(n_samples, int(end_time * sampling_rate))
    
    # 截取指定时间范围的数据
    selected_data = sample[:, start_idx:end_idx]
    time = np.arange(start_idx, end_idx) / sampling_rate  # 生成对应的时间轴 [[2]]
    
    n_channels = selected_data.shape[0] - target_channels
    
    # 绘图
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    for i in range(n_channels):
        axes[i].plot(time, selected_data[i])
        axes[i].set_ylabel(ch_labels[i], rotation=45)
        axes[i].grid(True)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()
    # plt.savefig("eeg_plot.png", dpi=300, bbox_inches='tight')

raw.times[-1]
plot_eeg(data, "EEG Signal Example", ch_labels, sampling_rate=raw.info['sfreq'], start_time=0, end_time=20, target_channels=len(raw.ch_names) - 8)


start_idx = 0 if start_time is None else max(0, int(start_time * sampling_rate))
end_idx = n_samples if end_time is None else min(n_samples, int(end_time * sampling_rate))

""" 修改 """
selected_data = data[0, start_idx:end_idx]


def compute_T_GWST(z, K_max=None):
    """
    GWST
        z : 时域脑电信号，一个通道
        K_max : K的最大值，设为100，或者说脑电波都OK
        T_GWST : 时频矩阵（N x K_max）
        k_values : 使用的k值数组
    """
    N = len(z)
    K_max = K_max or N//2
    
    Z = np.fft.fft(z) / N
    
    k_values = np.arange(1, K_max+1)
    
    m = np.arange(N)[:, np.newaxis]  
    k = k_values[np.newaxis, :]      
    W = np.exp(-2 * ((m * np.pi) / k) ** 2)
    
    T_GWST = np.zeros((N, len(k_values)), dtype=np.complex128)
    
    # 计算每个k对应的GWST
    for idx, k in enumerate(k_values):
        # 循环移位Z
        shifted_Z = np.roll(Z, -k)
        
        # 窗口函数
        windowed_Z = shifted_Z * W[:, idx]
        
        # 计算逆FFT并缩放
        T_GWST[:, idx] = N * np.fft.ifft(windowed_Z)
    
    return T_GWST, k_values


from matplotlib import colors


T_GWST, k_values = compute_T_GWST(selected_data, K_max=100)

freq = k_values * sampling_rate / len(selected_data)

time_axis = np.arange(len(selected_data)) / sampling_rate

# print(T_GWST.shape)

data = np.abs(T_GWST).T  # 形状应为(100, N)



# 自动范围计算
vmin = np.quantile(data, 0.05)  # 取5%分位数作为下限
vmax = np.quantile(data, 0.95)  # 取95%分位数作为上限

cmap = plt.get_cmap('viridis').copy()
cmap.set_under('white')  # 设置低于vmin的颜色为白色

# 创建带异常值处理的颜色标准化
norm = colors.SymLogNorm(linthresh=1e-3,  # 线性区间阈值
                        linscale=0.5,    # 线性区间缩放
                        vmin=0.8,
                        vmax=vmax)

# 创建图形
plt.figure(figsize=(15, 8))

# 使用pcolormesh绘制
pc = plt.pcolormesh(time_axis,      # X轴：时间
                   freq,            # Y轴：频率
                   data,            # 数据矩阵
                   shading='auto',  # 自动选择最佳渲染方式
                   cmap=cmap,    # 推荐使用高对比度颜色映射
                   norm=norm)       # 应用混合对数标准化

# 添加颜色条
cbar = plt.colorbar(pc, extend='both')
cbar.set_label('Energy Intensity (a.u.)', rotation=270, labelpad=20)

# 坐标轴优化
plt.ylim(freq.min(), freq.max())  # 确保频率轴范围正确
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Frequency (Hz)', fontsize=12)
plt.title(f'Time-Frequency Analysis (Dynamic Range: {vmin:.1e}-{vmax:.1e})', fontsize=14)

plt.ylim(0, 60)

# 添加网格线
plt.grid(True, linestyle=':', alpha=0.5, which='both')

# 显示图形
plt.tight_layout()
plt.show()




""" """ """  """ """ """


brain_waves = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 60)
}

# freq = k_values * sampling_rate / len(selected_data)

selected_freqs = {}
for wave, (low, high) in brain_waves.items():
    # 找到在频段内的所有频率
    mask = (freq >= low) & (freq <= high)
    if np.any(mask):
        # 取该频段的中位数频率作为代表
        median_freq = np.median(freq[mask])
        idx = np.abs(freq - median_freq).argmin()
        selected_freqs[wave] = {
            'index': idx,
            'freq': freq[idx],
            'data': data[idx, :]  # data是时频矩阵的幅度
        }

plot_data = []
for wave, info in selected_freqs.items():
    for value in info['data']:
        plot_data.append({
            'Frequency Band': wave,
            'Amplitude': value,
            'Center Frequency (Hz)': f"{info['freq']:.1f} Hz"
        })

df = pd.DataFrame(plot_data)

boxprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
flierprops = dict(marker='o', markersize=5, markerfacecolor='grey')

order = sorted(selected_freqs.keys(), 
              key=lambda x: selected_freqs[x]['freq'])

plt.figure(figsize=(12, 6))
sns.boxplot(x='Frequency Band', 
           y='Amplitude',
           hue='Center Frequency (Hz)',
           data=df,
           order=order,
           palette='viridis',
           width=0.7,
           linewidth=1.5,
           flierprops=dict(markersize=4))

plt.title('EEG Frequency Band Distribution (Seaborn)', fontsize=14)
plt.xlabel('Frequency Band with Center Frequency', fontsize=12)
plt.ylabel('Amplitude (a.u.)', fontsize=12)
plt.legend(title='Center Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=10, rotation=45)
plt.tight_layout()
plt.show()