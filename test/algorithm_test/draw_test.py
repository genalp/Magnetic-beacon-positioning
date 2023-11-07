import numpy as np
import random
from scipy import signal
import matplotlib.pyplot as plt

# 1. 生成正弦信号
fs = 1000  # 采样频率
t = np.linspace(0, 1, fs, endpoint=False)  # 时间数组
frequency = 5  # 正弦信号的频率
amplitude = 1.0  # 正弦信号的振幅
sinusoid = amplitude * np.sin(2 * np.pi * frequency * t)

# 2. 添加限定最大绝对值的高斯噪声
mean = 0  # 噪声的均值
stddev = 0.1  # 噪声的标准差
max_absolute_value = 0.3  # 噪声的最大绝对值
noise = [random.gauss(mean, stddev) for _ in range(len(t))]
noise_scale_factor = max_absolute_value / max([abs(n) for n in noise])
noise = [n * noise_scale_factor for n in noise]
noisy_signal = sinusoid + noise

# 3. 定义IIR低通滤波器的参数
cutoff_frequency = 30  # 截止频率 (Hz)
filter_order = 4  # 滤波器阶数

# 4. 创建IIR低通滤波器
b, a = signal.butter(filter_order, cutoff_frequency / (0.5 * fs), btype='low')
filtered_signal = signal.lfilter(b, a, noisy_signal)

# 5. 进行FFT
fft_result = np.fft.fft(filtered_signal)
fft_freq = np.fft.fftfreq(len(t), 1 / fs)

# 6. 提取特定频率对应的幅值
target_frequency = 5  # 你想提取的频率

# 1. 计算目标频率在FFT结果中的索引
index = int(target_frequency * len(fft_result) / fs)

# 2. 根据FFT的幅值公式将幅值除以FFT数据点数量的一半
amplitude_at_target_frequency = np.abs(fft_result[index]) / (len(fft_result) / 2)

# 7. 绘制信号和FFT结果
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, sinusoid, label='Original Sinusoid')
plt.title('Original Sinusoid Signal')

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label='Noisy Signal', color='red')
plt.title('Signal with Gaussian Noise')

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, label='Filtered Signal', color='green')
plt.title('Filtered Signal (IIR Low-Pass Filter)')

plt.tight_layout()

# 8. 绘制FFT结果
plt.figure(figsize=(10, 4))
plt.plot(fft_freq, np.abs(fft_result))
plt.title('FFT Result')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 50)  # 限制绘图范围，根据需要调整

# 打印特定频率处的幅值
print("Amplitude at", target_frequency, "Hz:", amplitude_at_target_frequency)

plt.show()
