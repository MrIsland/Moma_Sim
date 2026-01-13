import numpy as np
import matplotlib.pyplot as plt

# ---- Step 1: 构造蓝色虚线（原始速度信号） ----
n = 200
velocity_cmd = np.ones(n)
velocity_cmd[50:100] = -1
velocity_cmd[150:] = -1  # 方波式变化

# ---- Step 2: 设置低通滤波参数 ----
alpha = 0.9  # 平滑系数 (越大越平滑)
filtered_velocity = np.zeros_like(velocity_cmd)
filtered_velocity[0] = velocity_cmd[0]

# ---- Step 3: 递推实现低通滤波 y[k] = alpha * y[k-1] + (1 - alpha) * x[k] ----
for i in range(1, n):
    filtered_velocity[i] = alpha * filtered_velocity[i-1] + (1 - alpha) * velocity_cmd[i]

# ---- Step 4: 可视化结果 ----
plt.figure(figsize=(10,3))
plt.plot(velocity_cmd, 'b--', label='Original Velocity')
plt.plot(filtered_velocity, 'r-', linewidth=2, label=f'Filtered Velocity (alpha={alpha})')
plt.title('Effect of Low-Pass Filter on Sudden Velocity Command Changes')
plt.xlabel('Time Step')
plt.ylabel('Velocity Amplitude')
plt.legend()
plt.grid(True)
plt.show()
