import numpy as np
import pywt
import matplotlib.pyplot as plt

# 1️⃣ Tạo tín hiệu ví dụ
t = np.linspace(0, 1, 512)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

# 2️⃣ Thực hiện MRA
coeffs = pywt.mra(x, wavelet='db4', level=3, transform='swt', mode='periodization')

# 3️⃣ Hiển thị kết quả
plt.figure(figsize=(10, 8))
plt.subplot(len(coeffs)+1, 1, 1)
plt.plot(t, x)
plt.title('Tín hiệu gốc')

for i, c in enumerate(coeffs):
    plt.subplot(len(coeffs)+1, 1, i+2)
    plt.plot(t, c)
    plt.title(f'Cấp độ {i}')

plt.tight_layout()
plt.show()
