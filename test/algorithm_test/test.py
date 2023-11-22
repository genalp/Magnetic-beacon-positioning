from tqdm import tqdm
import time

# 示例循环
for i in tqdm(range(20), desc="Processing", unit="iteration"):
    # 模拟耗时的任务
    time.sleep(0.1)
