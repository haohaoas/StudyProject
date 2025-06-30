import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import cv2
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=16)

def draw_frame(weights, left, right, mid, capacity, day_used, idx, loaded, step, filename):
    plt.figure(figsize=(8, 4))
    plt.title(f"二分查找第{step}步", fontproperties=font)
    plt.xlim(0, len(weights)+1)
    plt.ylim(0, max(weights)*2)
    for i, w in enumerate(weights):
        plt.bar(i+1, w, color='skyblue' if i not in loaded else 'orange')
        plt.text(i+1, w+0.2, str(w), ha='center', fontproperties=font)
    plt.text(0.5, max(weights)*1.8, f"left={left}, right={right}, mid={mid}", fontproperties=font, fontsize=12)
    plt.text(0.5, max(weights)*1.6, f"当前运载能力: {capacity}，用天数: {day_used}", fontproperties=font, fontsize=12)
    plt.text(0.5, max(weights)*1.4, f"第{step}步，处理第{idx+1}个包裹", fontproperties=font, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def binary_search_animation_mp4(weights, D, output_video='ship_capacity_demo_cn.mp4', seconds_per_frame=2, answer_last=5):
    left, right = max(weights), sum(weights)
    images = []
    step = 1

    while left < right:
        mid = (left + right) // 2
        capacity = 0
        day_used = 1
        loaded = []
        filenames = []
        for idx, w in enumerate(weights):
            if capacity + w > mid:
                day_used += 1
                capacity = 0
                loaded = []
            capacity += w
            loaded.append(idx)
            fname = f"frame_{step}_{idx}.png"
            draw_frame(weights, left, right, mid, capacity, day_used, idx, loaded, step, fname)
            filenames.append(fname)
        for fname in filenames:
            img = imageio.imread(fname)
            images.append(img)
        for fname in filenames:
            os.remove(fname)
        if day_used > D:
            left = mid + 1
        else:
            right = mid
        step += 1

    # 最终答案帧
    draw_frame(weights, left, right, left, 0, 1, -1, [], step, "final.png")
    img = imageio.imread("final.png")
    for _ in range(answer_last):  # 答案帧多展示几秒
        images.append(img)
    os.remove("final.png")

    # 合成mp4
    height, width = images[0].shape[:2]
    fps = 1 / seconds_per_frame  # 例如2秒一帧，则fps=0.5
    # 但cv2不支持<1fps，所以我们直接用fps=1，重复帧实现更慢
    repeat_n = int(seconds_per_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 1, (width, height))
    for img in images:
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        for _ in range(repeat_n):  # 每帧重复repeat_n次，实现2秒一帧
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"已生成动画视频：{output_video}")

# 示例参数
weights = [1,2,3,4,5,6,7,8,9,10]
D = 5

# 每帧2秒，答案帧多展示5秒
binary_search_animation_mp4(weights, D, output_video='ship_capacity_demo_cn.mp4', seconds_per_frame=2, answer_last=5)