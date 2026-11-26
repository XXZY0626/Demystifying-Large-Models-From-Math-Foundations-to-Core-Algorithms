"""固定版式词云：粗体蓝橙版。用文字绘制近似重建，不读取原图。
使用 F:\\Python\\Python312\\python.exe 运行，仅依赖 Pillow。
坐标与字号基于1860×846画布，按2倍分辨率输出，可直接修改下表。
词汇位置和字号为人工设计，不表示论文词频统计。
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


W, H = 1860, 846
alpha = 2
font = r"C:\Windows\Fonts\msyhbd.ttc"
blue = "#5988CF"
dark = "#245780"
gray = "#62798C"
orange = "#E76F54"

# 每行：[词语，左边，上边，宽度，高度，字号，颜色，是否竖排]
x = [
    ["不确定性", 535, 343, 790, 205, 178, dark, False],
    ["车流波动", 240, 222, 475, 128, 113, orange, False],
    ["计划偏差", 1135, 222, 455, 128, 108, orange, False],
    ["混编列车", 80, 395, 450, 125, 105, blue, False],
    ["满轴条件", 1340, 385, 465, 130, 105, blue, False],
    ["动态调整", 700, 58, 400, 105, 90, orange, False],
    ["车流集结", 365, 109, 315, 90, 73, blue, False],
    ["接续改编", 1120, 112, 290, 85, 66, gray, False],
    ["信息不完整", 720, 190, 400, 90, 73, gray, False],
    ["列车径路", 283, 551, 385, 101, 85, gray, False],
    ["日班计划", 695, 588, 360, 103, 79, blue, False],
    ["编组计划", 1090, 568, 267, 83, 61, blue, False],
    ["运行图", 1220, 650, 235, 83, 71, gray, False],
    ["运输需求", 1370, 536, 260, 80, 60, orange, False],
    ["概率分布", 167, 177, 180, 48, 40, blue, False],
    ["联合熵", 74, 264, 150, 52, 43, gray, False],
    ["统一度量", 33, 341, 200, 56, 45, blue, False],
    ["条件熵", 85, 530, 53, 135, 41, blue, True],
    ["列车熵", 183, 553, 53, 135, 41, gray, True],
    ["集结规律", 1425, 156, 200, 51, 43, blue, False],
    ["计划执行", 1590, 220, 176, 48, 39, gray, False],
    ["不确定性消减", 1595, 296, 240, 49, 37, blue, False],
    ["数量状态", 1647, 526, 176, 49, 40, blue, False],
    ["开行对数", 1642, 598, 177, 48, 39, gray, False],
    ["直达列车", 460, 707, 201, 55, 45, blue, False],
    ["待送车辆", 721, 725, 214, 55, 46, gray, False],
    ["运输组织", 999, 727, 214, 55, 46, blue, False],
    ["随机变量", 1263, 740, 195, 48, 41, gray, False],
]

im = Image.new("RGB", (W * alpha, H * alpha), "white")
draw = ImageDraw.Draw(im)
for i in range(len(x)):
    t, a, b, w, h, s, c, v = x[i]
    if v:
        t = "\n".join(t)
    f = ImageFont.truetype(font, s * alpha)
    box = draw.multiline_textbbox((0, 0), t, font=f, spacing=0, align="center")
    while box[2] - box[0] > w * alpha or box[3] - box[1] > h * alpha:
        s -= 1
        f = ImageFont.truetype(font, s * alpha)
        box = draw.multiline_textbbox((0, 0), t, font=f, spacing=0, align="center")
    a = (a + w / 2) * alpha - (box[0] + box[2]) / 2
    b = (b + h / 2) * alpha - (box[1] + box[3]) / 2
    draw.multiline_text((a, b), t, font=f, fill=c, spacing=0, align="center")

y = Path(__file__).resolve().with_name("词云图1_粗体蓝橙版.png")
im.save(y, dpi=(300, 300))
print(f"已保存：{y}")
