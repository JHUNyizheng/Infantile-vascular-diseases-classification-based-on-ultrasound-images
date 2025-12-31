import os
import json
from PIL import Image, ImageDraw


def draw_rectangles(image_path, gray_coords, phlebolith_coords, output_path):
    """
    在给定的灰度图上绘制灰度图和phlebolith的矩形框。

    :param image_path: 图片的路径
    :param gray_coords: 灰度图的坐标列表
    :param phlebolith_coords: phlebolith的坐标列表
    :param output_path: 绘制矩形后输出的图片路径
    """
    # 打开灰度图
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 绘制灰度图的矩形框（红色）
    for coord in gray_coords:
        top_left = tuple(coord[0])  # 左上角坐标
        bottom_right = tuple(coord[1])  # 右下角坐标
        draw.rectangle([top_left, bottom_right], outline="red", width=3)

    # 绘制phlebolith的矩形框（绿色）
    for coord in phlebolith_coords:
        top_left = tuple(coord[0])
        bottom_right = tuple(coord[1])
        draw.rectangle([top_left, bottom_right], outline="green", width=3)

    # 保存绘制后的图片
    image.save(output_path)
    # print(f"Image saved to {output_path}")


def process_json_and_draw(json_file, root_dir, output_dir):
    """
    读取JSON文件，提取图片和坐标信息，并在灰度图上绘制矩形框。

    :param json_file: JSON文件路径
    :param root_dir: 根目录路径，用于拼接灰度图的完整路径
    """
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        datalist = json.load(f)

    for data in datalist:
        # 获取类别（VM, IH, NR 等），用来拼接路径
        category = data['label']  # 从JSON中获取类别标签，例如 'VM', 'IH', 'NR'

        # 获取灰度图路径和坐标信息
        gray_image_rel_path = data['gray']  # 相对路径的灰度图文件名
        gray_image_path = os.path.join(root_dir, category, gray_image_rel_path)  # 完整的灰度图路径，包含类别文件夹
        gray_coords = data['gray_coords']  # 灰度图坐标
        phlebolith_coords = data['phlebolith_coords']  # phlebolith坐标（灰度图上）

        # 绘制灰度图上的矩形框（包括灰度图和phlebolith的矩形框）
        if gray_image_path:
            output_gray_image = os.path.join(output_dir, category, f"annotated_{os.path.basename(gray_image_rel_path)}")
            draw_rectangles(gray_image_path, gray_coords, phlebolith_coords, output_gray_image)


# 示例使用
json_file = 'output_file.json'  # 你的JSON文件路径
root_dir = './data'  # 根目录路径，替换为你的实际根目录
output_dir = './draw'

process_json_and_draw(json_file, root_dir, output_dir)
