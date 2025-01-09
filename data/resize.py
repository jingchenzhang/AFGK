# from PIL import Image

# # 打开图像
# img = Image.open("/home/zjc/work_dir/code/refit/data/examples/COCO_val2014_000000149903.jpg")  # 替换为你的图片路径

# # 调整大小
# img_resized = img.resize((333, 500))

# # 保存调整后的图像
# img_resized.save("output.jpg")  # 保存为新文件
# print("图片已调整为 333x500 并保存为 output.jpg")
from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = "/home/zjc/work_dir/code/refit/data/images"  # 输入文件夹
output_folder = "/home/zjc/work_dir/code/refit/data/output"  # 输出文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # 只处理图片文件
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # 调整大小并保存
            img_resized = img.resize((256, 256))
            img_resized.save(os.path.join(output_folder, filename))
            print(f"{filename} 已调整大小并保存")

