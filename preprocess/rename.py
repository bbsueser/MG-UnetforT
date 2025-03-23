import os

# 配置
author_info = """# Author: Zhj
# Date: 2025-03-23
# Description: This file is part of the MG-UNet project
"""

# 文件夹路径
folder_path = r"D:\pythonProject\usedcode\shilifenge\postprocess"

# 支持的文件类型
file_extensions = [".py", ".js", ".java", ".cpp", ".h", ".cs"]

# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 检查文件扩展名是否匹配
        if any(file.endswith(ext) for ext in file_extensions):
            file_path = os.path.join(root, file)

            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 如果文件已经包含作者信息，则跳过
            if author_info.strip() in content:
                print(f"Skipping {file} (already contains author info)")
                continue

            # 将作者信息写入文件开头
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(author_info + "\n" + content)

            print(f"Updated {file}")