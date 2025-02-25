import os

def extract_portrait_text(file_path):
    """从指定的文件中提取 '体制内' 和 '体制外' 部分的文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 初始化两个变量来存储“体制内”和“体制外”的文本
    inside_text = []
    outside_text = []
    
    # 遍历文件的每一行，提取对应部分的内容
    in_inside = False
    in_outside = False
    
    for line in lines:
        line = line.strip()  # 去除每行的首尾空白字符
        if "体制内" in line:
            in_inside = True
            in_outside = False
        elif "体制外" in line:
            in_outside = True
            in_inside = False
        elif in_inside and line:
            inside_text.append(line)  # 仅非空行添加到 inside_text
        elif in_outside and line:
            outside_text.append(line)  # 仅非空行添加到 outside_text
    
    return "\n".join(inside_text), "\n".join(outside_text)

def save_to_file(file_name, text):
    """保存文本到文件，并计算字数"""
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)
        f.write("\n\n字数统计：\n")
        f.write(f"字数：{len(text)}")

def process_directory(root_dir):
    """遍历根目录，处理每个文件夹中的 `4_new_portrait.txt` 文件"""
    inside_all_text = []
    outside_all_text = []
    
    # 遍历目录中的文件结构
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "4_new_portrait.txt":
                file_path = os.path.join(root, file)
                print(f"正在处理文件: {file_path}")
                
                # 提取 "体制内" 和 "体制外" 的文本
                inside_text, outside_text = extract_portrait_text(file_path)
                
                # 将提取到的文本累积
                if inside_text:  # 如果有“体制内”文本
                    inside_all_text.append(inside_text)
                if outside_text:  # 如果有“体制外”文本
                    outside_all_text.append(outside_text)
    
    # 保存处理结果到文件
    save_to_file("体制内_新画像.txt", "\n\n".join(inside_all_text))
    save_to_file("体制外_新画像.txt", "\n\n".join(outside_all_text))
    print("处理完成，结果已保存至 '体制内_新画像.txt' 和 '体制外_新画像.txt'")

if __name__ == "__main__":
    root_dir = "."  # 当前目录
    process_directory(root_dir)