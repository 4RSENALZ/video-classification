#检查下载失败的视频

import os
import pandas as pd
from pathlib import Path
import re

def extract_bvid(text):
    """通用BV号提取函数"""
    # 匹配两种格式：
    # 1. 直接包含的BV号（如 [BV1vb4y1q7QL]）
    # 2. 链接形式的BV号（如 https://b23.tv/BV1gr4y1U7cR）
    match = re.search(r'(BV[\dA-Za-z]{10})', str(text))
    return match.group(1) if match else None

def find_missing_videos():
    # 配置路径参数（保持不变）
    excel_path = r"C:\Users\28302\OneDrive\文档\我的B站视频数据集.xlsx"
    text_dirs = [
        r"E:\毕业设计\datasets\bili_datasets\raw_data\train\text",
        r"E:\毕业设计\datasets\bili_datasets\raw_data\test\text",
        r"E:\毕业设计\datasets\bili_datasets\raw_data\val\text"
    ]
    output_path = "missing_bvids.xlsx"

    # 读取Excel数据
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        # 提取Excel中的BV号（假设列名为"BV号"）
        df['bvid'] = df['BV号'].apply(extract_bvid)
        excel_bvids = set(df['bvid'].dropna().unique())
    except Exception as e:
        print(f"Excel读取失败: {str(e)}")
        return

    # 获取已下载BV号
    downloaded_bvids = set()
    for dir_path in text_dirs:
        txt_file = Path(dir_path) / "titles.txt"
        if not txt_file.exists():
            print(f"警告：{txt_file} 不存在")
            continue
            
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                next(f)  # 跳过首行标题行
                for line_num, line in enumerate(f, 2):
                    # 从文件名提取BV号
                    filename_part = line.split('|', 1)[0].strip()
                    if bvid := extract_bvid(filename_part):
                        downloaded_bvids.add(bvid)
        except UnicodeDecodeError:
            with open(txt_file, 'r', encoding='gbk') as f:
                next(f)
                for line_num, line in enumerate(f, 2):
                    filename_part = line.split('|', 1)[0].strip()
                    if bvid := extract_bvid(filename_part):
                        downloaded_bvids.add(bvid)

    # 调试信息
    print(f"Excel BV号数量：{len(excel_bvids)}")
    print(f"已下载BV号数量：{len(downloaded_bvids)}")
    print("前5个Excel BV号示例：", list(excel_bvids)[:5])
    print("前5个下载BV号示例：", list(downloaded_bvids)[:5])

    # 计算缺失项
    missing_bvids = excel_bvids - downloaded_bvids

    # 生成结果报告
    if missing_bvids:
        result_df = df[df['bvid'].isin(missing_bvids)].drop_duplicates('bvid')
        result_df.to_excel(output_path, index=False)
        print(f"发现{len(missing_bvids)}条缺失记录，已保存至 {output_path}")
    else:
        print("未发现缺失视频记录")

if __name__ == "__main__":
    find_missing_videos()
