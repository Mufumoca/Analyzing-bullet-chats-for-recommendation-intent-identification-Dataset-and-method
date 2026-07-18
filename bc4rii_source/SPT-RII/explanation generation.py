import os
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxxx",
    base_url="https://api.deepseek.com"
)


def generate_background(context: list, current: str, pre: int, post: int = 0) -> str:
    prompt = f"""
你是一名智能客服助手，负责辅助判断电商直播弹幕是否表达了用户购买兴趣。你正在分析一场社交平台的电商直播弹幕。
当前目标弹幕是：“{current}”
以下是该直播间在此之前的{pre}条弹幕：
{"；".join(context)}
你需要模拟目标用户的可能心理和表达动机，并结合前文弹幕内容构建合理的语境背景理解，从而对目标弹幕含义进行准确释义。
回答： 该用户想表达的含义是：（请用一句通顺自然的现代汉语表达，不重复原文）
为了信息密度高，不用拿具体的弹幕举例说明
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content


def get_context_window_forward_only(data: list, idx: int, window_size: int = 20):
    start_idx = max(0, idx - window_size)
    context = data[start_idx:idx]  # 不包含当前弹幕
    pre = len(context)
    return context, pre


def process_csv(file_path: str, output_path: str):
    print(f"🚀 开始处理文件：{file_path}")

    # 判断是否已存在已处理文件
    if os.path.exists(output_path):
        df = pd.read_csv(output_path, encoding='gbk')
        print(f"📄 检测到已有输出文件，尝试继续处理未完成部分")
    else:
        df = pd.read_csv(file_path, encoding='gbk')
        df['background'] = None  # 添加背景列

    if 'content' not in df.columns:
        raise ValueError(f"CSV文件 {file_path} 中必须包含 'content' 列")

    data = df['content'].tolist()
    total_count = len(df)

    # 只保留目标列
    cols_to_save = ['live_id', 'species', 'content', 'label', 'background']
    cols_existing = [c for c in cols_to_save if c in df.columns]

    for idx in range(total_count):
        if pd.notna(df.loc[idx, 'background']):
            print(f"✅ 跳过索引 {idx}，已存在背景信息")
            continue

        context_window, pre = get_context_window_forward_only(data, idx)
        current_danmaku = data[idx]

        try:
            background = generate_background(context_window, current_danmaku, pre, post=0)
        except Exception as e:
            background = f"生成失败：{str(e)}"

        df.loc[idx, 'background'] = background

        # 保存进度
        df.loc[:, cols_existing].to_csv(output_path, index=False, encoding='gbk')
        print(f"💾 已处理索引 {idx + 1}/{total_count}，背景信息：{background}")

    print(f"🎉 文件处理完成，已保存至：{output_path}")


def batch_process_folder(input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                process_csv(input_path, output_path)
            except Exception as e:
                print(f"❌ 处理文件 {filename} 出错：{e}")


# 设置输入输出目录
input_dir = r"your files"
output_dir = r"output files with generation"

batch_process_folder(input_dir, output_dir)
