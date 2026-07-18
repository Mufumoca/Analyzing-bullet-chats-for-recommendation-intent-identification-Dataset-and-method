import csv
import logging
import random
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from itertools import product
from datetime import datetime
import hanlp
# 加载电商领域微调模型
import pandas as pd
from sentence_dis import TemporalAnchorManager
import os
import time

tok = hanlp.load("UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6")
zeroshot_platform = "rec-related"
input_file_path = f"datasets/veb/{zeroshot_platform}/all_test.csv"
output_file = f"datasets/veb/{zeroshot_platform}/test.csv"  # 同一个文件


#扩展词判断
def zeroshot():
    cmd = (
        f"python zeroshot.py --result_file ./output_zeroshot1.txt "
        f"--dataset {zeroshot_platform} --template_id 0 --seed 188 "
        f"--verbalizer kpt --calibration"
    )
    print(cmd)
    logging.info(f"Executing command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        logging.info(f"Command executed successfully: {cmd}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")
#扩展词更新，需要重新训练模型
def fewshot(n,t,j,i,m,k,v,e):

    cmd = (
        f"python fewshot.py --result_file ./result/{fewshot_platform}.txt "
        f"--dataset {n} --template_id {t} --seed {j} "
        f"--batch_size {i} --shot {m} --learning_rate {k} "
        f"--verbalizer {v} --max_epochs {e}"
    )
    print(cmd)

    logging.info(f"Executing command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
        logging.info(f"Command executed successfully: {cmd}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")
#扩展词未更新，直接测试
def fewshot1(n,t,j,i,v):
    # 直接运行（跳过product循环）

    cmd = (
        f"python fewshot1.py --result_file ./result/{fewshot_platform}.txt "
        f"--dataset {n} --template_id {t} --seed {j} "
        f"--batch_size {i}  --verbalizer {v}"
    )
    print(cmd)

    logging.info(f"Executing command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
        logging.info(f"Command executed successfully: {cmd}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {cmd}. Error: {e.stderr.decode().strip()}")

from itertools import product
if __name__ == '__main__':
    fewshot_platform = "rec-dy"
    number = [15]
    anchor0 = ['闲聊',"闲聊系统","不可推荐",'售后','物流','差评','气氛','主播个人提问','陈述感受']
    anchor1 = ['购买',"推荐系统","可推荐",'购买意图','推荐价值','商品兴趣','商品功能','细节询问','价格促销','使用场景','积极评价','决策辅助']

    dataset = {'rec-dy'}
    template = {0}
    seed = [100,101]
    batch_size = [32]
    learning_rate = [4e-5]
    shot = [50]
    verbalizer = {"kpt"}
    max_epochs = [15]
    p = [500]
    # 初始化时间衰减管理器
    keyword_manager = TemporalAnchorManager(
        anchor_groups=[anchor0, anchor1],
        model_path=r'/home/zy-4090-1/hqq/SPT-RII/model/paraphrase-multilingual-MiniLM-L12-v2'
    )

    for n, t, j, i, m, k, v, e ,pbs,num in product(dataset, template, seed, batch_size, shot, learning_rate, verbalizer, max_epochs,p,number):
        # 清空计算所用文件夹
        folder_path = 'result'
        file_path = os.path.join(folder_path, 'label_for_cal.csv')
        os.makedirs(folder_path, exist_ok=True)
        df = pd.DataFrame(columns=['true_label', 'pred_label'])
        df.to_csv(file_path, index=False, encoding='utf-8')

        # 批量收集弹幕
        processing_batch_size = pbs
        batch_rows = []
        file_path = input_file_path  # all_test.csv

        with open(file_path, "r", encoding="utf-8") as fin:
            start_time = time.time()
            reader = csv.reader(fin)
            count = 0

            # 初始化kpt.txt文件
            with open(fr"scripts/TextClassification/{fewshot_platform}/kpt.txt", 'w', encoding='utf-8') as f:
                f.write(anchor0[0] + "\n")
                f.write(anchor1[0])

            for row in reader:
                batch_rows.append(row)

                if len(batch_rows) == processing_batch_size:
                    print(f"\n=== 开始处理第 {count + 1} 批（{processing_batch_size}条） ===")
                    round_start_time = time.time()

                    # 1. 批量分词处理
                    word_lists = []
                    for row in batch_rows:
                        output = tok(row[2])
                        tokens = output['tok']
                        pos_tags = output['pos']
                        filtered_words = [
                            word for word, pos in zip(tokens, pos_tags)
                            if pos in {'NOUN', 'PROPN', 'VERB'} and len(word) > 1
                        ]
                        word_lists.append(filtered_words)
                    true_labels = [row[2] for row in batch_rows]

                    # 2. 批量写入临时文件
                    with open(output_file, 'w', encoding='utf-8', newline='') as fout:
                        writer = csv.writer(fout)
                        for words in word_lists:
                            for word in words:
                                writer.writerow([word, 0])

                    with open(f"datasets/TextClassification/{fewshot_platform}/test.csv", 'w', encoding='utf-8',
                             newline='') as fout1:
                        writer = csv.writer(fout1)
                        writer.writerows(batch_rows)

                    # 3. zeroshot 执行
                    print("\n[ZeroShot 阶段]")
                    zeroshot()

                    # 4. 结果处理
                    df = pd.read_csv(f'datasets/veb/{zeroshot_platform}/test.csv')
                    df = df[df['predict'] == 1]['text']
                    related_wordlist = df.values.tolist()

                    flag = False
                    if len(related_wordlist) != 0:
                        # 计算所有相关词的距离
                        new_words = []
                        for test_word in related_wordlist:
                            result = keyword_manager.calculate_distance(test_word)
                            new_words.append(result)

                        # 获取经过时间衰减处理的关键词
                        chat_words, rec_words = keyword_manager.update_keywords(new_words, num)
                        chat_words.insert(0,anchor0[0])
                        rec_words.insert(0, anchor1[0])
                        print("chat_words:",chat_words)
                        print("rec_words:",rec_words)
                        #这里写入闲聊和推荐锚定词
                        with open(f"scripts/TextClassification/{fewshot_platform}/kpt.txt", "w", encoding='utf-8') as f:
                            f.write(','.join(chat_words) + '\n')
                            f.write(','.join(rec_words) + '\n')
                    else:
                        flag = True

                    if count != 0 and flag:
                        fewshot1(n, t, j, i, v)  # no train
                    else:
                        fewshot(n, t, j, i, m, k, v, e)

                    count += 1
                    round_end_time = time.time()
                    print(f"本批处理耗时: {round_end_time - round_start_time:.2f}秒")
                    batch_rows = []

            # 处理剩余不足batch_size条的弹幕
            if batch_rows:
                print(f"\n=== 开始处理最后一批（{len(batch_rows)}条） ===")
                round_start_time = time.time()

                # 1. 批量分词处理
                word_lists = []
                for row in batch_rows:
                    output = tok(row[1])
                    tokens = output['tok']
                    pos_tags = output['pos']
                    filtered_words = [
                        word for word, pos in zip(tokens, pos_tags)
                        if pos in {'NOUN', 'PROPN', 'VERB'} and len(word) > 1
                    ]
                    word_lists.append(filtered_words)
                true_labels = [row[2] for row in batch_rows]

                # 2. 批量写入临时文件
                with open(output_file, 'w', encoding='utf-8', newline='') as fout:
                    writer = csv.writer(fout)
                    for words in word_lists:
                        for word in words:
                            writer.writerow([word, 0])

                with open(f"datasets/TextClassification/{fewshot_platform}/test.csv", 'w', encoding='utf-8',
                          newline='') as fout1:
                    writer = csv.writer(fout1)
                    writer.writerows(batch_rows)

                # 3. zeroshot 执行
                print("\n[ZeroShot 阶段]")
                zeroshot()

                # 4. 结果处理
                df = pd.read_csv(f'datasets/veb/{zeroshot_platform}/test.csv')
                df = df[df['predict'] == 1]['text']
                related_wordlist = df.values.tolist()

                flag = False
                if len(related_wordlist) != 0:
                    # 计算所有相关词的距离
                    new_words = []
                    for test_word in related_wordlist:
                        result = keyword_manager.calculate_distance(test_word)
                        new_words.append(result)

                    # 获取经过时间衰减处理的关键词
                    chat_words, rec_words = keyword_manager.update_keywords(new_words, num)
                    chat_words.insert(0, anchor0[0])
                    rec_words.insert(0, anchor1[0])
                    print("chat_words:", chat_words)
                    print("rec_words:", rec_words)
                    # 这里写入闲聊和推荐锚定词
                    with open(f"scripts/TextClassification/{fewshot_platform}/kpt.txt", "w", encoding='utf-8') as f:
                        f.write(','.join(chat_words) + '\n')
                        f.write(','.join(rec_words) + '\n')

                else:
                    flag = True

                if count != 0 and flag:
                    fewshot1(n, t, j, i, v)  # no train
                else:
                    fewshot(n, t, j, i, m, k, v, e)

                count += 1
                round_end_time = time.time()
                print(f"本批处理耗时: {round_end_time - round_start_time:.2f}秒")

            # 评价指标计算
            df = pd.read_csv(f'result/label_for_cal.csv')
            y_true = df['true_label']
            y_pred = df['pred_label']

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')

            # 记录结果
            content_write = "=" * 20 + "\n"
            content_write += f"dataset {fewshot_platform}\t"
            content_write += f"temp_id {t}\t"
            content_write += f"seed {j}\t"
            content_write += f"shot {m}\t"
            content_write += f"verb_spe {v}\t"
            content_write += f"verb_num {num}\t"
            content_write += f"batch_size {i}\t"
            content_write += f"lr {k}\t"
            content_write += f"max_epochs {e}\t"

            content_write += "\n"

            content_write += f"Acc: {accuracy:.4f}\t"
            content_write += f"Pre: {precision:.4f}\t"
            content_write += f"Rec: {recall:.4f}\t"
            content_write += f"F1s: {f1:.4f}\t"
            content_write += "\n\n"

            print(content_write)

            name = '推荐意图识别'
            data = {
                'name': [name],
                'dataset': [fewshot_platform],
                'template_id': [t],
                'Seed': [j],
                'Shot': [m],
                'verb_spe': [v],
                'verb_num': [num],
                'pici': [processing_batch_size],
                'learning_rate': [k],
                'batch_size': [i],
                'max_epochs': [e],

                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1]
            }
            df = pd.DataFrame(data)

            file_path = f'./result/{fewshot_platform}.xlsx'
            if not os.path.exists(file_path):
                df.to_excel(file_path, index=False, header=True)
            else:
                with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay') as writer:
                    df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

            with open(f"{fewshot_platform}", "a") as fout_result:
                fout_result.write(content_write)

            end_time = time.time()
            print(f"总耗时: {end_time - start_time:.2f}秒")










