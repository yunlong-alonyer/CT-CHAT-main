import numpy as np
import json
import os

# 1. 配置文件路径
ctclip_weights_path = 'inference_zeroshot/predicted_weights.npz'
accessions_path = 'inference_zeroshot/accessions.txt'
# 假设你把Qwen的提取结果存成了如下结构的JSON:
# { "case_001": {"Cardiomegaly": 1, "Pleural effusion": 0, ...}, ... }
qwen_results_path = 'qwen_text_labels.json'

# 设置错配的置信度阈值 (可根据严格程度调节)
HIGH_CONFIDENCE_THRESHOLD = 0.8  # CT-CLIP 认为极可能存在的概率下限
LOW_CONFIDENCE_THRESHOLD = 0.2  # CT-CLIP 认为极不可能存在的概率上限

# 建立中英文映射字典
disease_mapping = {
    'Emphysema': '肺气肿',
    'Atelectasis': '肺不张',
    'Consolidation': '肺实变',
    'Lung nodule': '肺结节',
    'Bronchiectasis': '支气管扩张',
    'Pleural effusion': '胸腔积液',
    'Pericardial effusion': '心包积液',
    'Arterial wall calcification': '主动脉硬化',
    'Lung opacity': '磨玻璃影',
    'Interlobular septal thickening': 'Kerley B线'
}

# 按照 CT-CLIP 推理时的顺序提取英文 Keys
english_pathologies = list(disease_mapping.keys())

#pathologies = [
#    'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
#    'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
#    'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
#    'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
#    'Interlobular septal thickening'
#]

# 2. 加载数据
ctclip_probs = np.load(ctclip_weights_path)['data']  # 形状应为 (N, 18)
with open(accessions_path, 'r') as f:
    accession_names = [line.strip() for line in f.readlines()]

with open(qwen_results_path, 'r', encoding='utf-8') as f:
    qwen_labels = json.load(f)

# 3.筛查逻辑修改
for idx, case_id in enumerate(accession_names):
    if case_id not in qwen_labels:
        continue

    image_probs = ctclip_probs[idx]
    text_labels = qwen_labels[case_id]  # 这里的键是中文
    mismatch_details = []

    for p_idx, eng_disease in enumerate(english_pathologies):
        zh_disease = disease_mapping[eng_disease]  # 获取对应的中文名

        img_prob = image_probs[p_idx]
        text_label = text_labels.get(zh_disease, 0)  # 用中文名去Qwen的结果里取值

        # 对比逻辑不变
        # 错配情况A: 图像极度确信有病，但文本报告里写明没有/未提及
        if img_prob >= 0.8 and text_label == 0:
            mismatch_details.append(f"{zh_disease} (图像推断有病概率: {img_prob:.2f}, 文本报告提取: 无)")
        # 错配情况B: 图像极度确信没病，但文本报告明确写了有
        elif img_prob <= 0.2 and text_label == 1:
            mismatch_details.append(f"{zh_disease} (图像推断有病概率: {img_prob:.2f}, 文本报告提取: 有)")
    # 如果发现错配，则记录该病例准备剔除
    if len(mismatch_details) > 0:
        mismatched_cases.append({"case_id": case_id, "conflicts": mismatch_details})

# 4. 输出并保存脏数据名单
print(f"总计检查样本: {len(accession_names)} 例")
print(f"发现建议剔除的高危错配样本: {len(mismatched_cases)} 例\n")

with open('mismatched_blacklist.json', 'w', encoding='utf-8') as f:
    json.dump(mismatched_cases, f, indent=4, ensure_ascii=False)

for item in mismatched_cases[:5]:  # 打印前5个看看具体冲突原因
    print(f"病例 ID: {item['case_id']} | 冲突项: {', '.join(item['conflicts'])}")