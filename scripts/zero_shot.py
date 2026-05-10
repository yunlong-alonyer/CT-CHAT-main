from pathlib import Path
from transformers import BertTokenizer
from eval import evaluate_internal
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_inference_nii import CTReportDatasetinfer
import numpy as np
import tqdm
import pandas as pd
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from ct_clip import CTCLIP


# helpers
def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        meta_file: "meta_data.csv",
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(data_folder=data_folder, reports_file=reports_file, meta_file=meta_file, labels=labels)

        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=6,
            batch_size=1,
            shuffle = True,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
 			self.dl_iter,
            self.CTClip,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
        )

        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 替换 scripts/zero_shot.py 中的 infer 函数
    def infer(self, log_fn=noop):
        device = self.device
        steps = int(self.steps.item())
        logs = {}

        with torch.no_grad():
            models_to_evaluate = ((self.CTClip, str(steps)),)

            for model, filename in models_to_evaluate:
                model.eval()
                predictedall = []
                accession_names = []

                # 固定的18种疾病
                pathologies = [
                    'Emphysema',
                    'Atelectasis',
                    'Consolidation',
                    'Lung nodule',
                    'Bronchiectasis',
                    'Pleural effusion',
                    'Pericardial effusion',
                    'Arterial wall calcification',#字面上不完全一致
                    'Lung opacity',#字面上不完全一致
                    'Interlobular septal thickening'#字面上不完全一致
                ]

                for i in tqdm.tqdm(range(len(self.ds))):
                    # 获取数据，忽略 text 和 onehotlabels（即使你的Dataloader返回了空值或伪造值也没关系）
                    valid_data, _, _, acc_name = next(self.dl_iter)

                    plotdir = self.result_folder_txt
                    Path(plotdir).mkdir(parents=True, exist_ok=True)

                    predictedlabels = []

                    # 遍历18种疾病构建Prompt进行零样本推理
                    for pathology in pathologies:
                        text = [f"{pathology} is present.", f"{pathology} is not present."]
                        text_tokens = self.tokenizer(
                            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
                        ).to(device)

                        # 推理
                        output = model(text_tokens, valid_data.cuda(), device=device)
                        output = apply_softmax(output)  # 应用 Softmax，获取存在/不存在的概率

                        append_out = output.detach().cpu().numpy()
                        predictedlabels.append(append_out[0])  # 索引0对应 "is present" 的概率

                    predictedall.append(predictedlabels)
                    accession_names.append(acc_name[0])

                predictedall = np.array(predictedall)

                # 仅保存预测权重和病例ID，不计算 AUROC
                np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                with open(f"{plotdir}accessions.txt", "w") as file:
                    for item in accession_names:
                        file.write(item + "\n")

        self.steps += 1
        log_fn(logs)
        self.print('Inference complete. Weights saved for screening.')
    self.print('Inference complete')