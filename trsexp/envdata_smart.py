from pathlib import Path
from lazyexp import exenv, exper, runners
from subprocess import Popen
import os
from lazyexp.exenv import *
import typing

from lazyexp.exenv import ExpEnv, Path
from envdata import *
from collections import defaultdict
import json

TEST_RUN = False

runner_inference_tool_prompt = runners.CmdExec(
    lambda env: [
        PYTHON,
        "inference/inference_tool_prompt.py",
        "--model_name_or_path",
        env.model.path,
        "--data_path",
        env.dataset.path,
        "--max_seq_length",
        "4096",
        "--save_path",
        env.get_output_path().as_posix(),
        "--test_start_id",
        "0",
        "--max_test_num",
        "-1",
        "--method",
        "mistral" if "mistral" in env.model.name else "llama",
    ],
    [],
    [Path("result.json")],
    name="inference_tool_prompt",
)

runner_inference_tool_prompt_less = runners.CmdExec(
    lambda env: [
        PYTHON,
        "inference/inference_tool_prompt.py",
        "--model_name_or_path",
        env.model.path,
        "--data_path",
        env.dataset.path,
        "--max_seq_length",
        "4096",
        "--save_path",
        env.get_output_path().as_posix(),
        "--test_start_id",
        "0",
        "--max_test_num",
        "-1",
        "--method",
        "mistral" if "mistral" in env.model.name else "llama",
        "--instruction",
        "\n\nPlease try to solve the problem without using tools if possible. Only use tools when you are sure that you cannot solve the problem with your own knowledge and reasoning. Remember, the more you rely on your own knowledge and reasoning, the better it is for your learning and growth. So think twice before using any tool, and use them as a last resort."
    ],
    [],
    [Path("result.json")],
    name="inference_tool_prompt",
)


runner_inference_smart = runners.CmdExec(
    lambda env: [
        PYTHON,
        "inference/inference_smart.py",
        "--model_name_or_path",
        env.model.path,
        "--data_path",
        env.dataset.path,
        "--max_seq_length",
        "4096",
        "--save_path",
        env.get_output_path().as_posix(),
        "--test_start_id",
        "0",
        "--max_test_num",
        "-1",
    ],
    [],
    [Path("result.json")],
    name="inference_smart",
)


runner_inference_eval = runners.CmdExec(
    lambda env: [
        PYTHON,
        f"evaluate/inference_eval_{env.dataset.tags['domain']}.py",
        "--data_path",
        env.get_output_path().as_posix(),
        "--save_path",
        env.get_output_path("smart_judged.json").as_posix(),
    ],
    [Path("result.json")],
    [Path("smart_judged.json")],
    name="inference_eval",
)


# runner_inference_eval = runners.skip_if_output_exists(runner_inference_eval, "smart_judged.json")

Datasets_Domain_Tool = []
Datasets_Domain_Smart = []
Datasets_OOD_Tool = []
Datasets_OOD_Smart = []
PROMPT_SMART_BASE = "You are an advanced assistant designed to solve tasks autonomously using your knowledge and reasoning. Clearly articulate your thought process and reasoning steps before presenting the final response to ensure transparency and accuracy."

for file in os.listdir("data_inference"):
    name = f"Dataset_{file.split('.')[0]}"
    # if "time" in name or "mint" in name:
    #     continue
    if "intention" in name:
        domain = "intention"
    elif "math" in name or "gsm" in name:
        domain = "math"
    else:
        domain = "time"
    d = DatasetEnv(
        f"data_inference/{file}",
        filetype="json",
        name=name,
        prompt_template=PROMPT_SMART_BASE + "\n\n{input}\n\n",
        tags={"domain": domain},
    )
    if "tool_prompt" in file:
        if "ood" in file:
            Datasets_OOD_Tool.append(d)
        else:
            Datasets_Domain_Tool.append(d)
    else:
        if "ood" in file:
            Datasets_OOD_Smart.append(d)
        else:
            Datasets_Domain_Smart.append(d)


ModelLLaMA31_8B_SMARTAgent = ModelEnv(
    "llama3.1-8b-smartagent",
    f"{DIR_DATA}/models/SMART/SMARTAgent-Llama-3.1-8B/",
    32,
)
ModelMistral_7B_SMARTAgent = ModelEnv(
    "mistral-7b-smartagent",
    f"{DIR_DATA}/models/SMART/SMARTAgent-Mistral-7B-Instruct-v0.3",
    32,
)

MODELS_BASE = [ModelLLaMA31_8B, ModelMistral_7B, ModelQwen25_7B]
MODELS_SMART = [ModelLLaMA31_8B_SMARTAgent, ModelMistral_7B_SMARTAgent]

PROMPT_SMART_JUDGE = """You are a helpful assistant to jusge whether the model's final response (might be word, phrase or sentence) and the given correct answer is same in value.
- If their intrinsic value of the answer is not equal, please mark it as wrong.
- If they are just expressed in different format or wording or unit, but have the same main value, please mark it as correct.
- Please output only "Correct" or "Wrong" as your judgment result in the final line.

- Model response: 
{model_output}

- Ground truth: 
{output}

- Judgment: """

def line_check_judge(model_output, sample):
    last_line = model_output.strip().split("\n")[-1].strip()
    res = {}
    if last_line in ["Correct", "Wrong"]:
        res["Acc"] = 1 if last_line=="Correct" else 0
        return res
    return None

class SmartJudgeFormater(runners.Runner):
    def __init__(self):
        super().__init__("smart_judge_formater", [Path("smart_judged.json")], [Path("smart_judge_formatted.json")])
        
    def run(self, exp_env: ExpEnv):
        outputs = json.load(exp_env.get_output_path(self.required_paths[0]).open("r"))
        task_results = {}
        for t in outputs:
            task = t['task']
            res = {}
            res["Tool Call"] = len([p for p in t['predict'] if p['type']=="tool"])
            if 'judge' in t:
                judgement = t['judge'].strip().split("\n")[-1].strip()
                if "wrong" in judgement.lower() or "incorrect" in judgement.lower():
                    res["Acc"] = 0
                elif "correct" in judgement.lower():
                    res["Acc"] = 1
                else:
                    res["Acc"] = 0
            else:  # For intention
                acc = defaultdict(list)
                for m in t["missing_results"]:
                    imp = int(m["importance"])
                    acc[imp].append(1 if m["judgment"] == "Yes" else 0)

                # Overall missing-detail recovery; optional, mainly for diagnosis.
                all_missing = [v for vals in acc.values() for v in vals]
                if all_missing:
                    res["Missing Details Recovery"] = sum(all_missing) / len(all_missing)

                for i, vals in acc.items():
                    if vals:
                        res[f"Missing Details Recovery Lv.{i}"] = sum(vals) / len(vals)

                summary_hits = [
                    1 if r["judgment"] == "Yes" else 0
                    for r in t.get("summary_results", [])
                ]
                if summary_hits:
                    res["Summarized Intention Coverage"] = sum(summary_hits) / len(summary_hits)
            task_results[task]=res
        l = []
        dataset = runners.get_dataset_cached(exp_env.dataset)
        for sample in dataset:
            t = sample["input"].split("### Task")[1].split("###")[0].strip()
            if t in task_results:
                l.append(task_results[t])
            else:
                l.append(None)
        json.dump(l, exp_env.get_output_path(self.output_paths[0]).open("w"))
            
        