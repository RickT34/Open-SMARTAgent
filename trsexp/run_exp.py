from envdata import *
from envdata_smart import *
from lazyexp import exper, exenv, runners, datapro
import os
import json

def get_summery_table(envs:list[ExpEnv]):
    def process_fc(envs:list[ExpEnv]):
        assert len(envs) == 1
        return json.load(envs[0].get_output_path("summery.json").open())
    return datapro.extable(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis, datapro.ExpAxis.LabelAxis), process_fc)
envs_base:list[ExpEnv] = exenv.genEnvs(
        MODELS_BASE + MODELS_SMART, 
        Datasets_Domain_Tool+Datasets_OOD_Tool,
        [AlgoNULL],
        "base_tool"
    )
envs_base_less: list[ExpEnv] = exenv.genEnvs(
    MODELS_BASE + MODELS_SMART,
    Datasets_Domain_Tool + Datasets_OOD_Tool,
    [AlgoNULL],
    "base_tool_less",
)
envs_smart:list[ExpEnv] = exenv.genEnvs(
        MODELS_SMART, 
        Datasets_Domain_Smart+Datasets_OOD_Smart,
        [AlgoNULL],
        "smart_tool"
    )
envs_no_tool:list[ExpEnv] = exenv.genEnvs(
        MODELS_BASE + MODELS_SMART, 
        Datasets_Domain_Tool+Datasets_OOD_Tool,
        [AlgoNULL],
        "base_no_tool"
    )

import pandas as pd

def cross_ansly():
    assert len(envs_no_tool) == len(envs_base)
    res = {}
    for env_notool, env_tool in zip(envs_no_tool, envs_base):
        assert env_notool.model.name == env_tool.model.name
        model = env_notool.model.name
        assert env_notool.dataset.name == env_tool.dataset.name
        dataset = env_notool.dataset.name
        if env_notool.dataset.tags["domain"] == "intention":
            continue
        res_notool = json.load(
            env_notool.get_output_path("llmeval/result_cleared.json").open()
        )
        res_tool = json.load(env_tool.get_output_path("smart_judge_formatted.json").open())
        assert len(res_notool) == len(res_tool), f"{len(res_notool)} != {len(res_tool)} in {env_notool.model.name}-{env_notool.dataset.name}"
        assert (model, dataset) not in res
        d = defaultdict(float)
        for r_notool, r_tool in zip(res_notool, res_tool):
            if r_notool is None or len(r_notool) == 0 or r_tool is None:
                continue
            try:
                if r_tool["Tool Call"] == 0:
                    continue
                d[(f"notool_{r_notool['Acc']}", f"tool_{r_tool['Acc']}")] += 1/len(res_notool)
            except Exception as e:
                print(f"Error in {model}-{dataset}: {e}")
                print(r_notool)
                print(r_tool)
                continue
        res[(model, dataset)] = d
    t = pd.DataFrame(res)
    print(t)
    t.to_csv("outputs/cross_ansly.csv")

def cross_ansly_tool_vs_smart():
    
    envs_base_tool:list[ExpEnv] = exenv.genEnvs(
            MODELS_BASE[:2], 
            Datasets_Domain_Tool+Datasets_OOD_Tool,
            [AlgoNULL],
            "base_tool"
        )
    assert len(envs_base_tool) == len(envs_smart)
    res = {}
    for env_base, env_smart in zip(envs_base_tool, envs_smart):
        assert env_base.model.name + "-smartagent" == env_smart.model.name
        model = env_base.model.name
        assert env_base.dataset.name[:-len("_tool_prompt")] == env_smart.dataset.name[:-len("_smart")]
        dataset = env_base.dataset.name[:-len("_tool_prompt")]
        if env_base.dataset.tags["domain"] == "intention":
            continue
        res_base = json.load(
            env_base.get_output_path("smart_judge_formatted.json").open()
        )
        res_smart = json.load(env_smart.get_output_path("smart_judge_formatted.json").open())
        assert len(res_base) == len(res_smart), f"{len(res_base)} != {len(res_smart)} in {env_base.model.name}-{env_base.dataset.name}"
        assert (model, dataset) not in res
        d = defaultdict(list)
        for r_base, r_smart in zip(res_base, res_smart):
            if r_base is None or r_smart is None:
                continue
            try:
                d[(int(r_base['Acc']), int(r_smart['Acc']))].append((r_base, r_smart))
            except Exception as e:
                print(f"Error in {model}-{dataset}: {e}")
                print(r_base)
                print(r_smart)
                continue
        dd = {}
        for acc_bac, acc_smart in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            dd[(f"base_acc_{acc_bac}", f"smart_acc_{acc_smart}")] = {
                "count": len(d[(acc_bac, acc_smart)]),
                "ratio": len(d[(acc_bac, acc_smart)])/len(res_base),
                "tool_calls_base": sum(r[0]["Tool Call"] for r in d[(acc_bac, acc_smart)])/len(d[(acc_bac, acc_smart)]) if d[(acc_bac, acc_smart)] else 0,
                "tool_calls_smart": sum(r[1]["Tool Call"] for r in d[(acc_bac, acc_smart)])/len(d[(acc_bac, acc_smart)]) if d[(acc_bac, acc_smart)] else 0,
            }
        res[(model, dataset)] = dd
    print(res)
    t = pd.DataFrame(res)
    print(t)
    t.to_csv("outputs/cross_ansly_tool_vs_smart.csv")

def exp_inference_tool(judge:bool=True):
    if not judge:
        # 0 for a llm user by small vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        tasks = exper.gen_tasks(
            envs_base,
            runner_inference_tool_prompt,
        )
        tasks += exper.gen_tasks(
            envs_base_less,
            runner_inference_tool_prompt_less
        )
        tasks += exper.gen_tasks(
            envs_smart,
            runner_inference_smart,
        )
    else:
        ##Judge
        # 0,1,2,3 for a llm judge by big vllm model
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        envs = envs_base + envs_smart + envs_base_less
        workflow = runners.Workflow(
            "inference_tool_eval",
            [
                runner_inference_eval,
                SmartJudgeFormater(),
                runners.SummeryTable("smart_judge_formatted.json")
            ],
            skip_success=False,
            skip_exclude_paths=["smart_judge_formatted.json", "summery.json"],
            promised_paths=["result.json"]
        )
        workflow.info()
        # workflow.clear_outputs(envs, dry_dun=False)
        # return
        tasks = exper.gen_tasks(
            envs,
            workflow,
        )
    exper.run_tasks(tasks, ui=False)

def exp_inference_no_tool():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    workflow = runners.Workflow(
        "base_no_tool",
        [
            runners.EmvDump(),
            *runners.prefab_vllmeval(),
            *runners.prefab_llmeval(
                ModelQwen35_32B_AWQ,
                PROMPT_SMART_JUDGE,
                model_output_field="model_output",
            ),
            runners.LineCheck(
                line_check_judge, "llmeval/result.json", "llmeval/result_cleared.json"
            ),
            runners.SummeryTable("llmeval/result_cleared.json"),
        ],
        True,
        # skip_exclude_paths=["llmeval/result_cleared.json", "summery.json"],
    )
    workflow.info()
    tasks = exper.gen_tasks(
        envs_no_tool,
        workflow
    )
    exper.run_tasks(tasks, ui=False)

def summery():
    t = get_summery_table(envs_base+envs_smart+envs_no_tool)
    print(t)
    with open("outputs/summery.md", "w") as f:
        t.to_markdown(f)
    t.to_csv("outputs/summery.csv")

# exp_inference_tool(False)
exp_inference_tool(True)
# exp_inference_no_tool()
# summery()
# cross_ansly()
# cross_ansly_tool_vs_smart()
