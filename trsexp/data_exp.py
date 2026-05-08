from envdata_smart import *
from lazyexp import datapro
import json
from run_exp import envs_base, envs_smart, envs_no_tool, envs_base_less

envs_exptools = exenv.genEnvs(
    MODELS_BASE,
    Datasets_Domain_Tool + Datasets_OOD_Tool,
    [AlgoNULL],
    "base_tool"
) + exenv.genEnvs(
    MODELS_SMART,
    Datasets_Domain_Smart + Datasets_OOD_Smart,
    [AlgoNULL],
    "smart_tool"
)


def get_summery_table(envs:list[ExpEnv]):
    def process_fc(envs:list[ExpEnv]):
        assert len(envs) == 1
        return json.load(envs[0].get_output_path("summery.json").open())
    return datapro.extable(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis, datapro.ExpAxis.LabelAxis), process_fc)
def mean(lst):
    return sum(lst)/len(lst) if len(lst) > 0 else 0.0

def gen_4_cate(env_tool:ExpEnv):
    data = json.load(env_tool.get_output_path("smart_judge_formatted.json").open())
    path_notool = env_tool.get_output_path("llmeval/result_cleared.json").as_posix().replace("base_tool", "base_no_tool").replace("smart_tool", "base_no_tool").replace("_smart", "_tool_prompt")
    data_no_tool = json.load(open(path_notool))
    res = defaultdict(list)
    for i, (r_notool, r_tool) in enumerate(zip(data_no_tool, data)):
        if r_notool is None or r_tool is None:
            continue
        acc_notool = int(r_notool["Acc"])
        acc_tool = int(r_tool["Acc"])
        res[(acc_notool, acc_tool)].append(i)
    return res


def gen_4_cate2(env1: ExpEnv, env2: ExpEnv):
    data = json.load(env1.get_output_path("smart_judge_formatted.json").open())
    data_no_tool = json.load(env2.get_output_path("smart_judge_formatted.json").open())
    res = defaultdict(list)
    for i, (r_notool, r_tool) in enumerate(zip(data_no_tool, data)):
        if r_notool is None or r_tool is None:
            continue
        acc_notool = int(r_notool["Acc"])
        acc_tool = int(r_tool["Acc"])
        res[(acc_notool, acc_tool)].append(i)
    return res


import pandas as pd
import itertools

def figure2():
    dataset = None
    for d in Datasets_OOD_Tool:
        if d.tags["domain"] == "math":
            dataset = d
            break
    assert dataset is not None
    for model in [ModelLLaMA31_8B, ModelMistral_7B, ModelQwen25_7B]:
        env = ExpEnv(model, dataset, AlgoNULL, "base_tool")
        env_notool = ExpEnv(model, dataset, AlgoNULL, "base_no_tool")
        data = json.load(env.get_output_path("smart_judge_formatted.json").open())
        data_no_tool = json.load(env_notool.get_output_path("llmeval/result_cleared.json").open())
        res = defaultdict(int)
        tot =0
        for r_notool, r_tool in zip(data_no_tool, data):
            if r_notool is None or r_tool is None:
                continue
            acc_notool = int(r_notool["Acc"])==1
            acc_tool = int(r_tool["Acc"])==1
            tool_use = r_tool["Tool Call"] > 0
            res[(acc_notool, acc_tool, tool_use)] += 1
            tot += 1
        print(f"Model: {model.name}")
        print("Tool Used")
        for tool_acc in [True, False]:
            for notool_acc in [False, True]:
                print(res[(notool_acc, tool_acc, True)]/tot*100, end="\t")
            print()
        print("Tool Not Used")
        print(sum(res[(notool_acc, tool_acc, False)] for notool_acc, tool_acc in itertools.product([False, True], [False, True]))/tot*100)

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

def cate_by_tool_calls(env:ExpEnv, max_calls=10):
    MAXT = max_calls
    res = [[] for _ in range(MAXT+1)]
    data = json.load(env.get_output_path("smart_judge_formatted.json").open())
    for i, d in enumerate(data):
        if d is not None:
            calls = d["Tool Call"]
            acc = d.get("Acc", None)
            if acc is not None:
                res[min(calls, MAXT)].append((i, acc))
    return res
import numpy as np
DATASETS_MATH = [d for d in Datasets_Domain_Tool+Datasets_OOD_Tool if d.tags["domain"] == "math"]
def draw_tool_calls_bar():
    envs_ = exenv.genEnvs(
        [ModelLLaMA31_8B, ModelMistral_7B], 
        DATASETS_MATH,
        [AlgoNULL],
        "base_tool"
    )
    def data_fn(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        env_smart = get_smart_env(env)
        d = cate_by_tool_calls(env)
        d_smart = cate_by_tool_calls(env_smart)
        tot = sum(len(lst) for lst in d)
        return np.arange(len(d)), {"Base": [len(d[i])/tot for i in range(len(d))], "Smart": [len(d_smart[i])/tot for i in range(len(d_smart))]}
    def data_fn_acc(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        env_smart = get_smart_env(env)
        d = cate_by_tool_calls(env)
        d_smart = cate_by_tool_calls(env_smart)
        tot = sum(len(lst) for lst in d)
        return np.arange(len(d)), {"Base": [mean([acc for _, acc in d[i]]) for i in range(len(d))], "Smart": [mean([acc for _, acc in d_smart[i]]) for i in range(len(d_smart))]}
    f = datapro.explot(envs_, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), datapro.BarPlotter(process_fn=data_fn), xlabel="Tool Calls", translator=translate)
    f.savefig(f"outputs/tool_calls_hist_cnt.png")
    f = datapro.explot(envs_, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), datapro.BarPlotter(process_fn=data_fn_acc), xlabel="Tool Calls", translator=translate)
    f.savefig(f"outputs/tool_calls_hist_acc.png")
# def check_tool_calls_shift():
def acc_vs_toolcalls():
    envs = [e for e in envs_exptools if e.dataset.tags['domain'] != "intention"]
    def data_fn(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        data = json.load(env.get_output_path("smart_judge_formatted.json").open())
        K = 5
        res = {f"Tool Calls={i}": [] for i in range(K)}
        res[f"Tool Calls>={K}"] = []
        for t in data:
            if t is not None:
                calls = t["Tool Call"]
                acc = t.get("Acc", None)
                if acc is not None:
                    if calls < K:
                        res[f"Tool Calls={calls}"].append(acc)
                    else:
                        res[f"Tool Calls>={K}"].append(acc)
        res = {k: mean(v)*100 for k, v in res.items()}
        return res
    def plot_fn(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        data = json.load(env.get_output_path("smart_judge_formatted.json").open())
        K = 15
        res = {i:[] for i in range(K)}
        for t in data:
            if t is not None:
                calls = t["Tool Call"]
                acc = t.get("Acc", None)
                if acc is not None:
                    res[min(calls, K-1)].append(acc)
        return list(range(K)), {"Acc": [mean(res[i]) for i in range(K)], "Ratio": [len(res[i])/len(data) for i in range(K)]}
    
    # f = datapro.explot(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), datapro.LinePlotter(process_fn=plot_fn))
    # f.savefig(f"outputs/acc_vs_tool_calls_{envs[0].label}.png")
    t = datapro.extable(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), data_fn, translate)
    print(t)
    # t.to_markdown(f"outputs/acc_vs_tool_calls_{envs[0].label}.md")
    t.to_csv(f"outputs/acc_vs_tool_calls.csv")

def summery():
    t = get_summery_table(envs_base+envs_smart+envs_no_tool+envs_base_less)
    print(t)
    with open("outputs/summery.md", "w") as f:
        t.to_markdown(f)
    t.to_csv("outputs/summery.csv")

def data_harm_rescue():
    def data_fn(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        data = gen_4_cate(env)
        tot = sum(len(lst) for lst in data.values())
        return {"Harm": len(data[(1, 0)])/tot*100, "Rescue": len(data[(0, 1)])/tot*100}
    t = datapro.extable([e for e in envs_exptools if e.dataset.tags['domain'] != "intention"], (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), data_fn, translator=translate)
    print(t)
    t.to_csv("outputs/data_harm_rescue.csv")

def data_harm_smart():
    envs = exenv.genEnvs(
        MODELS_BASE[:2], 
        Datasets_Domain_Tool + Datasets_OOD_Tool,
        [AlgoNULL],
        "base_tool"
    )
    def data_fn(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        data = gen_4_cate2(env, get_smart_env(env))
        tot = sum(len(lst) for lst in data.values())
        return {"Harm": len(data[(1, 0)])/tot*100, "Rescue": len(data[(0, 1)])/tot*100}
    t = datapro.extable([e for e in envs if e.dataset.tags['domain'] != "intention"], (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), data_fn, translator=translate)
    print(t)
    t.to_csv("outputs/data_harm_smart.csv")

# acc_vs_toolcalls()
# summery()
# draw_tool_calls_bar()
# figure2()
# data_harm_rescue()
data_harm_smart()
