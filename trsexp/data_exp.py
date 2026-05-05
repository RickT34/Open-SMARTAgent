from envdata_smart import *
from lazyexp import datapro
import json

envs_base:list[ExpEnv] = exenv.genEnvs(
        MODELS_BASE + MODELS_SMART, 
        Datasets_Domain_Tool+Datasets_OOD_Tool,
        [AlgoNULL],
        "base_tool"
    )
envs_smart:list[ExpEnv] = exenv.genEnvs(
        MODELS_SMART, 
        Datasets_Domain_Smart+Datasets_OOD_Smart,
        [AlgoNULL],
        "smart_tool"
    )

def mean(lst):
    return sum(lst)/len(lst) if len(lst) > 0 else 0.0

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
        MODELS_BASE + MODELS_SMART, 
        DATASETS_MATH,
        [AlgoNULL],
        "base_tool"
    )
    def data_fn(envs:list[ExpEnv]):
        assert len(envs) == 1
        env = envs[0]
        d = cate_by_tool_calls(env)
        tot = sum(len(lst) for lst in d)
        return np.arange(len(d)), {"Count": [len(d[i])/tot for i in range(len(d))], "Acc": [mean([acc for _, acc in d[i]]) for i in range(len(d))]}
        
    f = datapro.explot(envs_, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), datapro.BarPlotter(process_fn=data_fn))
    f.savefig(f"outputs/tool_calls_hist_{envs_[0].label}.png")
    
# def check_tool_calls_shift():
def acc_vs_toolcalls():
    envs = exenv.genEnvs(
        MODELS_BASE+MODELS_SMART,
        DATASETS_MATH,
        [AlgoNULL],
        "base_tool"
    )
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
        res = {k: mean(v) for k, v in res.items()}
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
        return list(range(K)), {"Acc": [mean(res[i]) for i in range(K)], "Count": [len(res[i])/len(data) for i in range(K)]}
    # f = datapro.explot(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), datapro.LinePlotter(process_fn=plot_fn))
    # f.savefig(f"outputs/acc_vs_tool_calls_{envs[0].label}.png")
    t = datapro.extable(envs, (datapro.ExpAxis.ModelAxis, datapro.ExpAxis.DatasetAxis), data_fn)
    print(t)
    t.to_markdown(f"outputs/acc_vs_tool_calls_{envs[0].label}.md")
    
    
acc_vs_toolcalls()
# draw_tool_calls_bar()