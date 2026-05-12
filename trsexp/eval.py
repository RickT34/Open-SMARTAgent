from envdata_smart import *
from envdata import *
import numpy as np
 
def gen_4_cate(env_tool:ExpEnv):
    data = json.load(env_tool.get_output_path("smart_judge_formatted.json").open())
    path_notool = get_no_tool_env(env_tool).get_output_path("llmeval/result_cleared.json")
    data_no_tool = json.load(open(path_notool))
    res = np.zeros((2,2,2), dtype=np.long)
    for i, (r_notool, r_tool) in enumerate(zip(data_no_tool, data)):
        if r_notool is None or "Acc" not in r_notool or r_tool is None:
            continue
        acc_notool = int(r_notool["Acc"])
        acc_tool = int(r_tool["Acc"])
        tool_use = r_tool["Tool Call"] > 0
        res[acc_notool, acc_tool, tool_use] += 1
    return res


def tool_harm_info(env:ExpEnv):
    cate = gen_4_cate(env)
    tot = cate.sum()
    res = dict(
        harm = cate[1, 0, :].sum()/tot,
        rescue = cate[0, 1, :].sum()/tot,
        harm_tool = cate[1, 0, 1].sum()/tot,
        rescue_tool = cate[0, 1, 1].sum()/tot,
        overuse = cate[1, :, 1].sum()/tot
    )
    return res
    
    