# Controlled MATH Code-Tool Prompt Ablations

These prompts mirror the original MATH code-tool prompt from `data_inference/domain_math_tool_prompt.json`.

Controlled variables:
- Same system/task prefix.
- Same principles and output guidelines.
- Same few-shot question: `Find $1_6 + 2_6 + 3_6 + \cdots + 45_6$. Express your answer in base $6$.`
- Same mathematically correct few-shot target: `2003`.

Manipulated variable:
- The demonstrated trajectory contains 0, 1, or 2 tool calls.

Files:
- `original.txt`: exact original prompt copied from `data_inference/domain_math_tool_prompt.json`.
- `tools_0_3.txt`: three-stage reasoning trajectory with 0 tool calls.
- `tools_1_2.txt`: one complete code call computes the final answer directly, followed by final response.
- `tools_1_3.txt`: one code call for the first intermediate conversion, then reasoning-only completion.
- `tools_2_3.txt`: two code calls, close to the original prompt.

Important note:
- The original MATH few-shot prompt appears to contain a base-conversion inconsistency: `435` in base 10 is mathematically `2003_6`, but the source prompt says `2023`.
- The controlled variants use the correct answer `2003`, while `original.txt` remains an exact copy of the source prompt for baseline comparison.

Example:

```python
from envdata_smart import *

math_ds = [d for d in Datasets_Domain_Tool if "math" in d.name]
for prompt_name in ["original", "tools_0_3", "tools_1_2", "tools_1_3", "tools_2_3"]:
    envs = gen_prompt_envs(
        models=[ModelQwen25_7B, ModelLLaMA31_8B],
        base_datasets=math_ds,
        prompt=f"prompts/tool_prompt_ablation/controlled_math/{prompt_name}.txt",
        prompt_name=f"math_{prompt_name}",
        prompt_is_path=True,
        label=f"base_tool_math_{prompt_name}",
        mode="replace",
    )
```
