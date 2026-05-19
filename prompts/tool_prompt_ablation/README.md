# Tool Prompt Ablation Prompts

Use the controlled prompt folders for few-shot ablations. They keep each dataset's original tool prompt prefix, principles, output guidelines, and few-shot question fixed as much as possible, and vary the demonstrated solution trajectory.

Folders:

- `controlled_gsm/`: GSM8K-style Janet example, following the current edited file names (`original.txt`, `tools_0_3.txt`, `tools_1_2.txt`, `tools_1_3.txt`, `tools_2_3.txt`).
- `controlled_math/`: MATH base-conversion example from `data_inference/domain_math_tool_prompt.json`, with the same file naming scheme.
- `controlled_math_multi/`: multi-example variants for the MATH prompt. Currently includes `tools_1_2_multi3.txt`, which gives three independent one-complete-code-then-final trajectories.

Naming convention:

- `original.txt`: exact original prompt from the corresponding dataset.
- `tools_0_3.txt`: 0 tool calls over 3 staged examples.
- `tools_1_2.txt`: 1 complete tool call over 2 staged examples, then final response.
- `tools_1_3.txt`: 1 tool call over 3 staged examples.
- `tools_2_3.txt`: 2 tool calls over 3 staged examples, close to the original trajectory.
- `tools_1_2_multi3.txt`: 3 trajectory points, each using one complete tool call followed by final response.
