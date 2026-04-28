from lazyexp import exenv, exper
from subprocess import Popen

def runner_inference_tool_prompt(env: exenv.ExpEnv):
    p = Popen([f"python inference/inference_tool_prompt.py",
             f"--model_name_or_path {env.model.path}",
             f"--data_path {env.dataset.path}",
             f"--max_seq_length {env.param.max_seq_length}",
             f"--save_path {env.get_output_path()}",
             f"--test_start_id {env.param.test_start_id}",
             f"--max_test_num {env.param.max_test_num}",
             f"--method {exenv.ModelEnv.name}"])
    p.wait()