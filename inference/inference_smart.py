import json
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
from cprint import *

from utils_serper import search_serper
from utils_askuser import simulate_user_response
from utils_code import execute_code

FINAL_RESPONSE_MARKERS = ("### Final Response", "Final Answer:")


def split_final_response(text):
    for marker in FINAL_RESPONSE_MARKERS:
        if marker in text:
            prefix, suffix = text.split(marker, 1)
            return prefix.strip(), suffix.strip()
    return None, None


def parse_steps(text):
    results = []
    steps = text.strip().split("\n- Step")
    steps = [step.strip() for step in steps if step.strip() != ""]
    for step in steps:
        reasoning, final_output = split_final_response(step)
        if final_output is not None:
            results.append({
                "name": "Final Response",
                "type": "normal",
                "tool_name": None,
                "reasoning": final_output
            })
            continue
        try:
            first_line = step.split("\n")[0]
            index = first_line.rfind('(')
            name = first_line[:index].split(":")[1].strip()
            step_type = first_line[index+1:].split(")")[0].strip()
            reasoning_type = "tool" if step_type.startswith("tool") else "normal"
            tool_name = step_type.split(":")[1].strip() if reasoning_type == "tool" else None
            reasoning = "\n".join(step.split("\n")[1:]).strip()
            results.append({
                "name": name,
                "type": reasoning_type,
                "tool_name": tool_name,
                "reasoning": reasoning
            })
        except Exception as e:
            print(e)
    return results


def read_prompt_override(prompt_text="", prompt_path=""):
    if prompt_path:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    if prompt_text:
        return prompt_text
    return None


def apply_prompt_override(instruction, prompt_override=None, prompt_mode="replace"):
    if prompt_override is None:
        return instruction
    if prompt_mode == "replace":
        return prompt_override
    if prompt_mode == "append":
        return instruction.rstrip() + "\n\n" + prompt_override.lstrip()
    if prompt_mode == "prepend":
        return prompt_override.rstrip() + "\n\n" + instruction.lstrip()
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def preprocess_dataset(data_path, max_num=-1, start_id=0, prompt_override=None, prompt_mode="replace"):
    """
    Load and preprocess the dataset by applying the chat template.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    max_num = len(data) if max_num == -1 else max_num
    dataset = []
    for d in data[start_id:]:
        task = d["input"].split("### Task")[1].split("###")[0].strip()
        sample_instruction = apply_prompt_override(
            d["instruction"],
            prompt_override=prompt_override,
            prompt_mode=prompt_mode,
        )
        
        messages = [
                {
                    "role": "user",
                    "content": sample_instruction.strip() + "\n\n" + d["input"].strip()
                }
        ]
        dataset.append({"input": messages, "ground_truth": d["output"], "task": task})
        
        if len(dataset) >= max_num:
            break
    
    print(f"Length of data: {len(dataset)}")
    return dataset


def format_steps(steps):
    results = []
    for idx, step in enumerate(steps):
        name = step["name"]
        type = f"tool: {step['tool_name']}" if step['type'] == "tool" else "general reasoning"
        reason = step["reasoning"].strip()
        result = f"- Step {idx+1}: {name} ({type})\n{reason}"
        if "output" in step:
            output = step["output"]
            result += f"\n- Output: {output}"
        results.append(result.strip())
    return "\n\n".join(results)

 
def inference(args):
    """
    Perform inference using the pipeline API.
    """
    # Load the model using vLLM
    print("Loading model with vLLM...")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=int(os.getenv("WORLD_SIZE", 1)),
        gpu_memory_utilization=0.92,
        max_model_len=args.max_seq_length,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_seq_length,
        temperature=0.000001,
    )

    # Preprocess dataset
    print("Loading and preprocessing dataset...")
    prompt_override = read_prompt_override(args.prompt_text, args.prompt_path)
    dataset = preprocess_dataset(
        args.data_path,
        max_num=args.max_test_num,
        start_id=args.test_start_id,
        prompt_override=prompt_override,
        prompt_mode=args.prompt_mode,
    )
    
    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path, "r", encoding="utf-8"))
        existing_tasks = [r["task"] for r in results]
    else:
        results = []
        existing_tasks = []
    
    print(f"Length of existing data: {len(results)}")
    
    # Perform inference
    print("Starting inference...")
    log = {"fail": 0, "success": 0}
    example_count = 0

    for example in tqdm(dataset):
        example_count += 1
        input_messages = example["input"]
        ground_truth = example["ground_truth"]
        task = example["task"]
        
        example_word = task.split(" ")[0]

        if task in existing_tasks:
            continue
        
        steps = []
        raw = []
        step_time = 0
        
        while True:
            try:
                step_time += 1
                if step_time > 10:
                    # Change here to save those instances even if exceeding maximum length
                    steps.append({
                        "name": "Final Response",
                        "type": "normal",
                        "tool_name": None,
                        "reasoning": "Still do not get an answer after exceeding maximum step time! Please judge the answer for this question as wrong."
                    })
                    results.append({
                        "task": task,
                        "predict": steps,
                        "ground_truth": ground_truth,
                        "raw": raw
                    })
                    with open(args.save_path, "w") as f:
                        json.dump(results, f, indent=2)
                    break
                
                result = llm.chat(input_messages, sampling_params)
                assistant_output = result[0].outputs[0].text.strip()
                
                raw.append(assistant_output)
                
                cprint.info("\n\n", "+" * 10, "Round Response", "+" * 10)
                print(assistant_output)
                 
                new_steps, final_output = split_final_response(assistant_output)
                if final_output is not None:
                    new_steps = new_steps.strip()
                    steps.extend(parse_steps(new_steps))
                
                    steps.append({
                        "name": "Final Response",
                        "type": "normal",
                        "tool_name": None,
                        "reasoning": final_output
                    })
                    
                    log["success"] += 1
                    cprint.info("\n\n", "+" * 10, "Ground Truth", "+" * 10)
                    print(ground_truth)
                            
                    results.append({
                        "task": task,
                        "predict": steps,
                        "ground_truth": ground_truth,
                        "raw": raw
                    })
                    
                    with open(args.save_path, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    print(log)
                    break
                
                else:
                    steps.extend(parse_steps(assistant_output))
                    last_step = steps[-1]
                    if last_step["type"] == "tool":
                        tool_name = last_step["tool_name"]
                        if tool_name == "AskUser":
                            cprint.info("AskUser tool detected")
                            response = simulate_user_response(task, last_step["reasoning"])
                            steps[-1]["output"] = response
                        elif tool_name == "Search":
                            cprint.ok("Search tool detected")
                            if "intention" in args.data_path:
                                link = True
                            else:
                                link = False
                            response = search_serper(last_step["reasoning"], link=link)
                            steps[-1]["output"] = response
                        elif tool_name == "Code":
                            cprint.warn("Code tool detected")
                            response = execute_code(last_step["reasoning"], file_name=f"./env/{str(example_count)}_{example_word}.py")
                            steps[-1]["output"] = response
                        else:
                            assert False, "Unknown tool name"
                        cprint.info("\n\n", "+" * 10, f"Tool {tool_name} Response", "+" * 10)
                        print(response)
                # update the input message
                input_messages[-1]["content"] = input_messages[-1]["content"].split("### Task")[0] + "### Task\n" + task + "\n### Reasoning Steps\n" + format_steps(steps)
            
            except Exception as e:
                log["fail"] += 1
                print(e)
                break
        

def initialize():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the inference results")
    parser.add_argument("--test_start_id", type=int, default=0, help="The start id for testing")
    parser.add_argument("--max_test_num", type=int, default=-1, help="The max number of instances to test")
    parser.add_argument("--prompt_text", type=str, default="", help="Override dataset instruction with this prompt text")
    parser.add_argument("--prompt_path", type=str, default="", help="Override dataset instruction with prompt text read from this file")
    parser.add_argument("--prompt_mode", type=str, default="replace", choices=["replace", "append", "prepend"], help="How to apply prompt_text/prompt_path to the dataset instruction")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = initialize()
    inference(args)
