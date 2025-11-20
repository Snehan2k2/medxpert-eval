import os
import json
from utils import final_accuracy, final_accuracy_at_k
from model import APIAgent
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = "You are a helpful medical assistant."
USER_PROMPT = "Q: {question}\nA: Let's think step by step."

def init_file_if_needed(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass

def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def zero_shot_cot(input_sample, llm_agent, output, messages, folder_path):
    prompt = USER_PROMPT.format(question=input_sample['question'].strip())
    output['prompt'] = prompt
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    })

    # Add images to messages if present
    images = input_sample.get('images', [])
    if images:
        for image in images:
            image_url = image if isinstance(image, str) else os.path.join(folder_path, image.get('image_path', ''))
            messages[-1]["content"].append(llm_agent.image_content(image_url))

    response = llm_agent.get_response(messages)
    
    return response

def complete_item(llm_agent, input_sample, index, folder_path):
    try:
        print(f"### Starting Index: {index}")
        output = input_sample
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        response = zero_shot_cot(input_sample, llm_agent, output, messages, folder_path)
        print(f"### Completed Index: {index}")
        output['response'] = response

    except Exception as e:
        print("Error:", e)
        exit()

    return output, index

def general_inference(args, llm_agent):

    input_path = args.input_path
    tmp_path = os.path.join("outputs", f"{args.output_dir}.jsonl")

    os.makedirs("outputs", exist_ok=True)
    init_file_if_needed(tmp_path)

    # Load Inputs
    outputs = []
    with open(tmp_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line) for line in f if line.strip()]
    with open(input_path, 'r', encoding='utf-8') as f:
        inputs = [json.loads(line) for line in f if line.strip()]

    inputs = inputs[:6]

    start = len(outputs)
    end = len(inputs)

    assert end > 0

    batch_size = min(max(1, min(args.num_threads * 3, 500)), end - start)
    if start >= end:
        return
    
    for i in range(start, end, batch_size):
        batch_start, batch_end = i, min(i + batch_size, end)
        inputs_process = inputs[batch_start:batch_end]

        # Single Thread
        if args.num_threads == 1:
            for index, input_sample in enumerate(inputs_process):
                output, conf, _ = complete_item(llm_agent, input_sample, batch_start + index + 1, args.folder_path)
                outputs.append(output)
            write_to_file(tmp_path, outputs)

        # Multi Thread
        else:
            num_threads = min(args.num_threads, len(inputs_process))  # Ensure num_threads is less than or equal to the batch size
            chunk_size = max(1, len(inputs_process) // num_threads)  # Ensure chunk_size is at least 1
            futures = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for chunk_index in range(0, len(inputs_process), chunk_size):
                    chunk = inputs_process[chunk_index:chunk_index + chunk_size]
                    for index, item in enumerate(chunk):
                        futures.append(
                            executor.submit(
                                complete_item,
                                llm_agent,
                                item,
                                batch_start + chunk_index + index + 1,
                                args.folder_path
                            )
                        )

                results = [None] * len(inputs_process)
                for future in as_completed(futures):
                    output, index = future.result()
                    results[index - batch_start - 1] = output

                outputs.extend(results)

            write_to_file(tmp_path, outputs)

    if args.n == 1:
        final_accuracy(outputs)
    else:
        final_accuracy_at_k(outputs, args.n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default='result', type=str)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n", default=1, type=int, help="Number of completions to generate per prompt")
    parser.add_argument("--max-tokens", default=4096, type=int, help="Maximum tokens for LLM response")
    parser.add_argument("--folder-path", default="/project_pioneer/vaidya/medxpert/images", type=str, help="Path to the image directory")
    parser.add_argument("--input-path", default="/home/FRACTAL/snehan.j/medxpert/data/medxpertqa_text_input.jsonl", type=str, help="Path to the input file")
    parser.add_argument("--port", default=1111, type=int, help="Port number for the local LLM API server")
    args = parser.parse_args()

    llm_agent = APIAgent(args.temperature, args.max_tokens, args.n, args.port)
    general_inference(args, llm_agent)
