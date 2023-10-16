import os
import sys
import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Device type: {device}")

def main(
    prompt_path,
    dest_folder,
    base_model: str = "",
    lora: str = "",
    seed: int = 1,
    max_tokens: int = 200,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0 #unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    # if not load_in_4bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.7,
        top_p=0.80,
        top_k=60,
        repetition_penalty= 1.2,
        max_new_tokens=max_tokens,
        **kwargs,
    ):
        # set seed
        set_seed(seed)

        inputs = tokenizer(instruction, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=0, #unk
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output

    def evaluateFile(prompt_file_path, dest_folder):
        print("Prompt file:", prompt_file_path)

        with open(prompt_file_path, 'r', encoding='utf-8') as file_in:
            instruction = file_in.read()
            print("Prompt:\n", instruction)

        print("\nResponse:\n")
        response = evaluate(instruction)
        print(response)

        # Create dest_folder if not exists
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Save the response in dest_folder
        dest_file_path = os.path.join(dest_folder, f"{os.path.basename(prompt_file_path)}.result.txt")
        with open(dest_file_path, 'w', encoding='utf-8') as file_out:
            file_out.write(response)


    if os.path.isdir(prompt_path):
        for filename in os.listdir(prompt_path):
            if filename.endswith(".prompt"):
                prompt_file_path = os.path.join(prompt_path, filename)
                evaluateFile(prompt_file_path, dest_folder)
    elif os.path.isfile(prompt_path):
        evaluateFile(prompt_path, dest_folder)
    else:
        print("Invalid prompt_path")

if __name__ == "__main__":
    fire.Fire(main)
