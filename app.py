import re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import EvalModel


# MY CODE
def filter_code(completion: str, prompt: str = None, template: str = "") -> str:
    try:
        code = completion
        if prompt is not None:
            # remove the prompt, since it's a completion model
            code = code.replace(prompt, "")
        if template == "alpaca":
            ## Remove boilerplate for the function, reused pure_mode in generation_pipeline
            # select the text between the two '''
            code = code.split("'''")[1]
            # remove the first line (which is the language)
            code = "\n".join(code.split("\n")[1:])
        if template == "mistral":
            # get the code inside [CODE] ... [/CODE]
            code = code.split("[CODE]")[1]
            code = code.split("[/CODE]")[0]
        ## The program tends to overwrite, we only take the first function
        code = code.lstrip("\n")
        return code.split("\n\n")[0]
    except Exception as e:
        print(e)
        return code
    finally:
        return code


# /MYCODE


def extract_code_codegen(input_text):
    pattern = r"'''py\n(.*?)'''"
    match = re.search(pattern, input_text, re.DOTALL)

    if match:
        return match.group(1)
    else:
        return None  # Return None if no match is found


def extract_code_mistral(input_text):
    pattern = r"\[CODE\](.*?)\[/CODE\]"
    match = re.search(pattern, input_text, re.DOTALL)

    if match:
        return match.group(1)
    else:
        return None  # Return None if no match is found


def generate_code(input_text, modelName):
    if modelName == "codegen-350M":
        input_ids = codeGenTokenizer(input_text, return_tensors="pt").input_ids
        generated_ids = codeGenModel.generate(input_ids, max_length=128)
        result = codeGenTokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return extract_code_codegen(result)
    elif modelName == "mistral-7b":
        input_ids = mistralTokenizer(
            generate_prompt_mistral(input_text), return_tensors="pt"
        ).input_ids
        generated_ids = mistralModel.generate(input_ids, max_length=128)
        result = mistralTokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return extract_code_mistral(result)
    else:
        return None


def generate_prompt_mistral(text):
    system_msg = "Below is an instruction that describes a programming task. Write a response code that appropriately completes the request.\n"
    return f"<s>[INST] {system_msg}\n{text} [/INST]"


def respond(message, chat_history, additional_inputs):
    return f"Here's an example code:\n\n```python\n{generate_code(message,additional_inputs)}\n```"


# MAIN ----------------------------------------------
model_ids = {
    "codegen": "parsak/codegen-350M-mono-lora-instruction",
    "mistral": "parsak/mistral-code-7b-instruct",
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

mistralModel = AutoModelForCausalLM.from_pretrained(
    model_ids["mistral"], quantization_config=bnb_config, device_map={"": 0}
)
# tokenizer
mistralTokenizer = AutoTokenizer.from_pretrained(model_ids["mistral"])
mistralTokenizer.pad_token = mistralTokenizer.eos_token
mistralTokenizer.padding_side = "right"

mistralEvalModel = EvalModel()
mistralEvalModel.load_from(mistralModel, mistralTokenizer)
mistralEvalModel.model_name = model_ids["mistral"].split("/")[-1]

# CodeGen
codegenModel = AutoModelForCausalLM.from_pretrained(
    model_ids["codegen"], quantization_config=bnb_config, device_map={"": 0}
)
# tokenizer
codegenTokenizer = AutoTokenizer.from_pretrained(model_ids["codegen"])
codegenTokenizer.pad_token = codegenTokenizer.eos_token
codegenTokenizer.padding_side = "right"

codegenEvalModel = EvalModel()
codegenEvalModel.load_from(codegenModel, codegenTokenizer)
codegenEvalModel.model_name = model_ids["codegen"].split("/")[-1]


dropdown = gr.Dropdown(
    label="Models", choices=["codegen-350M", "mistral-7b"], value="codegen-350M"
)

interface = gr.ChatInterface(
    respond,
    retry_btn=gr.Button(value="Retry"),
    undo_btn=None,
    clear_btn=gr.Button(value="Clear"),
    additional_inputs=[dropdown],
)


if __name__ == "__main__":
    interface.launch()
