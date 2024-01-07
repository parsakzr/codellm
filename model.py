import torch
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from pydantic import BaseModel, ConfigDict
from typing import Optional, Tuple


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str = ""
    model: Optional[PreTrainedModel] = None
    model_name: str = ""
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda:0"
    load_8bit: bool = False
    max_input_length: int = 512
    max_output_length: int = 512

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_path != "":
            self.load()
        # else:
        #     self.load_from(**kwargs)

    def load_from(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # self.model_name = model_name

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                self.model = self.model.merge_and_unload()
            self.model.eval()
            if not self.load_8bit or not self.load_4bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # mandatory for prompt style
        self.model_name = self.model_path.split("/")[-1]

    def generate(
        self,
        prompt: str,
        prompt_mode: bool = True,
        verbose: bool = False,
        pure_mode: bool = False,
        **kwargs,
    ):
        def generate_prompt(text: str, *, prompt_template: str = ""):
            if prompt_template:
                try:
                    prompt = prompt_template.format(text)
                except KeyError:
                    # if the prompt template is not valid, use the original prompt
                    print(
                        "Invalid prompt template, using original prompt. Make sure to include {} in the template."
                    )
                    return text
                return prompt

            if "codegen" in self.model_name.lower():
                # style "alpaca":
                system_msg = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                return system_msg + f"### Instruction: {text}\n\n### Output:\n"

            if "mistral" in self.model_name.lower():
                system_msg = "Below is an instruction that describes a programming task. Write a response code that appropriately completes the request.\n"
                return f"<s>[INST] {system_msg}\n{text} [/INST]"

            return text

        # from kwargs if style is specified
        prompt_template = kwargs.get("prompt_template", "")
        if prompt_mode:
            prompt = generate_prompt(prompt, prompt_template=prompt_template)

        if verbose:
            print(f"------------ Prompt -------------\n{prompt}")

        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids
        if not self.load_8bit:
            input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # avoid warning
            **kwargs,
            # no_repeat_ngram_size=1,
            # early_stopping=True,
            # num_beams=2,
            # temperature=0.1,
            # do_sample=True,
        )

        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if pure_mode:
            try:
                # remove the prompt, since it's a completion model
                output = output.replace(prompt, "")
                # select the text between the two '''
                output = output.split("'''")[1]
                # remove the first line (which is the language)
                output = "\n".join(output.split("\n")[1:])
            except:
                pass
        if verbose:
            print(f"-------- Generated Output --------\n{output}")

        return output

    def run(self, prompt: str, **kwargs):  # TODO: temp fix, no need
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids
        if not self.load_8bit:
            input_ids = input_ids.to(self.device)

        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # avoid warning
            **kwargs,
        )

        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output


def test_Model():
    model_id = "Salesforce/codegen-350M-mono"
    model = EvalModel(model_path=model_id, device="cpu", load_4bit=True)

    print(model.model_name)
    prompt = "Create a function to print Hello world!"

    output = model.generate(
        prompt, prompt_mode=False, verbose=True, temperature=0.5, do_sample=True
    )

    # print(output)


if __name__ == "__main__":
    test_Model()
