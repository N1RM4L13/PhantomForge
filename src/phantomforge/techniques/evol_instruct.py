depth_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

breadth_instruction = "I want you act as a Prompt Creator.\r\n\
					Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
					This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
					The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
					The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
					'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

comparison_instruction = "Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:\r\n\
					1. They have same constraints and requirements.\r\n\
					2. They have same depth and breadth of the inquiry.\r\n\
					The First Prompt: <Here is first instruction.>\r\n\
					The Second Prompt: <Here is second instruction.>\r\n\
					Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason.):\r\n"

def createBreadthPrompt(instruction):
    prompt = breadth_instruction
    prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Created Prompt#:\r\n"
    return prompt


def createConstraintsPrompt(instruction):
    prompt = depth_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createDeepenPrompt(instruction):
    prompt = depth_instruction.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createConcretizingPrompt(instruction):
    prompt = depth_instruction.format("Please replace general concepts with more specific concepts.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createReasoningPrompt(instruction):
    prompt = depth_instruction.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createComparisonEliminatorPrompt(before, after):
    prompt = comparison_instruction
    prompt = prompt.replace("<Here is first instruction.>", before)
    prompt = prompt.replace("<Here is second instruction.>", after)
    return prompt

import random
# from openai_generate import OpenAI, AzureOpenAI
from src.phantomforge.utils import *
import os


class EvolInstruct:
    def __init__(self, num_iterations, initial_dataset, evol_model="gpt-4", eliminator_model="gpt-35-turbo",
                 azure_config_path=None, use_requests=True, verbose=False):

        self.num_iterations = num_iterations
        self.dataset = initial_dataset
        self.verbose = verbose
        # if azure_config_path:
        #     config = load_config(azure_config_path)
        #     self.evol_model = AzureOpenAI(config, evol_model, use_requests=use_requests)
        #     self.eliminator_model = AzureOpenAI(config, eliminator_model,
        #                                         use_requests=use_requests) if eliminator_model else None
        # else:
        #     openai_api_key = os.environ.get("OPENAI_API_KEY")
        #     self.evol_model = OpenAI(openai_api_key, evol_model, use_requests=use_requests)
        #     self.eliminator_model = OpenAI(eliminator_model, use_requests=use_requests) if eliminator_model else None

        self.evol_model_kwargs = {"temperature": 1, "max_tokens": 2048, "top_p": 0.95, "frequency_penalty": 0,
                                  "presence_penalty": 0, "stop": None}

        self.breadth_functions = {
            "Breadth": createBreadthPrompt,
        }
        self.depth_functions = {
            "Constraints": createConstraintsPrompt,
            "Deepen": createDeepenPrompt,
            "Concretizing": createConcretizingPrompt,
            "Reasoning": createReasoningPrompt,
        }
        self.evol_functions = self.breadth_functions | self.depth_functions

        self.eliminator_functions = {
            "Comparison": createComparisonEliminatorPrompt,
        }

    def select_prompt(self, instruction, function_names):
        # Select a prompt for in-depth or in-breadth evolution randomly
        function = self.evol_functions[random.choice(function_names)]
        return function(instruction)

    def instruction_evolver(self, instruction, function_names=None):
        if function_names is None:
            function_names = list(self.evol_functions.keys())
        prompt = self.select_prompt(instruction, function_names)
        return self.evol_model.generate(prompt, **self.evol_model_kwargs)

    def instruction_eliminator(self, original_instruction, evolved_instruction, response):
        # Rule 1: The evolved instruction does not provide any information gain compared to the original one.
        if original_instruction == evolved_instruction:
            return False, "Equal"
        if self.eliminator_model:
            prompt = createComparisonEliminatorPrompt(original_instruction, evolved_instruction)
            comparison_response = self.eliminator_model.generate(prompt, **self.evol_model_kwargs)
            if "equal" in comparison_response.lower():
                return False, "Equal"

        # Rule 2: The evolved instruction makes it difficult for the LLM to generate a response.
        if 'sorry' in response.lower() and "sorry" not in original_instruction.lower() and len(response.split()) < 80:
            return False, "Sorry"

            # Rule 3: The response generated by the LLM only contains punctuation and stop words.
            # Not sure if I should use this rule

        # Rule 4: The evolved instruction obviously copies some words from the evolving prompt.
        if any(phrase in evolved_instruction.lower() for phrase in ['given prompt', 'rewritten prompt', 'new prompt']):
            return False, "Leak from Prompt"

        return True, "Success"

    def evolve(self):
        for _ in range(self.num_iterations):
            new_dataset = []
            for example in self.dataset:
                instruction, response = example["input"], example["output"]

                # Stage 1: Evolve instruction
                evolved_instruction = self.instruction_evolver(instruction)
                evolved_instruction = clean_instruction(evolved_instruction)

                # Stage 2: Generate response from evolved instruction
                evolved_response = self.evol_model.generate(evolved_instruction, **self.evol_model_kwargs)

                # Stage 3: Eliminator
                success, reason = self.instruction_eliminator(instruction, evolved_instruction, evolved_response)

                if success:
                    new_dataset.append(
                        {"evolved_instruction": evolved_instruction, "evolved_response": evolved_response,
                         "original_instruction": instruction, "original_response": response})
                else:
                    self.dataset.append(example)

                if self.verbose:
                    print(f"###Instruction\n: {instruction}")
                    print(f"##Response\n: {response}\n")
                    print(f"###Evolved Instruction\n: {evolved_instruction}")
                    print(f"##Evolved Response\n: {evolved_response}")
                    print(f"Success: {success}, Reason: {reason}")
                    print("=" * 50)

            self.dataset = new_dataset
