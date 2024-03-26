# pip3 install -q -U datasets==2.17.0
from datasets import load_dataset
import pandas as pd

dataset1 = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")
dataset2 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
dataset4 = load_dataset("ajibawa-2023/Python-Code-23k-ShareGPT", split="train")
dataset6 = load_dataset("flytech/python-codes-25k", split="train")

dataset5 = load_dataset("TacoPrime/errored_python", split="train")
dataset3 = load_dataset("Fraser/python-state-changes", split="train")

dataset7 = load_dataset("espejelomar/code_search_net_python_10000_examples", split = 'train')
dataset8 = load_dataset("Programming-Language/codeagent-python", split = 'train')

df1 = dataset1.to_pandas()
df2 = dataset2.to_pandas()
df3 = dataset3.to_pandas()
df4 = dataset4.to_pandas()
df5 = dataset5.to_pandas()
df6 = dataset6.to_pandas()
df7 = dataset7.to_pandas()
df8 = dataset8.to_pandas()


instruction = []
input = []
output = []

tempins = dataset1['instruction'] + dataset2['instruction'] + dataset6['instruction']
tempinp = dataset1['input'] + dataset2['input'] + dataset6['input']
tempout = dataset1['output'] + dataset2['output'] + dataset6['output']


newdataset = pd.DataFrame(
{    'instruction': tempins,
    'input': tempinp,
    'output': tempout}
)



# Assuming tempins, tempinp, and tempout are lists containing data for 'instruction', 'input', and 'output' respectively

# Create a DataFrame
newdataset = pd.DataFrame({
    'instruction': tempins,
    'input': tempinp,
    'output': tempout
})

# Define the generate_prompt function
def generate_prompt(data_point):
    prefix_text = 'you are given the following instruction that describes the task.You are required to Write a python code that ' \
               'appropriately completes the request.\n'
    # Samples with additional context.
    if data_point['input'] and data_point['input'] != 'Not Applicable':
        text = f"""Instruction:\n{prefix_text} {data_point["instruction"]} with the inputs {data_point["input"]}\n\nResponse:\n{data_point["output"]}"""
    # Without additional context.
    else:
        text = f"""Instruction:\n{prefix_text} {data_point["instruction"]}\n\nResponse:\n{data_point["output"]}"""
    return text

# Apply generate_prompt function to each row in the DataFrame
promptnewdataset = [generate_prompt(data_point) for index, data_point in newdataset.iterrows()]

# Print or use the resulting list
# print(promptnewdataset)
