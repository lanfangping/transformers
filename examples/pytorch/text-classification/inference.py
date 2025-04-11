from transformers import AutoTokenizer, AutoModel, utils
from torch import tensor
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings
import difflib

def find_first_element_in_list(element, array_list, begin_idx=0):
        for idx in range(begin_idx, len(array_list)):
            if array_list[idx] == element:
                return idx
        return -1

def find_diff_indexes_in_arrays(array, array_pair, item_real_begin_index=0, item_pair_real_begin_index=0):
    # convert int array to strin array
    array = [str(a) for a in array]
    array_pair = [str(a) for a in array_pair]

    # Creating a Differ object
    d = difflib.Differ()

    # Calculating the difference
    diff = list(d.compare(array, array_pair))
    # print(diff)
    # # Extracting the index of changing words
    # changes = [i for i, word in enumerate(diff) if word.startswith("+ ") or word.startswith("- ")]

    i, j = 0, 0
    old_diff = []
    new_diff = []
    for idx, word in enumerate(diff):
        if word.startswith("+"):
            new_diff.append(i)
            i += 1
        elif word.startswith("-"):
            old_diff.append(j)
            j += 1
        else:
            i += 1
            j += 1

    return old_diff, new_diff

def process_attention_weight(tokenizer, input_ids):
    attetion_masks = []
    for i, ids in enumerate(input_ids):
        sep_idx1 = find_first_element_in_list(tokenizer.sep_token_id, ids, begin_idx=0)
        array = ids[1:sep_idx1]

        sep_idx2 = find_first_element_in_list(tokenizer.sep_token_id, ids, begin_idx=sep_idx1+1)
        array_pair = ids[sep_idx1+1:sep_idx2]
            
        temp_old_diff, temp_new_diff = find_diff_indexes_in_arrays(array, array_pair)
        old_diff = [idx+1 for idx in temp_old_diff]
        new_diff = [idx+sep_idx1+1 for idx in temp_new_diff]
        changes = old_diff + new_diff

        atte_mask = []
        for idx in range(len(ids)):
            if idx in changes:
                atte_mask.append(1)
            elif ids[idx] == tokenizer.pad_token_id:
                atte_mask.append(0)
            else:
                atte_mask.append(0.5)
        attetion_masks.append(atte_mask)

    return attetion_masks

model_name = "./checkpoints/checkpoints2402191335"  # Find popular HuggingFace models here: https://huggingface.co/models
# input_text = "The cat sat on the mat"  
old_snippet = "Less likely to cause problems is the minimum size for shared memory segments (SHMMIN), which should be at most approximately 500 kB for PostgreSQL (it is usually just 1)."
new_snippet = "Less likely to cause problems is the minimum size for shared memory segments (SHMMIN), which should be at most approximately 32 bytes for PostgreSQL (it is usually just 1). The maximum number of segments system-wide (SHMMNI) or per-process (SHMSEG) are unlikely to cause a problem unless your system has them set to zero."
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name, )
inputs = tokenizer(old_snippet, new_snippet, return_tensors='pt')  # Tokenize input text
inputs = {k: v.to(model.device) for k,v in inputs.items()}
print(inputs)
print("default attention", inputs['attention_mask'])
inputs['attention_mask'] = tensor(process_attention_weight(tokenizer, inputs['input_ids']))
print("updated attention", inputs['attention_mask'])

outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view