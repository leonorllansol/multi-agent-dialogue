import json
import random
from tqdm import tqdm
import copy
with open('personachat_self_original.json') as json_file:
    data = json.load(json_file)

train_data = data['train']
valid_data = data['valid']

no_context_data = []
shuffle_context_data = []
half_context_data = []

for i in tqdm(range(len(train_data))):
    no_context_entry = {'personality': train_data[i]['personality'], 'utterances':[]}
    shuffle_context_entry = {'personality': train_data[i]['personality'], 'utterances':[]}
    half_context_entry = {'personality': train_data[i]['personality'], 'utterances':[]}
    for j in range(len(train_data[i]['utterances'])):
        candidates = train_data[i]["utterances"][j]["candidates"]
        history = train_data[i]["utterances"][j]["history"]

        if len(history) > 1:
            """remove context"""
            no_history = history[-1]
            tmp = {'candidates': candidates, 'history': no_history}
            no_context_entry['utterances'].append(tmp)

            """randomly remove half of the utterances"""
            half_history = copy.deepcopy(history)
            for i in range(len(half_history)//2):
                # len - 1 so it does not pop the last utterance which is the question
                half_history.pop(random.randrange(len(half_history)-1))
            tmp = {'candidates': candidates, 'history': half_history}
            half_context_entry['utterances'].append(tmp)

            """shuffle context"""
            shuffled_history = copy.deepcopy(history[:-1])
            random.shuffle(shuffled_history)
            # append latest question
            shuffled_history.append(history[-1])
            tmp = {'candidates': candidates, 'history': shuffled_history}
            shuffle_context_entry['utterances'].append(tmp)
        # context only has question
        # else:
            # do nothing
    no_context_data.append(no_context_entry)
    half_context_data.append(half_context_entry)
    shuffle_context_data.append(shuffle_context_entry)

no_context_dct = {'train':no_context_data, 'valid': valid_data}
half_context_dct = {'train':half_context_data, 'valid': valid_data}
shuffle_context_dct = {'train':shuffle_context_data, 'valid': valid_data}

filename_dct = {'persona_no_context': no_context_dct, 'persona_half_context': half_context_dct, 'persona_shuffle_context': shuffle_context_dct}
for filename,d in filename_dct.items():
    data = json.dumps(d,ensure_ascii=False,indent=4).encode('utf8')
    with open(filename, 'w', encoding='utf8') as json_file:
        json_file.write(data.decode())
