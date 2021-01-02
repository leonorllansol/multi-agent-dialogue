import torch
from transformers import BertForNextSentencePrediction, BertTokenizer, BertForSequenceClassification
import random, numpy as np, pandas as pd
from tqdm import tqdm
import spacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getRandomResponse(all_dialogs, gold_response):
  # Selects a random response from a set of dialogs, different from a gold response
  # ----------
  # Parameters
  # ----------
  # all_dialogs: list of dialogs, where each dialog is a list of utterances
  # gold_response: string, random response must be different
  # ----------
  # Returns
  # ----------
  # random_response, string
  dialog = random.randrange(len(all_dialogs))
  ut = random.randrange(len(all_dialogs[dialog]))
  while all_dialogs[dialog][ut] == gold_response:
   ut = random.randrange(len(all_dialogs[dialog]))
  random_response = all_dialogs[dialog][ut]
  return random_response

def getDialogs(filepath, id_col='dialog_id', utterance_col='pt',maxSize = -1):
  # Retrieve a set of dialogs from a file with at least two columns, one with the dialog_id and another with the utterance
  # ----------
  # Parameters
  # ----------
  # filepath: string, path to the file with dialogs
  # id_col: string, name of the column with dialog ids
  # utterance_col: string, name of the column with utterance
  # max_size: int, max number of tokens in retrieved dialogs
  # ----------
  # Returns
  # ----------
  # all_dialogs, list
  all_dialogs = []

  df = pd.read_csv(filepath)
  dialog_ids = df[id_col].tolist()
  pt_utterances = df[utterance_col].tolist()
  i = -1
  curr_dialog = []
  for j in range(len(dialog_ids)):
    id = dialog_ids[j]
    # first dialog
    if i == -1:
      i = id
    # different dialog
    if id != i:
      all_dialogs.append(curr_dialog)
      curr_dialog = []
      i = id
    # same dialog as previous utterance
    if id == i:
      curr_dialog.append(pt_utterances[j])

  for k in range(len(all_dialogs)):
    all_dialogs[k].reverse()

  if maxSize != -1:
      all_dialogs = list(filter(lambda x: len(" ".join(x).split()) <= maxSize, all_dialogs))
  return all_dialogs

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def averageSimilarityInCorpus():
  tokenizer = BertTokenizer.from_pretrained('DeepPavlov/bert-base-multilingual-cased-sentence')

  all_dialogs = getDialogs('movie_lines_preprocessed.csv')
  nlp = spacy.load("pt_core_news_md")
  
  sentences = [getRandomResponse(all_dialogs,'') for i in range(100)]
  
  random_sentences = [getRandomResponse(all_dialogs, '') for i in range(5000)]
  for sentence in sentences:
      label_d = {'ENTAILMENT':0,'NEUTRAL':0,'PARAPHRASE':0,'CONTRADICTION':0}
      label_similarity = {'ENTAILMENT':0,'NEUTRAL':0,'PARAPHRASE':0,'CONTRADICTION':0}
      for i in tqdm(range(len(random_sentences))):
        other_sentence = random_sentences[i]
        label = predict_pf_cd(tokenizer, sentence, other_sentence)
        label_d[label] += 1
        
        if label == 'PARAPHRASE':
            print(sentence)
            print(other_sentence)
            print()
        sentence_spacy = nlp(sentence)
        other_sentence_spacy = nlp(other_sentence)
        label_similarity[label] += sentence_spacy.similarity(other_sentence_spacy)
      print(label_d)
      for k,v in label_similarity.items():
          label_similarity[k] /= label_d[label]
      print(label_similarity)

def similarityRelations(filepath="sick_br_assin_preprocessed.csv"):
  # Compute average semantic similarity by NLI label
  # ----------
  # Parameters
  # ----------
  # filepath: string, path to the file with annotated sentence-pairs 
  # ----------
  # Returns
  # ----------
  # label_scores: dict, {label:score}, with average similarity score per label
  df = pd.read_csv(filepath)
  sentences_a = df['sentence_A'].tolist()
  sentences_b = df['sentence_B'].tolist()
  labels = df['entailment_label'].tolist()
  label_scores = {'ENTAILMENT':0,'NEUTRAL':0,'PARAPHRASE':0,'CONTRADICTION':0}
  label_occur = {'ENTAILMENT':0,'NEUTRAL':0,'PARAPHRASE':0,'CONTRADICTION':0}
  nlp = spacy.load("pt_core_news_md")
  for i in range(len(sentences_a)):
    a_sp = nlp(sentences_a[i])
    b_sp = nlp(sentences_b[i])
    label_scores[labels[i]] += a_sp.similarity(b_sp)
    label_occur[labels[i]] += 1

  for k,v in label_scores.items():
    label_scores[k] /= label_occur[k]
  return label_scores

def getSimilarity(sentence1, sentence2,nlp):
  # Compute spacy similarity between two sentences
  # ----------
  # Parameters
  # ----------
  # sentence1: string 
  # sentence2: string
  # nlp: spacy model
  # ----------
  # Returns
  # ----------
  # score: float, similarity between sentence1 and sentence2 using nlp
  sent_sp_1 = nlp(sentence1)
  sent_sp_2 = nlp(sentence2)
  score = sent_sp_1.similarity(sent_sp_2)
  return score
    
