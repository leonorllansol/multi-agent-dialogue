from transformers import BertForNextSentencePrediction, BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agent import Agent

class ContextAgent(Agent):
    def __init__(self, configs):
        super(ContextAgent, self).__init__(configs)
        pre_trained_model = configs['pre_trained_model']
        fine_tuned_model_path = configs['fine_tuned_model_path']
        self.model = BertForNextSentencePrediction.from_pretrained(pre_trained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.load_state_dict(torch.load(fine_tuned_model_path,map_location=torch.device('cpu')))

    def requestAnswer(self, userInput):
        all_candidates = []
        context = self.context
        context.append(userInput)
        context = ' '.join(context)
        for c in self.candidates:
            answer = c.getAnswer()
            encoded = self.tokenizer.encode_plus(context, text_pair=answer, return_tensors='pt')
            seq_relationship_logits = self.model(**encoded)[0]
            probs = softmax(seq_relationship_logits, dim=1)
            p = probs.detach().numpy()[0][0]
            c.addScore(self.agentName,p )
            all_candidates.append(c)
            print(context)
            print(p, answer)
        print()
        all_candidates.sort(key=lambda x: x.getScoreByEvaluator(self.agentName), reverse=True)
        return all_candidates[:self.answerAmount]
