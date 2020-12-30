from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import time

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agent import Agent
numpy.set_printoptions(threshold=sys.maxsize)

class CosineAgent(Agent):
    def __init__(self,configs):
        super(CosineAgent, self).__init__(configs)
        self.questionSimValue = float(configs['questionSimValue'])
        self.answerSimValue = float(configs['answerSimValue'])
        self.normalizeUserInput = True
        self.error = 0.1


    def requestAnswer(self,userInput):
        candidates = self.candidates
        bestPairs = [candidates[0]]


        questionVect = [q.normalizedQuestion for q in candidates]
        questionVect.insert(0,userInput)
        qVectorizer = CountVectorizer(questionVect)
        qVectorizer.fit(questionVect)

        answerVect = [a.normalizedAnswer for a in candidates]
        answerVect.insert(0,userInput)
        aVectorizer = CountVectorizer(answerVect)
        aVectorizer.fit(answerVect)

        questionCosineSims = cosine_similarity(qVectorizer.transform(questionVect).toarray())
        answerCosineSims = cosine_similarity(aVectorizer.transform(answerVect).toarray())

        candidateCounter = 1



        for c in candidates:

            questionScore = questionCosineSims[0][candidateCounter]
            answerScore = answerCosineSims[0][candidateCounter]

            finalScore = self.getFinalScore(questionScore,answerScore)

            c.addScore(self.agentName,finalScore)
            try:
                if(c.getAnswer()[len(c.getAnswer())-1] != '?' and c != bestPairs[0]):
                    # if(c.getScoreByEvaluator(self.agentName) > bestPairs[0].getScoreByEvaluator(self.agentName)):
                    #     bestPairs = [c]
                    # elif(c.getScoreByEvaluator(self.agentName) == bestPairs[0].getScoreByEvaluator(self.agentName)):
                    #     bestPairs.append(c)
                    bestPairs.append(c)
            except IndexError:
                pass

            candidateCounter += 1

        bestPairs.sort(key=lambda x: x.getScoreByEvaluator(self.agentName), reverse=True)
        return bestPairs[:self.answerAmount]


    def getFinalScore(self,questionScore,answerScore):
        return questionScore * self.questionSimValue + answerScore * self.answerSimValue
