import multiprocessing, logging, sys, time, subprocess, os, operator, copy
import numpy as np
from multiprocessing import Process
import xml.etree.ElementTree as ET


import classificationAndMatching.classification as classification
import classificationAndMatching.query_agent_label_match as query_agent_label_match
import classificationAndMatching.query_answer_label_match as query_answer_label_match
from agents import *

import configsparser
#from DefaultAnswers import HighestWeightedScoreSelection, MultipleAnswerSelection
from Decisor import Decisor
from AgentHandler import AgentHandler
from learning.WeightedMajority import WeightedMajority
import DocumentManager
import agents.internalAgents
import preprocessing.excelToSubtle, preprocessing.labelsIntoXML


curr_dir = os.path.dirname(os.path.abspath(__file__))


def dialogue(agents_dict, system_dict, corpora_dict):

    defaultAgentsMode = configsparser.getDefaultAgentsMode()

    if(not configsparser.usePreviouslyCreatedIndex()):
        DocumentManager.createIndex(configsparser.getIndexPath(), configsparser.getCorpusPath())

    if(defaultAgentsMode == 'multi'):
        multiAgentAnswerMode(agents_dict, system_dict, corpora_dict)
    # elif(defaultAgentsMode == 'sequential'):
    #     sequentialConversationMode()
    elif(defaultAgentsMode == 'learning'):
        learningMode()
    elif(defaultAgentsMode == 'evaluation'):
        evaluationMode()
    else:
        classicDialogueMode()

def learningMode():
    wm = WeightedMajority()
    wm.learnWeights()

def evaluationMode():
    agentWeights = configsparser.getWeightResults()
    if(agentWeights == {}):
        print("Please set the weight dictionary in the configuration file according to the results obtained in the learning process (e.g: {'CosineAgent1': 24, 'CosineAgent2': 67, 'CosineAgent3': 9}).")
        return
    else:
        wm = WeightedMajority()
        wm.evaluation(agentWeights)

# def classicDialogueMode():
#
#     """
#     Classic mode for SSS: only calls the Evaluators inside SSS's source, doesn't take external agents into account
#
#     - Initializes an AnswerSelection object (in this case, HighestWeightedScoreSelection)
#     - Performs the "legacy" SSS routine: upon receiving a user query, it sends that query to the AnswerSelection object
#     - The AnswerSelection object then returns an answer String, which is subsequently printed to the user
#     """
#
#     highestWeightedScoreSelection = HighestWeightedScoreSelection()
#
#
#     while True:
#         query = ""
#
#         while (query == ""):
#             query = input("Say something:\n")
#
#         if query == "exit":
#             break;
#
#         logging.basicConfig(filename='log.txt', filemode='w', format='%(message)s', level=logging.INFO)
#         logging.info("Query: " + query)
#
#
#         answer = highestWeightedScoreSelection.provideAnswer(query)
#
#
#         print("Question:", query)
#         print("Answer:", answer)

# def sequentialConversationMode():
#
#     multipleAnswerSelection = MultipleAnswerSelection()
#     agentManager = AgentManager()
#     decisionMaker = DecisionMaker(configParser.getDecisionMethod())
#
#     questionsPath = configParser.getSequentialQuestionTxtPath()
#     targetPath = configParser.getSequentialTargetTxtPath()
#
#     f = open(questionsPath,'r',encoding='utf8')
#     questions = f.readlines()
#     f.close()
#
#     edgar_results = open(targetPath,'w',encoding='utf8')
#
#     for query in questions:
#
#         #defaultAgentsAnswers = multipleAnswerSelection.provideAnswer(query)
#         defaultAgentsAnswers = {}
#         externalAgentsAnswers = agentManager.generateAgentsAnswers(query)
#         # Both defaultAgentsAnswers and externalAgentsAnswers are dictionaries in the format {'agent1': 'answer1', 'agent2': 'answer2'}
#
#
#         # Calling the DecisionMaker after having all of the answers stored in the above dictionaries
#         answer = decisionMaker.decideBestAnswer(defaultAgentsAnswers,externalAgentsAnswers)
#
#
#         print("Question:", query)
#         print("Final Answer:", answer)
#         print()
#
#         edgar_results.write("Q - " + query)
#         edgar_results.write("A - " + answer + '\n\n')
#
#
#     edgar_results.close()




def multiAgentAnswerMode(agents_dict, system_dict, corpora_dict):
    global agentManager, decisionMethodsWeights, decisionMaker

    """
    :param agents_dict: A dictionary containing all of the agents' informations contained in the agents config xml file
    :param system_dict: A dictionary containing all of the system information contained in the system config xml file
    :return: Not applicable.
    """
    AMA_labels = open("corpora/AMAlabels.txt").readlines()
    for i in range(len(AMA_labels)):
        AMA_labels[i] = AMA_labels[i].strip('\n')

    covid_labels = open("corpora/covidLabels.txt").readlines()
    for i in range(len(covid_labels)):
        covid_labels[i] = covid_labels[i].strip('\n')

    agentManager = AgentHandler(AMA_labels, covid_labels)
    decisionMethodsWeights = configsparser.getDecisionMethodsWeights()

    decisionMaker = Decisor(decisionMethodsWeights, system_dict, agents_dict, corpora_dict)


    while True:
        query = ""

        while (query == ""):
            query = input("Diga algo: \n")
            print()

        if query == "exit":
            break

        logging.basicConfig(filename='logs/log.txt', filemode='w', format='%(message)s', level=logging.INFO)

        getAnswer(query, False)



def get_agent_answer(agent: str, method: str, path: str, import_path: str, query: str, labels: list, answer_list: dict)\
        -> dict:

    """
    This function receives an agent and all its information and instantiates the agent. Upon instantiating the agent, it
    calls its dialog method with the user's query and obtains an answer. In the end it returns the updated answer_list
    dictionary which now contains the new agent's answer.

    :param agent: Name of the agent of which we wish to obtain an answer from.
    :param method: Name of the method of the agent that returns the answer to a query.
    :param path: Path of the agent.
    :param import_path: Import path of the agent (path in the format of python import).
    :param query: Query that the user introduced.
    :param labels: Labels on which the agent is an expert on.
    :param answer_list: Dictionary to which we will add the new agent's answer.
    :return: The updated answer_list dictionary with the new agent's answer
    """

    # As each agent might use code that is in other folder within its directory, we must change the current working
    # directory to the agent's directory.

    os.chdir(path)

    # Create an instance of the agent with its import path, that should be "agents.agentfoldername.agentpythonfilename"
    #print("Import path")
    #print(import_path)
    module = eval(import_path)

    # Create through reflection the method that we will call to get the agent's answers

    method_to_call = getattr(module, method)
    answer = method_to_call(query)

    if len(answer) > 0 and answer[-1] == '\n':
        answer = answer[:-2]

    answer_list[agent] = answer

    return answer_list

def train(corpora_dicti):
    classification.train(corpora_dicti, 'query')
    classification.train(corpora_dicti, 'answer')

context = []

def getAnswer(query:str, firstCall: bool):
    global agentManager, decisionMethodsWeights, decisionMaker, context
    agents_dict = configsparser.getAgentsProperties(curr_dir + '/config/agents_config.xml')
    corpora_dict = configsparser.getCorporaProperties(curr_dir + '/config/corpora_config.xml')
    system_dict = configsparser.getSystemProperties(curr_dir + '/config/config.xml')

    AMA_labels = open("corpora/AMAlabels.txt").readlines()
    for i in range(len(AMA_labels)):
        AMA_labels[i] = AMA_labels[i].strip('\n')

    if firstCall:
        train(corpora_dict)
        covid_labels = open("corpora/covidLabels.txt").readlines()
        for i in range(len(covid_labels)):
            covid_labels[i] = covid_labels[i].strip('\n')

        agentManager = AgentHandler(AMA_labels, covid_labels)
        decisionMethodsWeights = configsparser.getDecisionMethodsWeights()

        decisionMaker = Decisor(decisionMethodsWeights, system_dict, agents_dict, corpora_dict)

    predictions_query = classification.predict(corpora_dict['query'], query, 'query')
    print(predictions_query)

    externalAgentsAnswers, candidateQueryDict = agentManager.generateExternalAgentsAnswers(query, predictions_query, context=context)
    # Mariana's agents
    internalAgentsAnswers = agentManager.generateInternalAgentsAnswers(get_agent_answer, agents_dict, query)





    previousWeight = -1
    currStrategy = ""
    previousDict = copy.deepcopy(decisionMethodsWeights)
    # increase weights of decision making strategies according to query labels

    #def increaseWeightsIfLabel(label, list_of_labels, strategy, increase):
        #if label in list_of_labels:
            #if decisionMethodsWeights[strategy] > 0:
                #previousWeight = decisionMethodsWeights[strategy]
                #decisionMethodsWeights[strategy] += increase
                #currStrategy = strategy
    #for label in predictions_query:
        #increaseWeightsIfLabel(label, AMA_labels, "AMAStrategy", 60)
    #increaseWeightsIfLabel("OR_QUESTION", predictions_query, "ORStrategy", 60)
    #increaseWeightsIfLabel("YN_QUESTION", predictions_query, "YNStrategy", 60)

    for label in AMA_labels:
        if label in predictions_query:
            if decisionMethodsWeights["AMAStrategy"] > 0:
                previousWeight = decisionMethodsWeights["AMAStrategy"]
                decisionMethodsWeights["AMAStrategy"] += 60
                currStrategy = "AMAStrategy"
    if currStrategy=="" and "OR_QUESTION" in predictions_query:
        if decisionMethodsWeights["OrStrategy"] > 0:
            previousWeight = decisionMethodsWeights["OrStrategy"]
            decisionMethodsWeights["OrStrategy"] += 60
            currStrategy = "OrStrategy"
    elif currStrategy=="" and "YN_QUESTION" in predictions_query:
        if decisionMethodsWeights["YesNoStrategy"] > 0:
            previousWeight = decisionMethodsWeights["YesNoStrategy"]
            decisionMethodsWeights["YesNoStrategy"] += 60
            currStrategy = "YesNoStrategy"

    # update weights of other strategies so that sum of dictionary is still 100
    if previousWeight != -1:
        # assuming that all weights sum to 100
        remainingWeights = 100 - decisionMethodsWeights[currStrategy]
        for strategy, w in decisionMethodsWeights.items():
            if strategy != currStrategy:
                decisionMethodsWeights[strategy] /= (100 - previousWeight) / remainingWeights
    #print(decisionMethodsWeights)

    for agent in internalAgentsAnswers:
        #if isinstance(internalAgentsAnswers[agent], list) or len(internalAgentsAnswers[agent]) == 0:
        if len(internalAgentsAnswers[agent]) == 0:
            # delete empty answers
            del internalAgentsAnswers[agent]

    answer_by_strategy, answer_tags = decisionMaker.bestAnswerByStrategy(query, internalAgentsAnswers, externalAgentsAnswers, predictions_query)

    # score per answer based on config weights per strategy {'answer':weight}
    score_per_answer = {}
    for strategy in answer_by_strategy:
        if answer_by_strategy[strategy] != '':
            if answer_by_strategy[strategy] == configsparser.getNoAnswerMessage():
                continue
            if answer_by_strategy[strategy] in score_per_answer:
                score_per_answer[answer_by_strategy[strategy]] += decisionMethodsWeights[strategy]
            else:
                score_per_answer[answer_by_strategy[strategy]] = decisionMethodsWeights[strategy]

    # if no answer different that the noAnswerMessage was found
    if len(score_per_answer) == 0:
        answer_max_weight = configsparser.getNoAnswerMessage()
    else:
        answer_max_weight = max(score_per_answer.items(), key=operator.itemgetter(1))[0]

    logging.info("Query: " + query)
    logging.info("Answer: " + answer_max_weight)

    agent_max_weight = ""
    for agent in externalAgentsAnswers:
        if externalAgentsAnswers[agent] == answer_max_weight:
            agent_max_weight = agent
            break
    if agent_max_weight in ['AntiCovidAgent','AMAAgent'] and answer_max_weight != configsparser.getNoAnswerMessage():
        answer = ""
        answer += "Percebi que a sua pergunta é: \"" + candidateQueryDict[agent_max_weight] + "\".\n"
        answer += answer_max_weight
        # if isinstance(source, float):
        #     source = "Desconhecida"
        # answer += " Fonte: " + source
        answer_max_weight = answer

    print()
    print("Questão: ", query)
    print("Resposta: ", answer_max_weight)
    print()


    context.append(query)
    context.append(answer_max_weight)
    while len((' '.join(context)).split()) > 512:
        context.pop(0)

    # repôr pesos
    decisionMethodsWeights = copy.deepcopy(previousDict)

    return answer_max_weight

def createCorpora():
    tree = ET.parse('agents/externalAgents/GeneralAgent/config.xml')
    root = tree.getroot()
    for i in range(len(root.findall('corpusPath'))):
        corpusPath = root.findall('corpusPath')[i].text
        agentName = root.findall('corpusPath')[i].get('name')
        excelPath = root.findall('excelPath')[i].text
        indexPath = root.findall('indexPath')[i].text
        labelsPath = root.findall('labelsPath')[i].text

        path = "" + corpusPath
        s = "corpora/"
        t = ".txt"
        if s in path:
            path = path.replace(s,"")
        if t in path:
            path = path.replace(t,"")
        keyword = path
        if excelPath != 'None':
            preprocessing.excelToSubtle.createQueryAnswerFromExcel(excelPath, keyword, corpusPath, labelsPath)
        preprocessing.labelsIntoXML.parseLabelsIntoXML(agentName, labelsPath)

if __name__ == "__main__":
    '''Mariana'''
    if 'GeneralAgent' in configsparser.getActiveAgents():
        createCorpora()
    agents_dicti = configsparser.getAgentsProperties(curr_dir + '/config/agents_config.xml')
    corpora_dicti = configsparser.getCorporaProperties(curr_dir + '/config/corpora_config.xml')
    system_dicti = configsparser.getSystemProperties(curr_dir + '/config/config.xml')
    train(corpora_dicti)
    dialogue(agents_dicti, system_dicti, corpora_dicti)
