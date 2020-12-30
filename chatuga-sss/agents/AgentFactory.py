from xml.dom import minidom
from xml.dom.minidom import Node
import os, sys, inspect
import re
import importlib
# to import file from main directory
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(current_dir)

import configsparser

"""
The AgentFactory module is a library used by the AgentManager class in order to discover and instantiate all available external agents.

To perform that role, it searches for all config.xml files inside the externalAgents folder and uses these configuration files to import each of the agent classes into SSS.
"""


"""
createExternalAgents(externalAgentsPath): Given the path location of the externalAgents folder, generates an array of agent objects corresponding to each of the available external agents

externalAgentsPath: String containing the path of the externalAgents folder relative to pySSS. This value can be configured in the config.xml file through the parameter <externalAgentsPath>

- Create externalAgents array in order to store an instance of each external agent
- Obtain a configFiles array of strings corresponding to the paths of each external agent's config.xml
- For each config.xml:
    - Generate a configs dictionary where all of the given config.xml's parameters are stored, in the format {'parameterName': 'parameterContents'}
    - Import the agent's main class through the mainClass parameter in the config.xml
    - Initialize an instance of the agent
    - Add that agent instance to the externalAgents array

Return an externalAgents array containing each agent's instance

"""

def createExternalAgents(externalAgentsPath):
    externalAgents = []

    configFiles = getConfigFiles(externalAgentsPath)
    disabledAgents = getDisabledAgents()

    for config in configFiles:


        configs = {}
        configDoc = minidom.parse(config)

        for elem in configDoc.getElementsByTagName('config'):
            for x in elem.childNodes:
                if(x.nodeType == Node.ELEMENT_NODE):
                    if (x.hasAttribute('name')):
                        if not x.tagName in configs:
                            configs[x.tagName] = {}
                        configs[x.tagName][x.getAttribute('name')] = x.firstChild.data
                    else:
                        configs[x.tagName] = x.firstChild.data

        if(len(configs) > 0):

            mainClass = configs['mainClass']
            if(mainClass not in disabledAgents and mainClass in configsparser.getActiveAgents()):
                try:
                    agentAmount = int(configs['agentAmount'])
                except KeyError:
                    agentAmount = 1

                try:
                    variedParameters = configs['variedParameters'].split(',')
                except KeyError:
                    variedParameters = []


                for a in range(agentAmount):
                    agentConfigs = configs.copy()

                    module = importlib.import_module('.' + mainClass + '.' + mainClass,'agents.externalAgents')
                    class_ = getattr(module,agentConfigs['mainClass'])

                    for param in variedParameters:
                        paramValues = agentConfigs[param]
                        agentConfigs[param] = paramValues.split(',')[a]

                    if mainClass == 'GeneralAgent':
                        agent = class_(agentConfigs, a)
                    else:
                        agent = class_(agentConfigs)

                    if(agentAmount > 1 and mainClass != 'GeneralAgent'):
                        agent.agentName += str(a+1)

                    agent.useWhoosh = True if agentConfigs['useWhoosh'] == 'true' else False

                    externalAgents.append(agent)
    return externalAgents


"""
getConfigFiles(dirName): Auxiliary function with the role of searching for all config.xml files inside a given directory

dirName: String corresponding to the path of the directory to search

- Check all entities (directories and files) inside dirName
- Create a configFiles array in order to store the path of each config.xml file
- For each entity inside dirName:
    - Check if the entity is a directory
        - If it is, calls the function getConfigFiles on that directory (recursive searching)
    - Check if the entity is a config.xml file
        - If it is, append the path of the config.xml to the configFiles array

Return the configFiles array containing all of the config.xml file paths

"""


def getConfigFiles(dirName):
    directories = os.listdir(dirName)
    configFiles = []
    for d in directories:

        fullpath = os.path.join(dirName, d)
        if os.path.isdir(fullpath):
            configFiles = configFiles + getConfigFiles(fullpath)
        elif fullpath.endswith('config.xml'):
            configFiles.append(fullpath)
    return configFiles


def getDisabledAgents():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    return open(curr_dir + '/externalAgents/disabledAgents.txt','r').read().splitlines()
