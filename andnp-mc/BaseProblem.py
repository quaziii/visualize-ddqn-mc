import sys
import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from typing import Optional
from PyExpUtils.utils.Collector import Collector
from RlGlue.environment import BaseEnvironment

from BaseAgent import BaseAgent
from agents.registry import getAgent

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d['agent']
        self.problem = d['problem']

        self.episode_cutoff = d.get('episode_cutoff', -1)
        self.total_steps = d.get('total_steps')

def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp


class BaseProblem:
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        self.exp = exp
        self.idx = idx

        self.collector = collector

        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']
        self.env_params = self.params.get('environment', {})
        self.exp_params = self.params.get('experiment', {})
        self.rep_params = self.params.get('representation', {})

        self.agent: Optional[BaseAgent] = None
        self.env: Optional[BaseEnvironment] = None
        self.gamma: Optional[float] = None

        self.seed = exp.getRun(idx)

        self.observations = (0,)
        self.actions = 0

    def getEnvironment(self):
        if self.env is None:
            raise Exception('Expected the environment object to be constructed already')

        return self.env

############################################################################
######################Need to fix this with our agent#######################
    def getAgent(self):
        if self.gamma is not None:
            self.params['gamma'] = self.gamma

        Agent = getAgent(self.exp.agent)
        self.agent = Agent(self.observations, self.actions, self.params, self.collector, self.seed)
        return self.agent
