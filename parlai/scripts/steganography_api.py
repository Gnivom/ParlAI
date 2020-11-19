#!/usr/bin/env python3

# An API exposing the information embedding of the stego_generator

import parlai
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task, validate
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger

import math
import random
import os
import csv

##############################
### Helpers!
##############################

def _setup_args(parser=None):
    parser = ParlaiParser(True, True, 'Steganography bot argument parser')
    parser.add_argument('-d', '--display-examples', type='bool', default=True)
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '--seed-messages-from-task',
        action='store_true',
        help='Automatically seed conversation with messages from task dataset.',
    )
    parser.add_argument(
        '--outfile', type=str, default=None, help='File to save self chat logs'
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
    )
    parser.set_defaults(interactive_mode=True, task='steganography_api')
#    WorldLogger.add_cmdline_args(parser)
    return parser

def _read_options_from_settings(opt, settings_file):
    assert settings_file.mode == 'r'
    settings = csv.reader(settings_file)

    for row in settings:
        name, value = row
        if name == 'seed':
            random.seed(int(value))
        elif name == 'model_file':
            opt['model_file'] = value
        elif name == 'model':
            opt['model'] = value
            opt['override']['model'] = value # Overrideing from model_file's default
        elif name == 'topp':
            opt['stego_topp'] = float(value) # Number of candidate words is limited to as few as possible with total probability >= p
        else:
            print("Unknown setting: ", name)
            assert False

class State:
    def __init__(self, opt):
        self.agents = None
        self.agent_ownership = None
        self.num_sent_secrets = None
        self.opt = opt

state = None

##############################
### API starts here!
##############################

def reset() -> None:
    global state
    state = None

def setup(settings_file) -> None:
    global state
    assert(state is None) # Must call reset() otherwise

    parser = _setup_args()
    opt = {}
    opt['override'] = {}
    opt['display_examples'] = 'False'
    opt['beam-size'] = '1' # We don't use beam search, so more than 1 is redundant
    _read_options_from_settings(opt, settings_file)

    state = State(opt)
    state.agents = []
    state.agent_ownership = []
    state.num_sent_secrets = []

# Return agent index
def create_agent(is_owned: bool) -> int:
    assert(state.agents is not None) # Must call setup() otherwise
    agent = parlai.core.agents.create_agent(state.opt, requireModelExists=True)
    agent_id = len(state.agents)
    agent.id = agent.id + '_' + str(agent_id)
    agent.observe(validate({'episode_done': False, 'id': 'context', 'text': 'Star Wars'})) # TODO
    state.agents.append(agent)
    state.agent_ownership.append(is_owned)
    state.num_sent_secrets.append(0)
    return agent_id

# Return secret index
def post_secret(agent_id: int, secret: bytes) -> None:
    assert(state.agents is not None and len(state.agents) > agent_id) # Unknown agent otherwise
    assert(state.agent_ownership[agent_id]) # Can only post secrets for owned agents
    assert(state.agents[agent_id].remainder is None) # Otherwise pending message already exists. Must call send_stegotext()

    state.agents[agent_id].postMessage(secret)
    state.num_sent_secrets[agent_id] += 1
#    return state.num_sent_secrets[agent_id]

def has_pending_message(agent_id: int) -> bool:
    assert(state.agents is not None and len(state.agents) > agent_id) # Unknown agent otherwise
    return state.agents[agent_id].pending_messages is not None or state.agents[agent_id].remainder is not None

# Return the stegotext
def send_stegotext(agent_id: int) -> str:
    assert(state.agents is not None and len(state.agents) > agent_id) # Unknown agent otherwise
    assert(state.agent_ownership[agent_id]) # Can only send stegotext for owned agents

    action = state.agents[agent_id].act()
    for i, agent in enumerate(state.agents):
        if i == agent_id:
            continue
        agent.observe(validate(action))
    return action['text']

# Returns None if only a partial secret has been retrieved
def receive_stegotext(agent_id: int, text: str) -> bytes:
    assert(state.agents is not None and len(state.agents) > agent_id) # Unknown agent otherwise
    assert(not state.agent_ownership[agent_id]) # Can only receive stegotext for non-owned agents
    
    observation = {'text': text, 'episode_done': False}
    secret = state.agents[agent_id].receiveMessage(observation)
    for i, agent in enumerate(state.agents):
        if i == agent_id:
            agent.self_observe(validate(observation))
        else:
            agent.observe(validate(observation))
    if secret is None:
        return None
    else:
        return secret

