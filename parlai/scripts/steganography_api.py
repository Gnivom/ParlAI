#!/usr/bin/env python3

"""
Allows a model to self-chat on a given task.
"""
import parlai
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task, validate
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger

import math
import random
import os

def setup_args(parser=None):
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

class State:
    def __init__(self, opt):
        self.agents = None
        self.agent_ownership = None
        self.num_sent_secrets = None
        self.opt = opt

state = None

def reset() -> None:
    global state
    state = None

def setup(seed: int) -> None:
    global state
    assert(state is None) # Must call reset() otherwise

    parser = setup_args()
    opt = {}
    opt['model_file'] = 'zoo:blender/blender_90M/model'
    opt['model'] = 'transformer/custom_generator'
    opt['override'] = {'model': 'transformer/custom_generator'} # Overrideing from model_file's default
    opt['beam-size'] = '1' # We don't use beam search, so more than 1 is useless
    opt['display_examples'] = 'False'
    opt['topp'] = 1.0 # Number of candidate words is limited to as few as possible with total probability >= p
    random.seed(seed)
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
    
    secret = state.agents[agent_id].receiveMessage({'text': text})
    for i, agent in enumerate(state.agents):
        if i == agent_id:
            agent.self_observe(validate({'text': text}))
        else:
            agent.observe(validate({'text': text}))
    if secret is None:
        return None
    else:
        return secret

seed = 12345
cleartext = b'The password is secret'

# Send
setup(seed)
my_id = create_agent(True)
other_id = create_agent(False)

secret_index = post_secret(my_id, cleartext)

stegotexts = []

while(has_pending_message(my_id)):
    stegotexts.append(send_stegotext(my_id))

reset()

# Receive
setup(seed)
other_id = create_agent(False)
my_id = create_agent(True)

responses = []

for stegotext in stegotexts:
    responses.append(receive_stegotext(other_id, stegotext))

assert(responses[-1] == cleartext)
