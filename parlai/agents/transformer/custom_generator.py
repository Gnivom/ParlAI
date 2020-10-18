from parlai.core.torch_generator_agent import TorchGeneratorAgent, TreeSearch
from parlai.core.opt import Opt

from .transformer import add_common_cmdline_args
from .modules import TransformerGeneratorModel

import parlai.utils.logging as logging

import torch
import torch.nn.functional as F

import random
import math
from fractions import Fraction
from typing import List

HIDDEN_MAX_MESSAGE = 256
HIDDEN_STOP_SIGNAL = 0


class HiddenMessage:
    def __init__(self, message: int):
        self.message = message


def empty_messages():
    return [HiddenMessage(HIDDEN_STOP_SIGNAL)]


def bytes_to_messages(bts: bytes):
    ret = []
    for message in list(bts):
        ret.append(HiddenMessage(1 + message))
    ret.append(HiddenMessage(HIDDEN_STOP_SIGNAL))
    return ret


def messages_to_bytes(messages: List[HiddenMessage]):
    bts = []
    for i, message in enumerate(messages):
        if message.message == HIDDEN_STOP_SIGNAL:
            return bytes(bts), messages[i + 1 :]
        bts.append(message.message - 1)
    return None, messages


def messages_to_fraction(messages):
    message = 0
    denominator = 1
    for m in messages:
        message *= HIDDEN_MAX_MESSAGE + 1
        message += m.message
        denominator *= HIDDEN_MAX_MESSAGE + 1
    return message, denominator


def fraction_to_messages(message, denominator):
    messages = []
    while denominator > 1:
        assert denominator % (HIDDEN_MAX_MESSAGE + 1) == 0
        messages.append(HiddenMessage(message % (HIDDEN_MAX_MESSAGE + 1)))
        message = message // (HIDDEN_MAX_MESSAGE + 1)
        denominator = denominator // (HIDDEN_MAX_MESSAGE + 1)
    messages.reverse()
    return messages


assert messages_to_bytes(bytes_to_messages(b'Hello, World!')) == (b'Hello, World!', [])


# Inspired by Nucleus Search
class CustomSearch(TreeSearch):
    def __init__(self, topp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = topp
        self.mode = None
        self.low = Fraction(0)
        self.high = Fraction(1)

    def setSendMessage(self, message: int, max_message: int):
        self.mode = "send"
        self.message = message
        self.max_message = max_message
        #        self.rand_message = random.uniform(message/(max_message+1), (message+1)/(max_message+1))
        self.rand_message = (Fraction(1, 2) + message) / (max_message + 1)
        return self

    # expected_message is a list of dictionary indices
    def setReceiveMessage(self, expected_message):
        self.mode = "receive"
        self.expected_message = expected_message
        self.received_tokens = 0
        return self

    def hasCompletedMessage(self, denominator):
        low, high = (
            math.floor(self.low * denominator),
            math.floor(self.high * denominator),
        )
        return low == high

    def getCompletedMessage(self, denominator):
        assert self.hasCompletedMessage(denominator)
        return math.floor(self.low * denominator)

    def getMessageRemainder(self):
        return (self.low, self.high)

    def continueFromRemainder(self, remainder):
        self.low, self.high = remainder
        return self

    @staticmethod
    def get_overlap(range1, range2):
        overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])
        return max(0.0, overlap)

    def _select_message_send(self, sprobs):
        remaining_size = self.high - self.low
        message = (self.rand_message - self.low) / remaining_size

        cumulative = 0.0
        for i, prob in enumerate(sprobs[0]):
            prob = float(prob) / self.p
            cumulative += prob
            if cumulative > message:
                self.low, self.high = (
                    self.low + Fraction(cumulative - prob) * remaining_size,
                    self.low + Fraction(cumulative) * remaining_size,
                )
                return [i for _ in sprobs]
        assert False, "There should have been overlap"

    def _select_message_receive(self, sprobs, sinds):
        remaining_size = self.high - self.low

        expected_token = self.expected_message[self.received_tokens]
        self.received_tokens += 1
        cumulative = 0.0
        for i, prob in enumerate(sprobs[0]):
            prob = float(prob) / self.p
            cumulative += prob
            if sinds[0, i] == expected_token:
                self.low, self.high = (
                    self.low + Fraction(cumulative - prob) * remaining_size,
                    self.low + Fraction(cumulative) * remaining_size,
                )
                return [i for _ in sprobs]
        assert False, "We should have found expected_token"

    def select_paths(self, logprobs, prior_scores, current_length):
        # Unlike the other treesearch methods, we have to switch to linspace
        # for the probabilities in order to compute the CDF.
        probs = torch.softmax(logprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        # TODO: pull request the subtraction?
        mask = (sprobs.cumsum(dim=-1) - sprobs) >= self.p
        sprobs[mask] = 0
        sprobs.div_(sprobs.sum(dim=-1).unsqueeze(1))
        if self.mode == "send":
            choices = self._select_message_send(sprobs)
        elif self.mode == "receive":
            choices = self._select_message_receive(sprobs, sinds)
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        # Convert back to logspace.
        scores = sprobs[hyp_ids, choices].log()
        best_scores = prior_scores.expand_as(scores) + scores
        return (hyp_ids, tok_ids, best_scores)


class CustomGeneratorAgent(TorchGeneratorAgent):
    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.pending_messages = None
        self.remainder = None

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(CustomGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def receiveMessage(self, messageObservation):
        self.model.eval() # Disable (non-deterministic) Dropout, etc.
        prevBatch = self.batchify([self.observation])
        beam_size = self.beam_size
        maxlen = self.label_truncate or 256

        vectorizedObs = self.dict.txt2vec(messageObservation['text'])
        vectorizedObs.append(self.END_IDX)

        def initBeam(beam):
            beam.setReceiveMessage(vectorizedObs)
            if self.remainder is not None:
                beam.continueFromRemainder(self.remainder)

        n_best_beam_preds_scores, beams = self._generateOptions(
            prevBatch, beam_size, maxlen, initBeam
        )

        self.remainder = None

        denominator = 1
        while True:
            denominator *= HIDDEN_MAX_MESSAGE + 1
            if beams[0].hasCompletedMessage(denominator):
                message = beams[0].getCompletedMessage(denominator)
                messages = fraction_to_messages(message, denominator)
                if messages[-1].message == HIDDEN_STOP_SIGNAL:
                    bts, remainder = messages_to_bytes(messages)
                    assert remainder == []
                    return bts
            else:
                self.remainder = beams[0].getMessageRemainder()
                return None

    def postMessage(self, message: bytes):
        assert self.pending_messages is None
        self.pending_messages = bytes_to_messages(message)

    def _get_pending_message(self, consume):
        messages = self.pending_messages or empty_messages()
        if consume:
            self.pending_messages = None

        message, denominator = messages_to_fraction(messages)
        return message, denominator - 1

    # override from TorchGeneratorAgent
    def _generate(self, batch, beam_size, max_ts):

        message, max_message = self._get_pending_message(consume=False)
        assert message <= max_message

        def initBeam(beam):
            beam.setSendMessage(message, max_message)
            if self.remainder is not None:
                beam.continueFromRemainder(self.remainder)

        n_best_beam_preds_scores, beams = self._generateOptions(
            batch, beam_size, max_ts, initBeam
        )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        if self.opt.get('verbose'):
            for i, beams in enumerate(n_best_beam_preds_scores):
                for b, (tokens, score) in enumerate(beams):
                    gen = self._v2t(tokens)
                    logging.debug(f"Batch[{i:3d}] Beam[{b:3d}]: ({score:4.2f}): {gen}")
                logging.debug('-')

        if beams[0].hasCompletedMessage(max_message + 1):
            self._get_pending_message(consume=True)
            self.remainder = None
        else:
            self.remainder = beams[0].getMessageRemainder()

        return beam_preds_scores, beams

    # override
    def _treesearch_factory(self, device):
        return CustomSearch(
            self.opt['topp'],
            beam_size=self.beam_size,
            min_length=0,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )

    # Implementation taken from TorchGeneratorAgent._generate
    def _generateOptions(self, batch, beam_size, max_ts, beam_init):
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths)
            if batch.text_lengths is not None
            else len(batch.image)
        )
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
            beams = [
                self._treesearch_factory(dev)
                .set_context(self._get_context(batch, batch_idx))
                .set_blacklist(self.beam_blacklist)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        for beam in beams:
            beam_init(beam)

        # repeat encoder outputs and decoder inputs
        decoder_input = (
            torch.LongTensor([self.START_IDX]).expand(bsz * beam_size, 1).to(dev)
        )

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            decoder_input = torch.index_select(decoder_input, 0, incr_state_inds)
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = torch.cat([decoder_input, selection], dim=-1)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(
                batch, n_best_beam_preds_scores
            )

        return n_best_beam_preds_scores, beams
