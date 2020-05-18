from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.opt import Opt

from .transformer import add_common_cmdline_args
from .modules import TransformerGeneratorModel

import parlai.utils.logging as logging

import torch
import torch.nn.functional as F

import random


class CustomGeneratorAgent(TorchGeneratorAgent):
    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.pending_message = None

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
        prevBatch = self.batchify([self.observation])
        beam_size = self.beam_size
        maxlen = self.label_truncate or 256
        max_message = beam_size

        n_best_beam_preds_scores, _ = self._generateOptions(
            prevBatch, beam_size, maxlen, max_message
        )

        for i, (tokens, _) in enumerate(n_best_beam_preds_scores[0]):
            text = self._v2t(tokens)
            if messageObservation['text'] == text:
                return i
        return None

    def postMessage(self, message):
        self.pending_message = message

    def _get_pending_message(self, consume):
        """
            :return:
                tuple (message, max_message)

                - message: integer in [0, max_message)
        """
        max_message = self.beam_size
        message = self.pending_message or 0
        if consume:
            self.pending_message = None
        return message, max_message

    # override from TorchGeneratorAgent
    def _generate(self, batch, beam_size, max_ts):

        message, max_message = self._get_pending_message(consume=True)
        assert beam_size >= max_message
        assert message < max_message

        n_best_beam_preds_scores, beams = self._generateOptions(
            batch, beam_size, max_ts, max_message
        )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [
            n_best_list[message] for n_best_list in n_best_beam_preds_scores
        ]

        if self.opt.get('verbose'):
            for i, beams in enumerate(n_best_beam_preds_scores):
                for b, (tokens, score) in enumerate(beams):
                    gen = self._v2t(tokens)
                    logging.debug(f"Batch[{i:3d}] Beam[{b:3d}]: ({score:4.2f}): {gen}")
                logging.debug('-')

        return beam_preds_scores, beams

    # Implementation taken from TorchGeneratorAgent._generate
    def _generateOptions(self, batch, beam_size, max_ts, max_message):
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
        n_best_beam_preds_scores = [b.get_rescored_finished(max_message) for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(
                batch, n_best_beam_preds_scores
            )

        return n_best_beam_preds_scores, beams
