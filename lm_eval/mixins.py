import os
import pathlib
import re
import collections
import functools
import inspect
import sys
import signal
from typing import List, Callable, TypeVar

T = TypeVar('T')

import sympy
from sympy.core.sympify import SympifyError
from sympy.parsing.latex import parse_latex

from lm_eval.utils import timeout
from lm_eval.base import rf

class MajorityVotingMixin:
    """
    Majority voting for an arbitrary definition of equivalence.

    Also enables support for temperature and top-p sampling. 

    The `majority_vote` function should likely be called by the subclass in `Task.process_results()`.
    The `construct_requests` method works with no code changes to the subclass, 
    but requires passing the `--description_dict_path` cli argument
    """
    MAJORITY_VOTING = "majority_voting"
    SAMPLING_TEMPERATURE = "sampling_temperature"
    TOP_P = "top_p"
    EVAL_BATCH_SIZE = "eval_batch_size"
    def majority_vote(
            self,
            sampled_answers: List[T],
            correct_answer: T,
            is_equiv : Callable[[T, T], bool] = lambda x, y: x==y,
            invalid_answer: T = None
    ):
        """
        Performs majority voting on a list of candidate answers. 
        Returns accuracy and pass rate checked against `correct_answer`.
        Supports arbitrary definitions of equivalence via `is_equiv` argument.
        
        Arguments:
            sampled_answers: List[T], list of sampled answers
            correct_answer: T, ground truth.
            is_equiv: Callable[[T, T], bool], a function that determines when two answers 
                should be treated as equivalent. Default is T-equivalence, i.e `lambda x y: x==y`.
            invalid_answer: T, answer that corresponds to a parsing failure from a sample. 
                If passed as arg, no votes for invalid answer should be counted, but it should
                count against pass_rate.
        Returns:
            acc: int, 0/1 for correct/incorrect
            pass_rate: float, proportion of `sampled_answers` equivalent to `correct_answer`
            votes: List[Tuple[T, int]], for each distinct answer, the amount of votes for that answer. 
                Sorted by descending amount of votes, so that `elected_answer==votes[0][0]`
        """
        if not sampled_answers:
            return 0, 0, []

        answer_votes = {}

        # we only count votes for successfully parsed answers, as we choose not
        # to allow a model to vote for [invalidanswer] as its response.
        # however, we do want to calculate pass_rate as a function of 
        # total K = *num. sampled answers*.
        if invalid_answer:
            valid_sampled_answers = [answer for answer in sampled_answers if answer != invalid_answer]
        else:
            valid_sampled_answers = sampled_answers

        for answer in valid_sampled_answers:
            if answer in answer_votes: 
                answer_votes[answer] += 1
            else:
                counted = False
                for ref in answer_votes:
                    if is_equiv(answer, ref) and not counted:
                        answer_votes[ref] += 1
                        counted=True
                if not counted: 
                    answer_votes[answer] = 1

        votes = list(sorted(answer_votes.items(), key=lambda x: -x[1]))

        elected_answer = votes[0][0]

        if is_equiv(correct_answer, elected_answer):
            acc = 1
            pass_rate = votes[0][1] / len(sampled_answers)
        else:
            acc = 0
            pass_rate = 0
            for candidate, num_votes in answer_votes.items():
                if is_equiv(correct_answer, candidate):
                    pass_rate = num_votes / len(sampled_answers)
                    break

        return acc, pass_rate, votes

    def construct_requests(self, doc, ctx, params={}):
        if params == {}:
            if isinstance(self.end_seq, str):
                return rf.generate(ctx, [self.end_seq])
            else:
                return rf.generate(ctx, self.end_seq)
        
        majority_voting_value = int(params.get(self.MAJORITY_VOTING, 1))
        sampling_temperature_value = float(params.get(self.SAMPLING_TEMPERATURE, 1.0))
        top_p = float(params.get(self.TOP_P, 1.0))
        eval_batch_size = params.get(self.EVAL_BATCH_SIZE, None)
        eval_batch_size = int(eval_batch_size) if isinstance(eval_batch_size, str) else eval_batch_size
        generation_params = {
            'num_return_sequences': majority_voting_value,
            'temperature': sampling_temperature_value,
            'top_p': top_p,
            'num_return_sequences_batch': eval_batch_size
        }
        if isinstance(self.end_seq, str):
            return rf.generate(ctx, [self.end_seq], generation_params)
        else:
            return rf.generate(ctx, self.end_seq, generation_params)
