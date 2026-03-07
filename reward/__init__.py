"""
Reward module for chemical experiment RL training
"""

from .reward_function import (
    compute_score,
    compute_score_with_details,
    compute_format_reward,
    compute_lcs_reward,
    parse_procedure,
    ALLOWED_ACTIONS,
    CHEMICAL_SYNONYMS,
    NAME_TO_SMILES,
)

__all__ = [
    'compute_score',
    'compute_score_with_details',
    'compute_format_reward',
    'compute_lcs_reward',
    'parse_procedure',
    'ALLOWED_ACTIONS',
    'CHEMICAL_SYNONYMS',
    'NAME_TO_SMILES',
]
