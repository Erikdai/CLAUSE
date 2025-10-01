# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    index: int
    description: str
    norm_range: Tuple[float, float]


# ------------------------------ Normalization constants ------------------------------ #

PATH_LEN_NORM_MAX: float = 8.0
CAND_NORM_MAX: float = 10.0
DEC_COMMON_ID_THRESHOLD: int = 100
DEC_SHORT_OUTPUT_LEN: int = 5


# ------------------------------ Observation features ------------------------------ #

GB_OBS_FEATURES: List[FeatureSpec] = [
    FeatureSpec("num_nodes", 0, "Number of nodes in subgraph", (0.0, float("inf"))),
    FeatureSpec("num_edges", 1, "Number of edges in subgraph", (0.0, float("inf"))),
    FeatureSpec("est_tokens", 2, "Estimated tokens for subgraph", (0.0, float("inf"))),
]

TR_OBS_FEATURES: List[FeatureSpec] = [
    FeatureSpec("path_len", 0, "Current traversal path length", (0.0, float("inf"))),
    FeatureSpec("candidate_count", 1, "# traversal candidates at step", (0.0, float("inf"))),
    FeatureSpec("stop_flag", 2, "Whether traversal chose stop (0/1)", (0.0, 1.0)),
]

DEC_OBS_FEATURES: List[FeatureSpec] = [
    FeatureSpec("out_len", 0, "Decoder output length so far", (0.0, float("inf"))),
    FeatureSpec("rule_count", 1, "# active rules in DLL", (0.0, float("inf"))),
    FeatureSpec("tau", 2, "DLL temperature", (0.0, 1.0)),
]


# ------------------------------ Action features ------------------------------ #

GB_ACT_FEATURES: List[FeatureSpec] = [
    FeatureSpec("op_index", 0, "0:add_edge, 1:delete_edge, 2:stop", (0.0, 2.0)),
    FeatureSpec("is_delete", 1, "Binary: 1 if delete_edge", (0.0, 1.0)),
    FeatureSpec("is_stop", 2, "Binary: 1 if stop", (0.0, 1.0)),
]

TR_ACT_FEATURES: List[FeatureSpec] = [
    FeatureSpec("stop_flag", 0, "Binary stop flag", (0.0, 1.0)),
    FeatureSpec("path_len_norm", 1, "path_len / PATH_LEN_NORM_MAX", (0.0, 1.0)),
    FeatureSpec("cand_norm", 2, "candidate_count / CAND_NORM_MAX", (0.0, 1.0)),
]

DEC_ACT_FEATURES: List[FeatureSpec] = [
    FeatureSpec("token_norm", 0, "token_id / (vocab_size-1)", (0.0, 1.0)),
    FeatureSpec("is_common", 1, "Binary: token_id < DEC_COMMON_ID_THRESHOLD", (0.0, 1.0)),
    FeatureSpec("is_short_output", 2, "Binary: out_len < DEC_SHORT_OUTPUT_LEN", (0.0, 1.0)),
]


# ------------------------------ Dimensions ------------------------------ #

OBS_DIM_PER_AGENT: int = len(GB_OBS_FEATURES)
ACT_DIM_PER_AGENT: int = len(GB_ACT_FEATURES)

