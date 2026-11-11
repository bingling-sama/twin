"""Shared domain literal types across core, services, and API layers."""

from typing import Literal

DatasourceType = Literal["skin", "cape"]
DiscoveryMode = Literal["confirmed", "variants"]
SeedStrategy = Literal["random", "sequential"]
ClusterMatchLevel = Literal["confirmed", "suspected"]

