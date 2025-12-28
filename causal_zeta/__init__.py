"""
Causal Zeta: LCM-style causal analysis for SpacingGPT.

Multi-agent system for causal discovery on zeta zero spacings.

Modules:
- variables: Z_t (latent mode), R_t (rigidity proxy)
- graph: DAG definition and d-separation
- ci_tests: Conditional independence tests (HSIC)
- interventions: do-operations for counterfactuals
- validators: Q3 constraint checkers
- llm_agents: Agent definitions and prompts
- orchestrator: Multi-agent coordination
"""

from .variables import LatentExtractor, RigidityCalculator
from .graph import CausalGraph
from .ci_tests import HSICTest, run_ci_tests
from .interventions import Intervention, InterventionSuite
from .validators import Q3Validator
from .llm_agents import (
    HYPOTHESIS_AGENT, TESTER_AGENT, INTERVENTION_AGENT,
    VALIDATOR_AGENT, SYNTHESIS_AGENT,
    format_agent_prompt, parse_agent_response
)
from .orchestrator import CausalZetaOrchestrator

__all__ = [
    # Variables
    'LatentExtractor', 'RigidityCalculator',
    # Graph
    'CausalGraph',
    # Tests
    'HSICTest', 'run_ci_tests',
    # Interventions
    'Intervention', 'InterventionSuite',
    # Validators
    'Q3Validator',
    # Agents
    'HYPOTHESIS_AGENT', 'TESTER_AGENT', 'INTERVENTION_AGENT',
    'VALIDATOR_AGENT', 'SYNTHESIS_AGENT',
    'format_agent_prompt', 'parse_agent_response',
    # Orchestrator
    'CausalZetaOrchestrator',
]
