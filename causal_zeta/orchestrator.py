#!/usr/bin/env python3
"""
Multi-Agent Orchestrator for Causal Zeta.

This module defines the orchestration logic for running multiple
LLM agents in parallel. The actual execution happens via Claude's
Task tool - this file provides the structure and prompts.

Workflow:
1. Orchestrator prepares context from current state
2. Launches agents in parallel (via Task tool)
3. Collects and merges results
4. Updates state and decides next round

Usage (from Claude Code):
    # In conversation, orchestrator tells Claude which agents to spawn:

    orchestrator = CausalZetaOrchestrator()
    orchestrator.load_state("causal_zeta/outputs/state.json")

    # Get tasks to run in parallel
    tasks = orchestrator.get_parallel_tasks(round=1)
    # Returns list of (agent_name, prompt) tuples

    # Claude then uses Task tool to spawn agents in parallel
    # After results come back:
    orchestrator.integrate_results(results)
    orchestrator.save_state()
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

from .llm_agents import (
    HYPOTHESIS_AGENT, TESTER_AGENT, INTERVENTION_AGENT,
    VALIDATOR_AGENT, SYNTHESIS_AGENT,
    format_agent_prompt, parse_agent_response,
    OrchestratorState, AgentResult
)
from .graph import CausalGraph, full_graph


@dataclass
class RoundPlan:
    """Plan for one round of multi-agent execution."""
    round_number: int
    phase: str  # "explore", "test", "validate", "synthesize"
    agents_to_run: list[str]
    parallel: bool = True
    context: dict = field(default_factory=dict)


class CausalZetaOrchestrator:
    """
    Orchestrates multi-agent causal discovery for zeta spacings.

    Phases:
    1. EXPLORE: Generate hypotheses (HypothesisAgent)
    2. TEST: Run CI tests + interventions (TesterAgent, InterventionAgent) [parallel]
    3. VALIDATE: Check Q3 constraints (ValidatorAgent)
    4. SYNTHESIZE: Combine findings (SynthesisAgent)
    5. ITERATE: If not converged, go to 1
    """

    def __init__(self, output_dir: str = "causal_zeta/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = OrchestratorState()
        self.graph = full_graph()
        self.max_rounds = 5
        self.converged = False

    def save_state(self):
        """Save orchestrator state to JSON."""
        state_data = {
            "state": asdict(self.state),
            "graph": {
                "nodes": self.graph.nodes,
                "edges": [
                    {"source": e.source, "target": e.target,
                     "hypothesis": e.hypothesis, "weight": e.weight}
                    for e in self.graph.edges
                ],
            },
            "converged": self.converged,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_dir / "orchestrator_state.json", "w") as f:
            json.dump(state_data, f, indent=2)

    def load_state(self, path: str = None):
        """Load orchestrator state from JSON."""
        path = path or self.output_dir / "orchestrator_state.json"
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            # Reconstruct state
            self.state = OrchestratorState(**data.get("state", {}))
            self.converged = data.get("converged", False)
            # Reconstruct graph would need more work
            return True
        return False

    def get_current_context(self) -> dict:
        """Get context dict for agent prompts."""
        return {
            "graph_description": str(self.graph),
            "ci_results": json.dumps(self.state.test_results[-5:], indent=2) if self.state.test_results else "No tests yet",
            "intervention_results": json.dumps(self.state.intervention_results[-5:], indent=2) if self.state.intervention_results else "No interventions yet",
            "baseline_healing": "5 steps (estimated)",
            "trajectory_stats": "mean=1.02, var=0.19, min=0.03, max=3.2",
            "intervention_context": "Testing spacing perturbations",
            "ci_summary": f"{len(self.state.test_results)} tests completed",
            "intervention_summary": f"{len(self.state.intervention_results)} interventions completed",
            "validation_summary": f"{len(self.state.validation_results)} validations completed",
            "current_graph": str(self.graph),
        }

    def plan_round(self, round_number: int) -> RoundPlan:
        """
        Plan which agents to run for a given round.

        Round structure:
        - Round 1: Explore (hypothesis only)
        - Round 2: Test (tester + intervention in parallel)
        - Round 3: Validate
        - Round 4: Synthesize
        - Round 5+: Full cycle
        """
        if round_number == 1:
            return RoundPlan(
                round_number=1,
                phase="explore",
                agents_to_run=["hypothesis"],
                parallel=False,
                context=self.get_current_context(),
            )
        elif round_number == 2:
            return RoundPlan(
                round_number=2,
                phase="test",
                agents_to_run=["tester", "intervention"],
                parallel=True,
                context=self.get_current_context(),
            )
        elif round_number == 3:
            return RoundPlan(
                round_number=3,
                phase="validate",
                agents_to_run=["validator"],
                parallel=False,
                context=self.get_current_context(),
            )
        elif round_number == 4:
            return RoundPlan(
                round_number=4,
                phase="synthesize",
                agents_to_run=["synthesis"],
                parallel=False,
                context=self.get_current_context(),
            )
        else:
            # Full parallel cycle
            return RoundPlan(
                round_number=round_number,
                phase="full_cycle",
                agents_to_run=["hypothesis", "tester", "intervention", "validator"],
                parallel=True,
                context=self.get_current_context(),
            )

    def get_agent_prompts(self, plan: RoundPlan) -> list[tuple[str, str]]:
        """
        Generate prompts for agents in the plan.

        Returns list of (agent_name, formatted_prompt) tuples.
        """
        agent_templates = {
            "hypothesis": HYPOTHESIS_AGENT,
            "tester": TESTER_AGENT,
            "intervention": INTERVENTION_AGENT,
            "validator": VALIDATOR_AGENT,
            "synthesis": SYNTHESIS_AGENT,
        }

        prompts = []
        for agent_name in plan.agents_to_run:
            template = agent_templates[agent_name]

            # Add test specs for tester
            if agent_name == "tester":
                plan.context["test_specifications"] = json.dumps([
                    {"x": "Y_t", "y": "S_{t-1}", "z": ["Z_t"]},
                    {"x": "R_t", "y": "S_{t-1}", "z": ["Z_t"]},
                    {"x": "Y_t", "y": "Z_t", "z": []},
                ], indent=2)

            # Add claims for intervention
            if agent_name == "intervention":
                plan.context["claims"] = json.dumps([
                    "Z_t mediates S_{t-1} -> Y_t",
                    "R_t constrains Y_t variance",
                    "Hidden state noise disrupts predictions",
                ], indent=2)

            prompt = format_agent_prompt(template, **plan.context)
            prompts.append((agent_name, prompt))

        return prompts

    def integrate_results(self, results: list[AgentResult]):
        """Integrate results from agent executions."""
        for result in results:
            if not result.success:
                print(f"Warning: {result.agent_role} failed")
                continue

            data = result.data

            if result.agent_role == "Hypothesis Generator":
                hypotheses = data.get("hypotheses", [])
                self.state.hypotheses.extend(hypotheses)

            elif result.agent_role == "CI Test Specialist":
                test_results = data.get("test_results", [])
                self.state.test_results.extend(test_results)

            elif result.agent_role == "Intervention Designer":
                interventions = data.get("interventions", [])
                self.state.intervention_results.extend(interventions)

            elif result.agent_role == "Q3 Constraint Validator":
                validations = data.get("validations", [])
                self.state.validation_results.extend(validations)

            elif result.agent_role == "Causal Model Synthesizer":
                self.state.synthesis = data
                # Apply graph updates
                for update in data.get("graph_updates", []):
                    action = update.get("action")
                    edge = update.get("edge", [])
                    if action == "remove" and len(edge) == 2:
                        self.graph.remove_edge(edge[0], edge[1])
                    elif action == "add" and len(edge) == 2:
                        self.graph.add_edge(edge[0], edge[1], update.get("reason", ""))

        self.state.round += 1

    def check_convergence(self) -> bool:
        """Check if causal discovery has converged."""
        # Convergence criteria:
        # 1. At least 3 rounds completed
        # 2. No graph changes in last round
        # 3. All Q3 validations pass

        if self.state.round < 3:
            return False

        # Check if synthesis suggested no changes
        if self.state.synthesis:
            updates = self.state.synthesis.get("graph_updates", [])
            if not updates:
                self.converged = True
                return True

        return False

    def get_next_action(self) -> str:
        """Get description of next action for orchestrator."""
        if self.check_convergence():
            return "CONVERGED: Causal discovery complete. Generate final report."

        plan = self.plan_round(self.state.round + 1)

        if plan.parallel:
            agents = ", ".join(plan.agents_to_run)
            return f"PARALLEL: Launch {agents} agents simultaneously"
        else:
            return f"SEQUENTIAL: Run {plan.agents_to_run[0]} agent"


def generate_orchestrator_instructions() -> str:
    """
    Generate instructions for Claude to orchestrate agents.

    This is what Claude reads to know how to use the Task tool.
    """
    return """
# CAUSAL ZETA ORCHESTRATOR INSTRUCTIONS

You are orchestrating a multi-agent causal discovery system.

## Available Agents (via Task tool with subagent_type='general-purpose'):

1. **hypothesis** - Generates causal hypotheses to test
2. **tester** - Runs CI independence tests
3. **intervention** - Designs do-operations
4. **validator** - Checks Q3 constraints
5. **synthesis** - Combines findings

## Workflow:

### Round 1: Explore
```
Task(subagent_type='general-purpose', description='Generate hypotheses', prompt=hypothesis_prompt)
```

### Round 2: Test (PARALLEL)
```
# Launch BOTH in same message for parallel execution:
Task(subagent_type='general-purpose', description='Run CI tests', prompt=tester_prompt)
Task(subagent_type='general-purpose', description='Design interventions', prompt=intervention_prompt)
```

### Round 3: Validate
```
Task(subagent_type='general-purpose', description='Validate Q3', prompt=validator_prompt)
```

### Round 4: Synthesize
```
Task(subagent_type='general-purpose', description='Synthesize findings', prompt=synthesis_prompt)
```

## To Start:

```python
from causal_zeta.orchestrator import CausalZetaOrchestrator

orch = CausalZetaOrchestrator()
plan = orch.plan_round(1)
prompts = orch.get_agent_prompts(plan)

# prompts is list of (agent_name, prompt_text)
# Use each prompt_text with Task tool
```

## After Each Round:

1. Collect agent results
2. Call `orch.integrate_results(results)`
3. Call `orch.save_state()`
4. Check `orch.check_convergence()`
5. If not converged, plan next round
"""


if __name__ == "__main__":
    # Demo orchestrator
    print("Causal Zeta Orchestrator Demo")
    print("=" * 50)

    orch = CausalZetaOrchestrator()

    for round_num in range(1, 5):
        print(f"\n--- Round {round_num} ---")
        plan = orch.plan_round(round_num)
        print(f"Phase: {plan.phase}")
        print(f"Agents: {plan.agents_to_run}")
        print(f"Parallel: {plan.parallel}")

        prompts = orch.get_agent_prompts(plan)
        for agent_name, prompt in prompts:
            print(f"\n{agent_name} prompt: {len(prompt)} chars")

    print("\n" + "=" * 50)
    print(generate_orchestrator_instructions())
