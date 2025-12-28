"""
LLM Agents for Causal Zeta (DEMOCRITUS-style).

Multi-agent system where specialized agents work in parallel:
1. HypothesisAgent - generates causal hypotheses/questions
2. TesterAgent - runs CI tests and reports results
3. InterventionAgent - designs and runs do-operations
4. ValidatorAgent - checks Q3 constraints
5. SynthesisAgent - combines findings into coherent model

The orchestrator (main Claude) coordinates all agents.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class AgentPrompt:
    """Prompt template for an agent."""
    role: str
    system_context: str
    task_template: str
    output_format: str


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

HYPOTHESIS_AGENT = AgentPrompt(
    role="Hypothesis Generator",
    system_context="""You are a causal inference expert analyzing zeta zero spacings.

Context:
- We have a transformer model (SpacingGPT) trained on 2M unfolded zeta zeros
- Variables: S_t (spacing), Z_t (latent mode from PCA), R_t (rigidity), Y_t (target)
- Goal: discover causal structure, not just correlations

Your expertise:
- Random Matrix Theory (GUE universality)
- Causal inference (DAGs, d-separation, interventions)
- Spectral theory (SFF, rigidity, level repulsion)
""",
    task_template="""Given the current causal graph and test results:

CURRENT GRAPH:
{graph_description}

RECENT CI TEST RESULTS:
{ci_results}

RECENT INTERVENTION RESULTS:
{intervention_results}

Generate 3-5 NEW causal hypotheses to test. For each hypothesis:
1. State it as "X causes Y" or "X causes Y through Z"
2. Explain the physical/mathematical intuition
3. Suggest how to test it (CI test or intervention)
4. Rate confidence (low/medium/high)
""",
    output_format="""Return JSON:
{
  "hypotheses": [
    {
      "statement": "Z_t mediates the effect of S_{t-1} on Y_t",
      "intuition": "Phase encodes long-range correlations from GUE kernel",
      "test_type": "ci_test",
      "test_spec": {"x": "Y_t", "y": "S_{t-1}", "z": ["Z_t"]},
      "confidence": "medium"
    }
  ]
}
"""
)


TESTER_AGENT = AgentPrompt(
    role="CI Test Specialist",
    system_context="""You are a statistical testing expert specializing in causal inference.

Your tools:
- HSIC (Hilbert-Schmidt Independence Criterion) for independence tests
- Conditional HSIC for X ⊥ Y | Z tests
- Permutation tests for p-values

Data available:
- CausalState objects with (S_t, Z_t, R_t, Y_t) for 2000+ samples
- Z_t is 2D (PCA components), others are scalars
""",
    task_template="""Run the following CI tests and interpret results:

TESTS TO RUN:
{test_specifications}

For each test:
1. Run HSIC with 100 permutations
2. Report HSIC value and p-value
3. Interpret: does this support or refute the hypothesis?
4. Suggest graph modifications if needed
""",
    output_format="""Return JSON:
{
  "test_results": [
    {
      "test": "Y_t ⊥ S_{t-1} | Z_t",
      "hsic": 0.0023,
      "p_value": 0.34,
      "independent": true,
      "interpretation": "Z_t screens off local effect, supporting mediation hypothesis",
      "graph_action": "consider_removing_edge S_{t-1} -> Y_t"
    }
  ],
  "summary": "2/3 tests support the latent mediation model"
}
"""
)


INTERVENTION_AGENT = AgentPrompt(
    role="Intervention Designer",
    system_context="""You are an expert in causal interventions and counterfactual reasoning.

Available interventions:
- do(S_t := S_t + δ): Perturb spacing value
- do(Z_t := Z_t + δ_vec): Perturb latent phase via hidden state
- do(R_t := const): Clamp rigidity to fixed value
- do(H_t := H_t + noise): Add noise to hidden state

Key metrics:
- Healing time: steps until trajectory returns to baseline
- Q3 violations: constraint breaches after intervention
""",
    task_template="""Design interventions to test these causal claims:

CLAIMS TO TEST:
{claims}

CURRENT HEALING TIME BASELINE: {baseline_healing}

For each claim:
1. Design a specific intervention
2. Predict expected outcome if claim is TRUE vs FALSE
3. Specify parameters (δ, position, etc.)
4. Define success criteria
""",
    output_format="""Return JSON:
{
  "interventions": [
    {
      "claim": "Rigidity constrains output distribution",
      "intervention": "do(R_t := 0.5)",
      "if_true": "Y_t distribution narrows, variance decreases",
      "if_false": "No change in Y_t distribution",
      "params": {"target_rigidity": 0.5, "window": 10},
      "success_metric": "variance_ratio < 0.8"
    }
  ]
}
"""
)


VALIDATOR_AGENT = AgentPrompt(
    role="Q3 Constraint Validator",
    system_context="""You are an expert in Riemann Hypothesis constraints and RMT.

Q3 Constraints (from PROSHKA formulation):
- C1 Rigidity: Var(spacings) / 0.178 < 2.0
- C2 Repulsion: min(spacing) > 0.01, fraction_tiny < 1%
- C3 Mean: |mean(spacing) - 1.0| < 0.1
- C4 GUE: K-S test against Wigner surmise p > 0.01
- C5 SFF: Spectral form factor shows ramp->plateau

Key principle: Q3 constraints are VALIDATORS not TEACHERS.
We don't train to satisfy them - we check if generated data is physically plausible.
""",
    task_template="""Validate these trajectories against Q3 constraints:

TRAJECTORY STATISTICS:
{trajectory_stats}

INTERVENTION CONTEXT:
{intervention_context}

For each constraint:
1. Check pass/fail
2. Compute actual value vs threshold
3. Interpret physical meaning of violations
4. Suggest if this reveals model failure or expected behavior
""",
    output_format="""Return JSON:
{
  "validations": [
    {
      "constraint": "C1_Rigidity",
      "passed": true,
      "value": 1.23,
      "threshold": 2.0,
      "interpretation": "Variance within GUE bounds, model respects spectral rigidity"
    }
  ],
  "overall_valid": true,
  "physical_interpretation": "Generated trajectories are consistent with GUE universality"
}
"""
)


SYNTHESIS_AGENT = AgentPrompt(
    role="Causal Model Synthesizer",
    system_context="""You are a causal discovery expert who synthesizes findings into coherent models.

Your job:
1. Combine CI test results, intervention outcomes, and Q3 validations
2. Propose refined causal graph
3. Identify remaining uncertainties
4. Suggest next experiments
""",
    task_template="""Synthesize findings from this round of causal discovery:

CI TEST SUMMARY:
{ci_summary}

INTERVENTION SUMMARY:
{intervention_summary}

VALIDATION SUMMARY:
{validation_summary}

CURRENT GRAPH:
{current_graph}

Produce:
1. Updated causal graph (edges to add/remove/modify)
2. Confidence assessment for each edge
3. Key insights about zeta spacing mechanism
4. Top 3 next experiments to run
""",
    output_format="""Return JSON:
{
  "graph_updates": [
    {"action": "remove", "edge": ["S_{t-1}", "Y_t"], "reason": "screened by Z_t"},
    {"action": "strengthen", "edge": ["Z_t", "Y_t"], "reason": "intervention confirmed"}
  ],
  "edge_confidences": {
    "Z_t -> Y_t": 0.85,
    "R_t -> Y_t": 0.70
  },
  "insights": [
    "Latent phase Z_t is the primary driver of spacing dynamics",
    "Direct repulsion S_{t-1} -> Y_t is weaker than expected"
  ],
  "next_experiments": [
    "Test if Z_t[0] and Z_t[1] have different causal roles",
    "Intervention on specific attention heads",
    "Compare with shuffled baseline"
  ]
}
"""
)


# =============================================================================
# ORCHESTRATION HELPERS
# =============================================================================

def format_agent_prompt(agent: AgentPrompt, **kwargs) -> str:
    """Format agent prompt with current context."""
    filled_task = agent.task_template.format(**kwargs)

    full_prompt = f"""# Role: {agent.role}

## Context
{agent.system_context}

## Your Task
{filled_task}

## Output Format
{agent.output_format}

IMPORTANT: Return ONLY valid JSON, no markdown code blocks."""

    return full_prompt


def parse_agent_response(response: str) -> dict:
    """Parse JSON from agent response."""
    # Try to extract JSON from response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```"):
        lines = response.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not line.startswith("```"):
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except:
                pass
        return {"error": "Failed to parse response", "raw": response}


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_role: str
    success: bool
    data: dict
    raw_response: str = ""


@dataclass
class OrchestratorState:
    """State maintained by the orchestrator."""
    round: int = 0
    graph_version: str = "v0.1"
    hypotheses: list = field(default_factory=list)
    test_results: list = field(default_factory=list)
    intervention_results: list = field(default_factory=list)
    validation_results: list = field(default_factory=list)
    synthesis: dict = field(default_factory=dict)

    def to_context(self) -> dict:
        """Convert state to context dict for agents."""
        return {
            "round": self.round,
            "graph_version": self.graph_version,
            "n_hypotheses": len(self.hypotheses),
            "n_tests": len(self.test_results),
            "n_interventions": len(self.intervention_results),
        }


# =============================================================================
# AGENT TASK SPECS (for Task tool)
# =============================================================================

AGENT_SPECS = {
    "hypothesis": {
        "description": "Generate causal hypotheses",
        "prompt_template": HYPOTHESIS_AGENT,
        "subagent_type": "general-purpose",
    },
    "tester": {
        "description": "Run CI tests",
        "prompt_template": TESTER_AGENT,
        "subagent_type": "general-purpose",
    },
    "intervention": {
        "description": "Design interventions",
        "prompt_template": INTERVENTION_AGENT,
        "subagent_type": "general-purpose",
    },
    "validator": {
        "description": "Validate Q3 constraints",
        "prompt_template": VALIDATOR_AGENT,
        "subagent_type": "general-purpose",
    },
    "synthesis": {
        "description": "Synthesize findings",
        "prompt_template": SYNTHESIS_AGENT,
        "subagent_type": "general-purpose",
    },
}


if __name__ == "__main__":
    # Test prompt formatting
    print("Testing agent prompts...")

    test_context = {
        "graph_description": "S_{t-1} -> Z_t -> Y_t, R_t -> Y_t",
        "ci_results": "Y_t ⊥ S_{t-1} | Z_t: p=0.34 (independent)",
        "intervention_results": "do(S_t+0.05): healing_time=3 steps",
    }

    prompt = format_agent_prompt(HYPOTHESIS_AGENT, **test_context)
    print(f"Hypothesis agent prompt ({len(prompt)} chars):")
    print(prompt[:500] + "...")
