"""
Causal Graph for Zeta Spacing Model.

DAG structure based on DEMOCRITUS-style causal decomposition:
- Z_t (latent mode) as driver
- R_t (rigidity) as constraint
- S_{t-1} as local input
- Y_t as observable output

Key insight: H_t (model hidden state) is NOT a node.
It's the "observer", not the "system". Including it would be data leakage.
"""

from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class CausalEdge:
    """An edge in the causal graph."""
    source: str
    target: str
    hypothesis: str  # Human-readable hypothesis
    weight: float = 1.0  # Strength (updated by CI tests)
    validated: bool = False


@dataclass
class CausalGraph:
    """
    Causal DAG for zeta spacing dynamics.

    Nodes:
    - S_{t-1}: Previous spacing (observable)
    - Z_t: Latent mode (PCA of hidden, dim=2)
    - R_t: Rigidity proxy (local variance)
    - Y_t: Target spacing (next token)

    Default edges (v0.1):
    - S_{t-1} -> Z_t: Local spacing affects phase
    - S_{t-1} -> Y_t: Direct repulsion (short-range)
    - Z_t -> R_t: Phase modulates rigidity
    - Z_t -> Y_t: Phase mediates long-range
    - R_t -> Y_t: Rigidity constrains output
    """

    nodes: list[str] = field(default_factory=lambda: [
        "S_{t-2}",   # Spacing at t-2 (for CI tests)
        "S_{t-1}",   # Previous spacing
        "Z_t",       # Latent mode (2D)
        "R_t",       # Rigidity
        "Y_t",       # Target
    ])

    version: str = "0.2"  # Graph version

    edges: list[CausalEdge] = field(default_factory=list)

    def __post_init__(self):
        if not self.edges:
            self.edges = self._default_edges()
        self._build_nx_graph()

    def _default_edges(self) -> list[CausalEdge]:
        """
        Hypothesized edges (v0.2).

        Changes from v0.1:
        - Added S_{t-2} → S_{t-1} (temporal chain)
        - Added S_{t-1} → R_t (from CI-A FAIL in Round 001-003)
        """
        return [
            # Temporal chain
            CausalEdge(
                "S_{t-2}", "S_{t-1}",
                "Temporal ordering: past spacing affects next"
            ),
            # S_{t-1} edges
            CausalEdge(
                "S_{t-1}", "Z_t",
                "Local spacing encodes into latent phase"
            ),
            CausalEdge(
                "S_{t-1}", "Y_t",
                "Direct repulsion: S_{t-1} influences Y_t (short-range GUE)"
            ),
            CausalEdge(
                "S_{t-1}", "R_t",
                "Local spacing affects rigidity proxy (CI-A FAIL evidence)"
            ),
            # Z_t edges
            CausalEdge(
                "Z_t", "R_t",
                "Latent phase modulates global rigidity"
            ),
            CausalEdge(
                "Z_t", "Y_t",
                "Phase mediates long-range spectral correlations"
            ),
            # R_t edges
            CausalEdge(
                "R_t", "Y_t",
                "Rigidity constrains feasible spacings"
            ),
        ]

    def _build_nx_graph(self):
        """Build NetworkX DiGraph for d-separation queries."""
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes)
        for e in self.edges:
            self.G.add_edge(e.source, e.target, weight=e.weight)

    def add_edge(self, source: str, target: str, hypothesis: str):
        """Add a new edge to the graph."""
        self.edges.append(CausalEdge(source, target, hypothesis))
        self.G.add_edge(source, target)

    def remove_edge(self, source: str, target: str):
        """Remove an edge from the graph."""
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        if self.G.has_edge(source, target):
            self.G.remove_edge(source, target)

    def get_parents(self, node: str) -> list[str]:
        """Get direct parents of a node."""
        return list(self.G.predecessors(node))

    def get_children(self, node: str) -> list[str]:
        """Get direct children of a node."""
        return list(self.G.successors(node))

    def implied_independencies(self) -> list[tuple[str, str, set[str]]]:
        """
        Get implied conditional independencies from graph structure.

        Returns list of (X, Y, Z) tuples where X _||_ Y | Z
        (X independent of Y given Z).
        """
        independencies = []

        # For each pair of non-adjacent nodes
        for x in self.nodes:
            for y in self.nodes:
                if x >= y:  # Avoid duplicates
                    continue
                if self.G.has_edge(x, y) or self.G.has_edge(y, x):
                    continue

                # Find minimal conditioning set that d-separates
                # (simplified: just use parents)
                parents_x = set(self.get_parents(x))
                parents_y = set(self.get_parents(y))
                Z = parents_x | parents_y

                # Check if Z d-separates X and Y
                if self._d_separated(x, y, Z):
                    independencies.append((x, y, Z))

        return independencies

    def _d_separated(self, x: str, y: str, z: set[str]) -> bool:
        """
        Check if x and y are d-separated given z.

        Uses NetworkX's d_separated function.
        """
        try:
            return nx.d_separated(self.G, {x}, {y}, z)
        except nx.NetworkXError:
            return False

    def to_adjacency_dict(self) -> dict[str, list[str]]:
        """Convert to adjacency list format."""
        adj = {node: [] for node in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
        return adj

    def __str__(self) -> str:
        lines = [f"CausalGraph v{self.version} (Zeta Spacing)"]
        lines.append("=" * 40)
        lines.append("Nodes: " + ", ".join(self.nodes))
        lines.append("\nEdges:")
        for e in self.edges:
            status = "[x]" if e.validated else "[ ]"
            lines.append(f"  {status} {e.source} -> {e.target}")
            lines.append(f"      Hypothesis: {e.hypothesis}")
        return "\n".join(lines)

    def to_graphviz(self) -> str:
        """Generate DOT format for visualization."""
        lines = ["digraph CausalZeta {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=ellipse];")

        # Style nodes by type
        lines.append('  "S_{t-1}" [style=filled, fillcolor=lightblue];')
        lines.append('  "Z_t" [style=filled, fillcolor=lightgreen];')
        lines.append('  "R_t" [style=filled, fillcolor=lightyellow];')
        lines.append('  "Y_t" [style=filled, fillcolor=lightcoral];')

        for e in self.edges:
            style = "solid" if e.validated else "dashed"
            lines.append(f'  "{e.source}" -> "{e.target}" [style={style}];')

        lines.append("}")
        return "\n".join(lines)


# Pre-defined graph configurations for experiments

def minimal_graph() -> CausalGraph:
    """
    Minimal graph: only direct effects.
    S_{t-1} -> Y_t (repulsion)
    R_t -> Y_t (constraint)
    """
    g = CausalGraph(edges=[])
    g.add_edge("S_{t-1}", "Y_t", "Direct repulsion")
    g.add_edge("R_t", "Y_t", "Rigidity constraint")
    return g


def full_graph() -> CausalGraph:
    """Full hypothesized graph (default)."""
    return CausalGraph()


def latent_mediated_graph() -> CausalGraph:
    """
    Z_t mediates all effects.
    S_{t-1} -> Z_t -> Y_t
    R_t -> Y_t
    """
    g = CausalGraph(edges=[])
    g.add_edge("S_{t-1}", "Z_t", "Spacing encodes to latent")
    g.add_edge("Z_t", "Y_t", "Latent predicts output")
    g.add_edge("R_t", "Y_t", "Rigidity constrains")
    return g


if __name__ == "__main__":
    # Test graph creation and methods
    print("Testing CausalGraph...")

    g = CausalGraph()
    print(g)
    print()

    print("Implied independencies:")
    for x, y, z in g.implied_independencies():
        z_str = ", ".join(z) if z else "{}"
        print(f"  {x} _||_ {y} | {{{z_str}}}")

    print("\nGraphviz output:")
    print(g.to_graphviz())
