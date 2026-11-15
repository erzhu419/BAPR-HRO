"""V55 reductions composed with the retained V53 impact scheduler."""
from experiments.reconstruction_v53.priority import ImpactReference
from experiments.reconstruction_v55.reference import StructuralReference


class ImpactStructuralReference(StructuralReference, ImpactReference):
    """Keep outer chance reach, settled partials, and original root shares.

    StructuralReference tries the exact terminal integral first; its ordinary
    fallback resolves to ImpactReference through this cooperative MRO. The
    terminal integral spends the same cumulative budget without internal
    impact pruning, and retains full bounds when that budget runs out.
    """

    def solve(self, *args, **kwargs):
        result = super().solve(*args, **kwargs)
        result.update(implementation='structural_impact_conditional',
            scope='Same V53 causal targets, impact screening and original budgets; goal-dominance reduction and exact eligible terminal-platform integration. Terminal integration retains budget bounds without internal impact pruning. No new controller or independent method novelty claimed.')
        return result
