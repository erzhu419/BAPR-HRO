# V55: semantic goal barrier

Fix the V41 model, V50 physical menu and observation contract, received public
frames, and horizon `H`. The calculation below applies to the existing
`rolling_fixed` scorer, including its unchanged first-opportunity tail and
matched first-opportunity/single-target envelope. It is an exact reduction of
that scorer, not a new policy or an exact evaluation of its full execution.

**Proposition.** Let `g` be the first physical destination exit of a ride.
If its arrival is not publicly masked at the root, deleting the scoring exits
strictly after `g` preserves every pre-`g` action comparison, evaluated rolling
value, and boarding score. Recursively applying the deletion preserves the
entire executed policy before physical absorption.

**Proof.** On the ordered V41 support, nonnegative transfer times and causal
boarding imply that every fixed-history exit mark satisfies
`min(t,H) <= G(t) <= H`. For destination exits this is equality. For a platform,
each legal departure is at least its ready time and each child boarding mark
is at least its departure time, clipped at `H`. Both first opportunity and
each single-target expectation preserve this bound, as does their minimum.
This argument uses the supported departure clocks, at which child scoring
does not introduce an off-grid approximation.

At the unmasked destination, `G_g(t)=min(t,H)`. Any fixed target after `g` has
conditional cost at least `G_g(t)`: successful ordered paths cannot arrive
earlier, and failed mass costs `H`. The killed transition operator
`T_K(v)=H-K(H-v)` preserves order because `K` is nonnegative. Consequently,
the commitment to `g` dominates every later target when propagated to each
earlier stage. Removing those targets leaves the rolling comparison minima
unchanged. At `g` the policy absorbs at `min(t,H)` (or has cost `H` after the
horizon); backward policy evaluation therefore preserves earlier values and
boarding scores. Induct on the finite menu tree to include child scoring
platforms. With no future publications, later hard receipts can clear a mask
but cannot turn an initially clear goal into a masked goal. Thus the same
reduction holds after every supported causal history in the query.

**Read sets.** Removed scoring-only trips can be integrated out under the
fixed independent-trip law. A retained trip keeps its complete physical chain,
all hard evidence, and its original latest-message selector. In particular,
a forecast on a slot after `g` can alter the posterior before `g`; deleting
that likelihood is not justified. Original schedule ranks and their cache
metadata are retained separately. The executed menu and observation contract
are not rewritten. The reference's root P0 law is also unchanged.

**Limits.** A masked first destination is not a barrier: with costs
`(50,100,20)` at ordered times `(0,10,20)` and `H=100`, deleting the final exit
changes rolling CONTINUE to ALIGHT. The nominal heuristic is outside this
proposition: its mean-clock interpolation can change order across the clipped
horizon; the executable three-state counterexample is in
`tests/test_v55_semantic.py`. The compiled CONTINUE receipt at the destination
itself may change, but the physical executor has already absorbed there.
This reduction and a terminal subset/event DP remain standard dominance and
dynamic-programming tools; a speedup alone does not establish G2 novelty.
