# BAPR-HRO: Development Document
# Adaptive Hyperpath Routing under Non-Stationary Transit Delays

**Target**: Transportation Research Part C / Transportation Science
**Approach**: Engineering-driven, validated by experiments on real GTFS data

---

## 1. Problem & Motivation

> I waited for bus 402 (direct route). It never came. I took 102→transfer→317→transfer→another bus. At each transfer point I re-navigated and picked whichever bus came first.

**Gap in existing work**: Durner (2024) computes stochastic hyperpaths (sets of alternative connections at each stop, minimizing expected arrival time). But assumes **stationary** delay distributions. When bus 402 is canceled, the distributions are wrong and the hyperpath is stale.

**Our contribution**: Add two capabilities to Durner's algorithm:
1. **Regime detection** (from BAPR): detect when delay distributions shift in real-time via BOCD
2. **Fast re-planning** (from Cappart/Nair principles): neural surrogate to avoid full O(|C|^2) re-computation at every transfer point

---

## 2. System Architecture

```
OFFLINE (run once on historical data)
========================================
GTFS historical    →  Regime clustering (GMM/HMM)
                      → K delay distribution sets P(delay|r), r=1..K
                   →  Durner exact solutions (training labels)
                   →  Train V̂ ensemble (neural value networks)
                   →  Train GCN pruner (predict P(c ∈ hyperpath))

ONLINE (per user query, per transfer point)
========================================
GTFS-RT stream     →  BOCD regime detector
                      → regime_id, confidence
                   →  Select delay distributions P(delay|regime_id)
                   →  Route computation:
                        if confidence HIGH and V̂ agrees:
                            V̂ fast path → O(|C|)
                        elif confidence MEDIUM:
                            GCN pruned Bellman → O(|C|·k)
                        else:
                            Durner exact fallback → O(|C|^2·|T|)
                   →  Output: ranked alternatives + arrival distributions
```

---

## 3. Code Structure

```
BAPR-HRO/
├── data/
│   ├── gtfs/                    # Static GTFS timetable files
│   ├── gtfs_rt/                 # Historical GTFS-RT delay data
│   └── processed/               # Regime-labeled delay distributions
│
├── src/
│   ├── transit_graph.py         # Parse GTFS → time-expanded graph (stops, connections)
│   ├── delay_distributions.py   # Estimate PMFs from GTFS-RT historical data
│   ├── regime_clustering.py     # GMM/HMM to cluster delay patterns into K regimes
│   │
│   ├── durner/
│   │   ├── preprocessing.py     # Algorithm 5.1-5.2: DFS + cycle cutting + topological ordering
│   │   ├── topocsa.py           # Algorithm 5.3: Topological CSA (exact Bellman propagation)
│   │   ├── insert_label.py      # Algorithm 5.4: Insert departure label
│   │   └── distributions.py    # PMF operations: convolution, shifting, feasibility calc
│   │
│   ├── bocd/
│   │   ├── belief_tracker.py    # BOCD run-length posterior (adapted from BAPR)
│   │   ├── surprise.py          # Surprise signal: |actual_delay - predicted_delay|
│   │   └── regime_detector.py   # Online regime detection: BOCD → regime_id + confidence
│   │
│   ├── neural/
│   │   ├── value_network.py     # V̂ ensemble: predict E[arrival] from (stop, time, features)
│   │   ├── gcn_pruner.py        # GCN on transit graph: predict P(c ∈ optimal hyperpath)
│   │   ├── training.py          # Train V̂ and GCN on Durner's exact solutions
│   │   └── features.py          # Feature extraction for neural networks
│   │
│   ├── router.py                # Main routing engine: combines all components
│   └── evaluator.py             # Evaluation: run simulated journeys, compute metrics
│
├── experiments/
│   ├── configs/                 # Experiment configurations (YAML)
│   ├── run_baseline.py          # Run Durner static baseline
│   ├── run_adaptive.py          # Run our adaptive routing
│   └── analyze_results.py       # Generate tables and figures
│
├── tests/
│   ├── test_topocsa.py          # Verify Durner implementation against known examples
│   ├── test_bocd.py             # Verify BOCD detects injected regime shifts
│   └── test_router.py           # End-to-end routing tests
│
├── reference/                   # Cloned repos for reference
│   ├── public-transport-statistics/
│   ├── mlopt/
│   ├── hybrid-cp-rl-solver/
│   └── learntocut/
│
├── papers/                      # PDF papers
└── roadmap.md                   # This file
```

---

## 4. Dependencies

```
# Core
python >= 3.10
numpy
scipy
pandas

# GTFS parsing
gtfs-kit  or  partridge     # Parse GTFS static files
protobuf                    # Parse GTFS-RT (protobuf format)

# ML
torch >= 2.0
torch-geometric             # GCN for transit graph (GCN pruner)
scikit-learn                # GMM for regime clustering

# BOCD
# No external dep, implement from BAPR's bapr_components.py

# Visualization & evaluation
matplotlib
seaborn
tqdm
pyyaml
```

---

## 5. Data Sources

### GTFS Static Timetable
- **Switzerland (SBB)**: https://opentransportdata.swiss/en/dataset/timetable-2024-gtfs2
- **Germany (Deutsche Bahn)**: via DELFI e.V.
- **Alternative**: Use Durner's `public-transport-statistics` repo which already has processed Swiss data

### GTFS-RT Real-time Data
- **Durner's archive**: http://mirror.traines.eu (~70 GB/month, Swiss + German feeds)
- **Contains**: vehicle positions, trip updates, service alerts
- **Coverage**: 8+ months of historical data available

### What we need from the data
1. **Static**: stops, routes, trips, stop_times, transfers → build time-expanded graph
2. **Real-time**: for each connection, (scheduled_time, actual_time, delay) → build delay PMFs
3. **Regime labels**: cluster delay patterns by (day_of_week, hour, weather, disruption_events)

---

## 6. Implementation Plan (4 phases)

### Phase 1: Durner Baseline (Weeks 1-3)

**Goal**: Working implementation of Durner's exact algorithm.

**Steps**:
1. Parse GTFS → `TransitGraph` with stops, connections, transfers
2. Implement PMF operations (convolution, shifting, weighted sum)
3. Implement Algorithm 5.1-5.2 (DFS preprocessing, cycle cutting, topological ordering)
4. Implement Algorithm 5.3 (Topological CSA) with stop_labels
5. Implement Algorithm 5.4 (InsertDepartureLabel with domination checking)
6. Test on small examples, then Swiss timetable
7. Profile: measure runtime, identify bottleneck queries

**Validation**: reproduce Durner's reported <1 sec runtime and ~9.5 min improvement.

**Key data structures**:
```python
@dataclass
class Connection:
    id: int
    route: str
    dep_stop: int
    arr_stop: int
    dep_time_scheduled: int       # minutes from midnight
    arr_time_scheduled: int
    dep_distribution: PMF         # P(actual_dep_time)
    arr_distribution: PMF         # P(actual_arr_time)
    cancel_prob: float

@dataclass
class StopLabel:
    connection: Connection
    dest_arrival_distribution: PMF  # T_dest(c)
    mean_dest_arrival: float        # E[T_dest(c)]
    feasibility: float              # P_feasible(c)

class TransitGraph:
    stops: List[Stop]
    connections: List[Connection]    # sorted by topological order
    stop_labels: Dict[int, List[StopLabel]]  # stop_id → sorted labels
    cuts: Set[Tuple[int, int]]       # (c_a, c_b) cycle cuts
```

### Phase 2: BOCD Regime Detection (Weeks 4-5)

**Goal**: Detect delay distribution shifts in real-time.

**Steps**:
1. Extract delay time series from GTFS-RT historical data
2. Implement regime clustering (GMM with K=3..5 regimes)
   - Regime examples: normal, rush-hour, disruption, weather-delay
3. Port BAPR's `BeliefTracker` to transit context
   - Input: stream of (predicted_delay, actual_delay) pairs
   - Output: run-length posterior ρ(h), regime_id, confidence
4. Implement `RegimeDetector`:
   ```python
   class RegimeDetector:
       def __init__(self, regime_distributions, hazard_rate=0.05):
           self.belief_tracker = BeliefTracker(max_run_length=20, hazard_rate=hazard_rate)
           self.regime_dists = regime_distributions  # K sets of delay PMFs

       def update(self, predicted_delay, actual_delay) -> Tuple[int, float]:
           surprise = abs(actual_delay - predicted_delay)
           self.belief_tracker.update(surprise)
           confidence = 1.0 - self.belief_tracker.entropy / log(self.belief_tracker.max_h)
           regime_id = self._classify_regime(surprise, self.belief_tracker.effective_window)
           return regime_id, confidence
   ```
5. Validate: inject known disruptions into historical data, check detection delay

**Validation**: BOCD detects 90%+ of injected regime shifts within 3 observations.

### Phase 3: Neural Acceleration (Weeks 6-9)

**Goal**: Train V̂ and GCN pruner to speed up re-planning.

**Steps**:

**3a. V̂ Ensemble (value network)**:
1. Generate training data: run Durner exact on 10K+ (origin, dest, time, regime) queries
2. Extract features: (stop_id, time_of_day, day_of_week, avg_delay_at_stop, regime_id, ...)
3. Train K=5 MLPs: features → E[arrival_time_at_destination]
4. Ensemble disagreement σ_V̂ as uncertainty estimate
5. Evaluate: MAE of V̂ vs Durner exact, correlation, calibration of σ_V̂

**3b. GCN Pruner (connection pruner)**:
1. Build bipartite graph: connection nodes + stop nodes (like Nair et al.)
   - Connection features: (delay, time_to_dep, route_type, regime_features)
   - Stop features: (num_connections, avg_delay, transfer_time)
   - Edges: connection → arrival_stop, connection → departure_stop
2. Labels: from Durner exact, which connections are in the optimal hyperpath (binary)
3. Train GCN to predict P(c ∈ H*) for each connection
4. At inference: keep connections with P > threshold, run Durner on pruned subgraph

**3c. Integrated Router**:
```python
class AdaptiveRouter:
    def __init__(self, graph, durner, regime_detector, value_ensemble, gcn_pruner):
        self.graph = graph
        self.durner = durner
        self.regime_detector = regime_detector
        self.value_ensemble = value_ensemble
        self.gcn_pruner = gcn_pruner

    def route(self, origin, dest, time, gtfs_rt_delays):
        # 1. Detect regime
        regime_id, confidence = self.regime_detector.update(gtfs_rt_delays)
        delay_dists = self.regime_detector.regime_dists[regime_id]

        # 2. Choose computation mode
        v_hat, v_std = self.value_ensemble.predict(origin, time, regime_id)

        if confidence > 0.8 and v_std < threshold_low:
            # Fast path: trust V̂
            return self._route_with_value_network(origin, dest, time, delay_dists)

        elif confidence > 0.5:
            # Medium: GCN-pruned Bellman
            pruned_connections = self.gcn_pruner.prune(self.graph, time, regime_id)
            return self.durner.topocsa(origin, dest, time, delay_dists,
                                       connections=pruned_connections)

        else:
            # Fallback: full Durner exact
            return self.durner.topocsa(origin, dest, time, delay_dists)
```

### Phase 4: Evaluation & Paper (Weeks 10-14)

**Goal**: Comprehensive experiments + paper draft.

---

## 7. Evaluation Plan

### Dataset
- **Network**: Swiss timetable (SBB), ~6000 stops, ~200K connections/day
- **Period**: 8 months of GTFS-RT data from Durner's archive
- **Queries**: 1000 random (origin, dest, time) pairs, stratified by:
  - Short journeys (< 30 min, 0-1 transfers)
  - Medium journeys (30-90 min, 1-2 transfers)
  - Long journeys (> 90 min, 2+ transfers)

### Regime Shift Injection
Since we need controlled experiments, inject regime shifts into historical data:
- **Type A**: Single line cancellation (e.g., 402 disappears for 2 hours)
- **Type B**: Corridor disruption (all lines through a station delayed 15+ min)
- **Type C**: Gradual degradation (delays increase over 30 min, then recover)
- **Type D**: Real disruptions from historical data (if identifiable)

### Baselines
1. **Durner-Static**: Compute hyperpath once at origin, follow it (no re-planning)
2. **Durner-Periodic**: Re-compute every 5 minutes using latest GTFS-RT
3. **Durner-Oracle**: Re-compute with true future delays (upper bound)
4. **Google-Maps-Style**: Deterministic shortest path, re-route when deviation detected
5. **CSA-MEAT** (Dibbelt et al.): Stochastic approach from literature

### Our Methods
- **BAPR-HRO-Full**: BOCD regime detection + V̂ fast path + GCN pruning + fallback
- **BAPR-HRO-NoPrune**: BOCD + V̂ only (no GCN pruning)
- **BAPR-HRO-NoV**: BOCD + Durner exact (no neural acceleration)

### Metrics
| Metric | What it measures |
|---|---|
| **Mean arrival time** | Overall routing quality |
| **Worst-case arrival time** (95th percentile) | Robustness |
| **% of journeys improved** vs Durner-Static | Practical benefit |
| **Mean improvement (minutes)** | Magnitude of benefit |
| **Computation time per query** | Practical deployability |
| **Regime detection delay** (# observations) | BOCD quality |
| **False positive rate** of regime detection | Stability |
| **V̂ MAE** vs Durner exact | Neural surrogate quality |
| **Pruning recall** (% optimal connections kept) | GCN pruner quality |

### Expected Results (hypotheses to validate)
1. BAPR-HRO-Full improves mean arrival by 3-5 min over Durner-Static under regime shifts
2. Improvement grows with journey length and regime switch frequency
3. Under no regime shifts, BAPR-HRO matches Durner-Static (no degradation)
4. GCN pruning achieves 5-10x speedup with <1% optimality loss
5. BOCD detects regime shifts within 2-4 GTFS-RT observations (~2-4 min)

---

## 8. Paper Outline (Transportation Research Part C)

### Title
"Adaptive Stochastic Transit Routing under Non-Stationary Delays: Bayesian Change Detection with Learned Acceleration"

### Structure
1. **Introduction**: Bus story motivation, gap in Durner's stationarity assumption
2. **Related Work**: Stochastic routing (Durner, CSA-MEAT, RAPTOR), regime detection in transport, ML for routing
3. **Problem Formulation**: Durner's model + non-stationarity extension
4. **Method**:
   - 4.1 Regime-conditioned delay distributions
   - 4.2 BOCD for real-time regime detection (adapted from BAPR)
   - 4.3 Neural value surrogate V̂ for fast re-planning
   - 4.4 GCN connection pruner for medium-speed exact computation
   - 4.5 Adaptive routing engine (confidence-based mode selection)
5. **Experimental Setup**: Swiss timetable, regime injection, baselines
6. **Results**: Tables + figures for all metrics
7. **Discussion**: When does adaptive routing help most? Computational trade-offs.
8. **Conclusion**

### Key Figures
- Fig 1: System architecture diagram
- Fig 2: Example journey showing regime shift and adaptive re-routing
- Fig 3: Arrival time distributions: static vs adaptive under disruption
- Fig 4: Computation time comparison across methods
- Fig 5: BOCD regime detection on real GTFS-RT data
- Fig 6: V̂ accuracy vs training data size
- Fig 7: Pruning recall vs speedup trade-off

---

## 9. Reference Papers

| Paper | Role | Repo |
|---|---|---|
| Durner (2024) - Stochastic Strategies | Exact algorithm baseline + oracle | `public-transport-statistics/` |
| Bertsimas & Stellato (2019) - MLopt | Offline/online decomposition principle | `mlopt/` |
| Cappart et al. (2020) - RL+CP | V̂ replacing expensive DP principle | `hybrid-cp-rl-solver/` |
| Nair et al. (2020) - Neural MIP | GCN architecture + Neural Diving for pruning | — |
| Tang et al. (2019) - Learning to Cut | RL pruning agent design | `learntocut/` |
| BAPR (ours) | BOCD + ensemble uncertainty | `BAPR/jax_experiments/` |

---

## 10. MVP Definition

**Minimum viable result for paper submission:**

Phase 1 + Phase 2 alone (Durner + BOCD) is already publishable if experiments show significant improvement under regime shifts. Neural acceleration (Phase 3) adds engineering contribution but is not strictly required.

**Priority order:**
1. **Must have**: Working Durner + BOCD regime detection + experiments showing improvement
2. **Should have**: V̂ ensemble for speedup
3. **Nice to have**: GCN pruner
