# Frozen-date calibration

Dates are fixed in `date_manifest.json` before target values are opened. Root selection uses the original V34 structural order and at most12 roots per date; V50 removes the original four-vehicle cap and the existing prefix closure supplies the new menu. Every selected root and structural candidate/later-exit pair remains registered. The original V41 parameters, physical executor, message selection, station MCT and3600-second horizon remain unchanged.

GTFS trip IDs are scoped to a static release. On March16–19, an independent public-static comparison proves a bijection between634 old/new vehicle-days, preserving operator, train number, route and every regional station, sequence, scheduled clock, pickup/dropoff and platform. It preserves the frozen graph's physical chains and source-connection IDs. Decoding every raw ID through the inverse bijection therefore restores the same named physical vehicles without editing any raw numerical forecast or receipt clock. The resulting consumed histories and posteriors can change relative to an incorrect literal-ID join. In particular,38 IDs denote different services across releases, so an exact string match cannot bypass translation. All716 registered transfer-pair rules also agree across these releases. The original literal-ID receipts are retained in `literal_id_receipts`; only the four input-decoding repairs are recomputed before the single final bootstrap.

At the first onboard decision clock τ, let U be catching a named outgoing vehicle at the current exit and V catching that same vehicle at a later exit. Let q(x,y) be that vehicle's **joint** posterior departure law. Let a(t) be the incoming vehicle's subprobability of reaching the later exit at time t without cancellation, deadline or execution-order failure. The frozen distinct-vehicle model factors these two laws, conditional on the received history h. With MCTs m₀,m₁,

```
pU  = Σx,y q(x,y) 1{τ+m₀ ≤ x < H, departure active}
b(y)= 1{y < H, departure active} Σt≤y−m₁ a(t)
pV  = Σx,y q(x,y)b(y)
p11 = Σx,y q(x,y)1{τ+m₀ ≤ x < H, departure active}b(y)
(p00,p01,p10,p11) = (1−pU−pV+p11,pV−p11,pU−p11,p11).
```

This is the existing finite-chain integration, with no independence assumption between the outgoing vehicle's two stops. Its own hidden intermediate slots propagate probability; only executed incoming path endpoints impose physical failure conditions. Future messages never advance during this calculation. Same-chain or unsupported histories remain unresolved.

The archived label is separately reconstructed from observed endpoint times and consistent named-journey identities. Missing labels retain their feasible partition set S. For partition k its residual interval is `[min(s∈S)1{s=k}−p_k, max(s∈S)1{s=k}−p_k]`; unknown probabilities additionally range over[0,1]. A missing initial onboard history cannot be replaced by its scheduled clock. The scheduled registration preserves that root's denominator, while its probabilities and labels remain unknown. An absent whole predecision message archive is also unknown, not evidence that no public message was issued.

Events sharing either named incoming or outgoing vehicle form connected incidents within each date. Residuals are averaged equally within incidents, then over incidents, then over dates. The unchanged20,000 whole-date bootstrap draws use seed20261115 and quantiles0.00625/0.99375 for each of the four coordinates. All14 date denominators and simultaneous intervals inside±0.02 are required for equivalence. An interval wholly outside the margin is a failure; a crossing interval or missing date denominator remains unresolved. The opportunity-weighted result and every prespecified descriptive group remain secondary.

## Limitations

These are previously unused historical March2023 targets evaluated with models fitted on later October data, not prospective validation. The interval is an approximate14-date cluster screen, not a distribution-free guarantee. GTFS-RT comes from a third-party research collection of the official feed. Its filename clock denotes archival capture; the public source does not prove HTTP response completion or operator delivery time. Directory metadata initially established daily-file existence, but did not establish morning coverage or the feed's active static release. March9 starts at12:02 and stays in the panel as missing; directory-based ID alignment on March16–19 was repaired using the source's publication rule and the public service bijection. Published station/platform assignments and transfer rules establish planned feasibility only; the2023 actual archive supplies no observed platform identities or measured passenger walking times. Calibration alone cannot establish action regret, algorithm novelty or G2.
