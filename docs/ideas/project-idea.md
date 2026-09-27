# Community-Aware Preemptive Misinformation Containment

## The Idea

Social networks are power-law skewed and community-structured. Most users are low-degree; a small number of influencers anchor distinct communities (a political leader anchors one ideological cluster, a lifestyle creator anchors another). Communities overlap, but each has its own character and its own representative accounts.

Misinformation cascades typically start *inside* a community and then **jump across communities through bridge nodes** — accounts that sit on the paths connecting otherwise separate groups.

The proposal is to intervene at those bridges, before the jump happens:

1. **Detect communities** in the follower/interaction graph.
2. **Identify bridge nodes** between them — selected by **betweenness centrality**, not degree.
3. **Use uncertainty sampling** (active learning) to pick which ambiguous, high-value bridge nodes are worth labeling and acting on, rather than labeling everything.
4. **Intervene at those nodes** to block cross-community propagation, rather than waiting for post-hoc detection.

### Why betweenness, not degree

A high-degree node may sit entirely inside one community — large reach, but contained. A high-betweenness node sits on the paths *between* communities, so gating one can cut off an entire cross-community flow. Fewer interventions, larger effect.

### Where the novelty claim sits

The existing literature does community-aware intervention **or** uncertainty sampling on graphs. Combining them — using uncertainty sampling specifically to select *bridge nodes* for preemptive intervention — appears to be the unclaimed space. This needs to be stress-tested against the literature before it's asserted.

### Positioning

This is classical graph algorithms plus active learning, not deep learning. That is a deliberate choice, not a limitation — it is cleaner, more interpretable, and more defensible at project scale.

---

## Things to Consider

### Conceptual

- **Does the premise hold?** Compare the bridge-node set against the top-degree set on real data. Heavy overlap weakens the whole argument; meaningful divergence is the first real result.
- **What does "intervention" concretely mean?** Node removal, edge removal, reduced transmission probability, delayed propagation, or injecting a counter-cascade? The choice changes the model, the metrics, and the realism of the claim.
- **Are communities semantically coherent?** Louvain/Leiden will always return partitions. Whether those partitions correspond to real audience groups is a separate question that needs inspection.
- **Overlapping membership.** Influencers belong to multiple communities. Hard partitioning may misrepresent the structure — consider whether overlapping community detection is needed.
- **Is preemption realistic?** Acting on a bridge node requires knowing the content is false *before* it crosses. That either assumes a fast upstream detector or shifts the framing from "preemptive" to "early containment."

### Methodological

- **Estimating β and γ** from real cascade data rather than assuming values. This is the hard part of the epidemic modeling and is worth raising explicitly with the advisor.
- **Which epidemic model.** SIR assumes permanent immunity after fact-checking; SIS allows re-belief; SEIR adds a latency stage between seeing and sharing, which maps well to misinformation. Justify the choice rather than defaulting.
- **Betweenness is expensive** — O(VE) exact. Approximate/sampled betweenness will likely be necessary at realistic graph sizes.
- **Uncertainty over what, exactly?** Uncertainty in the node's classification, in its bridging role, or in its expected downstream impact? These are different objectives and the choice needs to be stated.
- **Baselines must be honest.** Compare against: no intervention, random-k, top-k by degree, top-k by betweenness, and the uncertainty-sampled variant. Fixed intervention budget across all conditions.
- **Metrics.** Total infected, number of cross-community jumps prevented, and time-to-containment — all normalized per unit of intervention budget.

### Practical

- **Scope.** Detection *and* intervention is likely too much for one semester. The original contribution lives on the intervention side; detection could reasonably be an off-the-shelf classifier.
- **Data.** Needs nodes, edges, timestamped cascades, and ground-truth labels together. FakeNewsNet is the most likely fit; verify the graph is complete enough to compute meaningful centrality.
- **Validation ceiling.** Interventions cannot be tested on a live network. Everything will be simulated on a static historical graph, which limits how strongly the results can be stated.
- **Ethics.** "Identify the accounts to suppress" is structurally close to a censorship tool. Worth acknowledging in the writeup and worth deciding early whether interventions are framed as blocking, throttling, or labeling.
