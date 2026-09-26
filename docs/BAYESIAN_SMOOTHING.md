# Bayesian live win probability — thought process (1 page)

**Repo:** [ufa-live-win-probability](https://github.com/raymramin/ufa-live-win-probability)  
**Live compare:** [Side-by-side chart](https://raymramin.github.io/ufa-live-win-probability/site/compare.html)

## Thought process (plain English)

Win probability should feel like a **belief that updates as the game unfolds**, not a seismograph that jumps on every throw.

Start near **50–50**. The scoreboard and the clock are the strongest evidence: a **one-goal lead late** should move belief more than the **same lead early**, because there is less time left to come back. Large goal gaps should **not** keep stacking forever — going from +3 to +4 is not as informative as going from tied to +1 — so score enters as a soft “edge,” not a raw goal counter.

Field events still matter, but they are **weaker evidence**. A completion or a short gain should barely nudge the line. A **goal** is strong evidence and may move more late. A **turnover or other negative throw** is real bad news, but it should **not crash** win % when context already says the team is fine (early game with time to recover, or holding a lead). Smoothness is not “make the line flat” — it is “move only as much as the context deserves.”

That is the Bayesian idea: **prior belief from score × time**, then **gentle updates** from play-by-play, with step sizes gated by how surprising and how decisive the event is.

## How smoothness was achieved

1. **Train a calm backbone** — a heavily regularized model on soft score × game progress (plus weak possession / yards / skill terms), so the raw curve already prefers scoreboard story over noise.
2. **Event-gated live smoother** — each throw’s raw prediction is capped before it is mixed in: tiny steps on quiet throws, larger on goals (especially late), softer on turnovers (especially early or while leading).
3. **Time-aware blend (EMA)** — new evidence is blended with the previous live value using a half-life clock, so belief drifts instead of teleporting.
4. **End snap** — the final point lands on 0% or 100% once the game is decided.

## Smoothing paragraph (for the new graph)

The blue proposal chart is smoothed by treating win probability as a running belief: each throw produces a raw model estimate, but that estimate is not drawn raw. Quiet throws are only allowed to move the live line a little; goals may move it more, and more so late in the game; turnovers and other negative throws are deliberately muted so they do not erase a lead or early-game optimism that the score and clock still support. Those capped updates are then blended with the previous live value using a short memory (exponential moving average), which is why the blue line looks continuous and context-aware instead of spiky like the deployed orange path.

## What to look for on the compare page

- **Orange:** deployed Game Center path — informative but jumpy.  
- **Blue:** Bayesian smooth proposal — same games, calmer updates.  
- **Middle:** field number line for the current throw (start → catch).
