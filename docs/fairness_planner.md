# Fairness of the shift planner

The planner uses the scores of Part 1 to decide where and when someone works inside a contract. A manager keeps the final word. This note says what that allows and forbids, which test enforces each line, what the first audit found in the planner's own design and how it was fixed, and what the audit finds now. The method is in `docs/method_planner.md`; the audit is `python -m ree.cli planner audit` and its report is `outputs/planner/reports/fairness_audit.md`.

## The line, and the tests that hold it

| Scores may influence | Scores may not influence | Where it is held |
| --- | --- | --- |
| Which register or post a worker holds, especially in rush slots (the daily plan) | Contract type or weekly contract hours, and so pay | Contract hours are a hard rule of the roster model and of the checker (`WEEKLY_HOURS_CONTRACT`). In coverage-first solving a score-blind first stage fixes every worker's weekly paid minutes and the second stage keeps them: `test_scores_never_change_anyones_weekly_paid_minutes_in_coverage_first_mode` and `test_swapping_the_scores_of_two_workers_changes_nobodys_weekly_paid_minutes`. The audit checks that no variant differs from plan B in paid hours and keeps a negative control that must show the leak |
| Which of several equivalent shifts a worker takes: shifts with the same paid minutes and the same burden on the day (the refinement step) | Pay, bonuses, hiring, firing or discipline | The planner has no field for any of these; the API and screens return plans, not scores |
| Who is placed next to a provisional colleague (the learning quota) | The total burden of unpopular shifts a worker carries | Equivalent-shift swaps keep each worker's burden exactly (`test_the_refinement_never_trades_hours_or_burden_for_a_score`). The tier guard, a priced term in the model followed by an equal-hours repair step, limits the burden gap between score tiers to `objective.tier_epsilon` where it can; it is a limit, not a guarantee, and the audit reports the gap and the correlation of the score with burden |
| Which floater covers a rush-weighted register | Leave approval, swaps or availability decisions | Availability is only blocked time with no reason; swaps are validated by the checker and approved by a manager |

Two further lines are tested as code: the solvers, the daily planner and the live controller never import the simulator or the hidden true skills (`tests/planner/test_planner_architecture.py`), and the planner reads Part 1 only through `ScoreProvider`.

## What the first audit found, and the fix

The first complete audit of the full planner (plan C) found that the principle did not hold: the Spearman correlation between a worker's score and the signed deviation of their paid hours from the contract was +0.51, tier A was paid 1.27 hours a week more than tier C against their contracts, and the score correlated with burden per paid hour at -0.35. Everything stayed inside the 1-hour tolerance, so no rule was broken; but the scores were buying hours and burden inside the tolerance, which is exactly what the principle forbids. These figures come from the audit of 5 October 2026 on the first version of the solver, before the fix; the negative control below reproduces the mechanism.

Cause, measured by switching the solver stages off one at a time: all of it came from the score refinement step, which swapped shifts of different length and burden whenever a swap put a stronger score on a busier shift and stayed inside the tolerance (without that step the figures equalled plan B's). A 24-seed probe of the single-stage model showed that the CP-SAT score reward buys hours as well, because the reward grows with the length of the pattern.

Fix (decision P-50): the reward is off by default; the refinement swaps only shifts with equal paid minutes and equal burden on the day, so it cannot change anyone's hours or burden; the repair step for the tier guard swaps only equal paid minutes; in coverage-first solving the second stage fixes every worker's weekly paid minutes to those of the score-blind first stage; and coverage-first is the default for every store except the 10-worker test store. The audit keeps the old behaviour as a negative control (`objective.neutral_swaps: false`): if the control stops showing the leak, the audit has stopped being able to see it.

The cost: the tier guard now acts only through equal-hours swaps, so on small stores a chance gap above the limit can remain; and the roster use of the scores became small enough that the evaluation cannot separate it from chance (README, plan C against plan F).

## What the planner stores

Pseudonymous worker ids, contract hours, blocked time windows and two role flags (self-checkout only, preferred self-checkout). The reason behind a blocked window or a flag is not a field anywhere in the code, the API or the screens, so it cannot be stored or shown. Manager overrides and live decisions keep a short reason that the manager types; they are stored in the audit log (`planner_audit` table in SQLite).

## Human control

- Every roster, daily plan and live proposal is a proposal. A proposal changes nothing until a manager accepts it; accepted, rejected, blocked and stale proposals are all logged.
- A manual edit is checked by `check(plan, rules, requirement)`, which no solver can influence. An override that would introduce a rule error is refused with the checker's reason; one that is allowed needs a reason of at least three characters.
- Workers see their own plan with the reasons behind it in words (no ranks and no scores of anyone) and can ask whether a swap is legal; a legal swap still needs approval.
- Part 1's appeal route stays the way to contest a score.
- Posts-only mode (`objective.scores_in_roster: false`, `--posts-only`, `posts_only` in the API, a checkbox in the manager screen) makes the roster ignore the scores altogether; the daily plan still uses them to choose who stands at which register.

## Metrics the audit reports

Burden spread and largest score-tier gap in burden per paid hour; the Spearman correlation of a worker's score with the share of their shifts that fall on a weekend, late or split, with their burden, and with the signed deviation of their paid hours from the contract; the paid-hours difference to plan B for every worker; how many worker-days and how many shifts differ from plan B; the correlation of the number of blocked days with burden (restricted availability must not be left with the worst shifts); the rush minutes new workers get with and without the learning quota; how much of plan C survives noise added to the scores; and the negative control. Plan stability week to week and the share of requested swaps accepted need real use and are not measured here (`planner stability` measures how a re-solve moves assignments).

## What the audit finds now

Six simulated workforces of 24 cashiers, one week each, the same coverage in every variant (`planner audit`, table in `outputs/planner/reports/fairness_audit.md`). The checks and the reading, as generated:

- Largest score-tier burden gap over the plans that use the scores (C and its variants) and every seed: 0.307 (limit 0.35): within the limit.
- For comparison, plan B (no tiers) and the placebo E (shuffled tiers) reach 0.307, because no guard acts on the real tiers there.
- Largest hours deviation from a contract: 1.00 h (tolerance 1.0 h): within the tolerance.
- Largest paid-hours difference between a worker's weekly total in any variant (the negative control apart) and in plan B: 0.00 h (none allowed: the score-blind coverage stage fixes the hours): holds.

- Tier gap: plan C's largest tier gap is above plan B's in 0 of 6 seeds (mean difference +0.000).
- Score against burden per paid hour: plan C -0.06 (below zero in 4 of 6 seeds), plan B -0.06, placebo E -0.06. A value below zero means that workers with higher scores carry less of it.
- Score against share of late shifts: plan C +0.00 (below zero in 3 of 6 seeds), plan B +0.00, placebo E -0.00. A value below zero means that workers with higher scores carry less of it.
- Hours: the correlation of the score with the signed deviation of paid hours from the contract (inside the tolerance of 1.0 h) is -0.03 in plan C, -0.03 in plan B and -0.03 in the placebo E; paid minus contract hours of tier A minus those of tier C: +0.01 h a week in plan C.
- What the scores change: against plan B, plan C moves 0 worker-days (a person working on a day when plan B does not) and gives a different shift of the day to 0% of its shifts. The correlations of the score with the weekend share and with the split share are the same in every variant.
- Negative control: when shifts of different length and burden may be swapped (`objective.neutral_swaps: false`, the behaviour of the first version), the correlation of the score with the hours deviation is +0.51 and tier A minus tier C is +1.27 h a week, so the audit detects a score that buys hours; the default swaps (equal hours, equal burden) show -0.03 and +0.01 h.
- Learning quota (a floor of 90 rush minutes a week for each new worker): new workers get 638 rush minutes a week in plan C, and the roster without the quota is identical to plan C in every seed, so the quota does not bind here and this audit cannot show what it does.
- Noise in the scores: with normal noise of 0.25, 0.5, 1.0 (in z units) added to the scores the planner sees, 100%, 100%, 100% of plan C's shifts survive, so in these seeds the roster does not respond to the scores at all.
- Restricted availability: the correlation of the number of blocked days with burden per paid hour is +0.31 in plan B (no scores), +0.31 in plan C and +0.32 in the placebo E. Workers with more blocked days carry a heavier burden in every plan, the score-blind one included, so this is not an effect of the scores; it has not been removed.

## Limits

- The tier guard is a priced term followed by an equal-hours repair, not a hard constraint. In the audit of the 24-worker store it never had to act (plan C's roster equals plan B's in every seed). In the stress scenarios of the evaluation it acted in some seeds, and a gap above the limit remained in 1 of 8 seeds of the season-peak and small-team scenarios (`evaluation.md`). On the 16-worker store `planner roster --store small --seed 3` leaves a gap of 48.4 percent against the limit of 35, which the roster summary and the screens show.
- Tiers come from Part 1. If Part 1's tiers are wrong, the guard protects the wrong groups.
- The audit uses 6 simulated workforces. A pilot on real rosters, with the works council informed, is the real test of whether scores leak into shift quality.
- A low score never reduces hours, and the planner cannot see why a worker has restricted availability; it can only check that restricted availability does not correlate with a heavier burden, and in this simulation it does, for every plan including the score-blind one.
- The learning quota never binds in the audit, so its effect has not been shown.
- Hours are fixed by the score-blind stage in coverage-first mode only. The single-stage mode used for the 10-worker test store is for tests; the tier guard and the quota use tiers and flags there and can move hours (measured in a probe, not kept as a script).
