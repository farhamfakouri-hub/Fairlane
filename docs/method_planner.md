# Part 2: Shift Planner and Daily Coordinator, Method

## Purpose and scope

Part 2 turns demand forecasts and Part 1's efficiency scores into three outputs: a weekly roster, a daily plan with breaks and replacements, and live adjustments during the day. A manager reviews and approves every plan, and the system explains each choice.

|  | Part 2 decides | Part 2 never decides |
| --- | --- | --- |
| Roster | Which shift pattern each worker gets on which day, within their contract | Contract type, weekly contract hours, hiring or firing |
| Day | Which register or post each worker holds in each time slot, and when each break happens | Anything outside the labour rules, or any pay matter |
| Scores | Where and when a worker is placed, so strong cashiers cover rush slots | How many hours a worker receives |
| Live | Proposed changes when someone is absent or a rush arrives early | Any change a manager has not accepted |

The boundary with Part 1 is a single interface, `ScoreProvider`, which returns each worker's peak and off-peak score, tier, confidence interval and flags. Part 2 reads scores through it and never sees raw register events, except an aggregated busyness signal described in the live re-planning section.

One principle governs the whole design: scores influence **where and when** someone works inside their contract, never **how much** they work. A low-rated cashier keeps the same contract hours as a high-rated one.

## Store facts the planner must respect

Every fact below describes the store the planner is built for. Anything marked assumed is a default to confirm.

| Area | Fact |
| --- | --- |
| Shift type 1: split | For example 9:30 to 14:00 with a break, a one-hour cut, then 15:00 to 20:00 with a second break |
| Shift type 2: continuous | 6 hours with one break near the middle, about 3 hours after the start |
| Break | 20 minutes; the exact time is flexible but must not pull too many people off the floor at once |
| Registers | 15 in total. Registers 1 to 7 are the busiest because the store layout ends at register 1, then 8 to 15 |
| Always open | Register 1 (priority lane for disabled customers) and register 4 (card-only lane) |
| Self-checkouts | 2 stations, supervised only. Some workers have priority for these posts, and register cashiers replace them when they take a break |
| Weekday rush | 10:30 to 14:00 and 18:00 to 20:00; weekdays are otherwise calm |
| Friday and Saturday | Busy days, with an extra rush from 8:00 to 9:00 |
| Sunday | Open 8:30 to 12:30, with the rush from 11:00 to 12:30 |
| Seasonal peaks | Christmas, 28 December to 4 January (as understood), and late August to early September |
| Opening hours | Monday to Saturday 8:30 to 21:30; Sunday 8:30 to 12:30 |
| Team size | About 25 cashiers per department in the prototype (assumed) |
| Contracts | Student contracts of 15 or 21 hours a week and full-time contracts; the contract sets the hours, not the score |

The relief practice already used on the floor is the model for the daily plan. The first person to break is replaced by a colleague who started a little earlier. That colleague keeps the register until their own break, and the returning worker takes over the register of the next person who breaks.

## How the planner is layered

The roster fixes who works when, the daily plan fixes where each person stands and when breaks happen, and live re-planning adjusts only what has gone wrong. Every plan passes the plan checker before the next layer uses it, and accepted live changes feed back into the daily plan.

## Demand forecast

The forecast gives the expected number of customers arriving in each 30-minute slot of the week ahead. It uses the store's own ticket counts, because every transaction at a register is one customer, and falls back to a profile built from the known rush windows until enough history exists.

The model is deliberately simple and explainable. For day d and slot h:

```latex
\hat{\lambda}(d,h) = L_{w(d)} \cdot P_{w(d),h} \cdot S_{s(d)} \cdot U_d
```

Here w(d) is the day of the week, L is the expected total customers for that weekday (an exponentially weighted average of recent same-weekday totals), P is the share of the day's customers who arrive in slot h (it sums to 1 over the day), S is a seasonal multiplier for the named periods (Christmas, 28 December to 4 January, and late August to early September), and U is a manual uplift for known events or public holidays that the manager can enter.

Customer counts are random, so the planner does not plan for the average. It plans for a chosen quantile of a negative binomial distribution fitted to past slot counts:

| Slot type | Planning quantile | Reason |
| --- | --- | --- |
| Rush slots | 0.80 (starting value) | A long queue costs more than an idle register |
| Calm slots | 0.50 | Idle capacity is waste with little benefit |

The forecast is tested by rolling-origin backtests: forecast week k from weeks before k, compare with actual counts, and report the weighted absolute percentage error, the bias, and whether the planning quantile is exceeded about as often as it should. If real counts are unavailable, the prototype uses the simulator's arrival process as the ground truth and reports that clearly.

## Staffing requirement per slot

For each slot the planner computes the smallest number of open registers that keeps customer waiting acceptable, using queueing theory instead of a headcount rule of thumb. The standard tool is the Erlang C model for c identical servers fed by one queue.

Let lambda be the arrival rate in customers per minute for the slot (from the forecast) and s the average time a register is occupied per customer, in minutes. The offered load, in Erlangs, is a = lambda times s, and the system is only stable when c is greater than a. The probability that an arriving customer has to wait is:

```latex
C(c,a) = \frac{\dfrac{a^{c}}{c!}\cdot\dfrac{c}{c-a}}{\displaystyle\sum_{k=0}^{c-1}\dfrac{a^{k}}{k!} + \dfrac{a^{c}}{c!}\cdot\dfrac{c}{c-a}}
```

The average wait and the chance that a wait exceeds a target time t are:

```latex
W_q = \frac{C(c,a)\, s}{c-a}, \qquad P(W > t) = C(c,a)\, e^{-(c-a)\,t/s}
```

The requirement c(t) is the smallest c for which P(W > t\_target) is at most alpha. The floor manager's tolerance is a queue: more than 5 customers waiting at every open register is too many. The planner converts that limit into t\_target (about 5 customers times s divided by the registers open, starting at 3 minutes) and the queue simulator also reports queue length. Starting values are alpha = 0.20 in rush slots and 0.35 in calm slots.

**Worked example.** In a slot, lambda = 1.0 customer per minute and s = 2 minutes, so a = 2 Erlangs. With c = 2 registers the system is not stable. The next rows show how quickly waiting falls:

| Registers c | Utilisation a/c | Chance of waiting C | Average wait Wq | P(wait over 3 min) |
| --- | --- | --- | --- | --- |
| 3 | 0.67 | 0.444 | 53 s | 9.9% |
| 4 | 0.50 | 0.174 | 10 s | 0.9% |
| 5 | 0.40 | 0.060 | 2 s | 0.07% |

The step from 3 to 4 registers is worth much more than the step from 4 to 5, which is exactly the trade-off a headcount rule cannot see.

**Which service time to use.** Part 1 measures the time from first scan to payment press, but a register stays busy a little longer (payment, bagging by the customer, the next customer arriving). The planner therefore uses an occupancy time estimated from Part 1's event stream as the median gap between successive transaction starts at a register that never idles, and the planner scales it by the average efficiency of the people expected on duty.

**Turning registers into people.** The number of workers needed on the clock in a slot is:

```latex
N(t) = \max\bigl(c(t),\,2\bigr) + k_{\text{sc}}(t) + \beta(t)
```

The two always-open lanes (registers 1 and 4) are counted inside c, k\_sc is the number of self-checkout supervisors (3 in calm periods, 5 in rush periods, as the store staffs them), and beta covers people on break, about the break share of a shift (20 minutes in a 6-hour shift is roughly 6%) applied to the staff on duty and rounded up. Registers open in a fixed order: 1 and 4, then 2, 3, 5, 6, 7, then 8 to 15.

**Limits of the model.** Erlang C assumes one shared queue and exponentially distributed service times. Real customers pick a register line, which is slightly worse, while real service times vary less than the model assumes, which is slightly better. The queue simulator in the evaluation plan checks the result, and the planner applies a safety multiplier if the simulation shows longer waits than the formula.

## Shift patterns and labour rules

The solver never invents free-form shifts. It chooses from a finite list of shift patterns, each defined by its start, end, cut and break window, and it checks every plan against a rule table. Every number in the rule table is a configurable parameter, and none of them is legal advice: they must be confirmed with HR, the store's collective agreement and the works council before a real deployment.

**Pattern families.**

- **Continuous:** 6 hours of work with one 20-minute break starting about 2.5 to 3.5 hours after the start. Start times every 30 minutes inside an allowed window.
- **Split:** two blocks separated by a one-hour cut, for example 9:30 to 14:00 and 15:00 to 20:00, with one break in each block. The planner offers only the split patterns the store already uses.
- **Short:** optional 4-hour patterns for student contracts, if the store uses them (open question).

Each pattern carries a burden value used by the fairness terms: late closing, Sunday and Saturday shifts, and split shifts are the least desirable.

**Rule table.**

| Rule | Default | Status |
| --- | --- | --- |
| Maximum work per day | 10 hours | General rule, confirm |
| Minimum rest between two shifts | 11 hours | General rule, confirm |
| Weekly rest | 35 consecutive hours | General rule, confirm |
| Break | Paid; 3 minutes per hour worked, rounded up: 10 minutes from 3 hours, 20 minutes at 6 hours | General rule, confirm |
| Maximum work per week | 48 hours, 44 on average over 12 weeks | General rule, confirm |
| Maximum consecutive days | 6 | Follows from weekly rest |
| Split shifts | At most 1 cut per day, length 1 hour | Store practice, confirm limits for part-time contracts |
| Weekly hours | Within a tolerance of the contract (students: 15 or 21 hours) | Tolerance assumed |
| Notice before a roster takes effect | No notice for manager changes; roster issued every 3 weeks | Assumed, confirm |
| Sunday work | Mandatory, not voluntary; Sunday patterns stay inside 8:30 to 12:30 and count as burden | Confirm rules and compensation |
| Breaks paid or unpaid | Paid: break time counts as worked hours | Assumed, confirm |

Availability is a second family of constraints: students declare the hours they cannot work (lectures, exams), and approved leave is blocked. A **plan checker** module validates any plan against this table independently of the solver, so a plan edited by hand in the interface is checked by exactly the same code.

## Weekly roster optimisation

The weekly roster is a constrained optimisation problem solved with integer programming (Google OR-Tools CP-SAT). It first guarantees coverage and legality, then uses the scores to put the strongest cashiers on the rush slots, then spreads the unpopular shifts fairly.

**Sets and data.** Workers w, days d, 30-minute slots t, and shift patterns p. The pattern table gives a(p,t) = 1 when pattern p is on duty in slot t, paid hours h(p) and burden b(p). From earlier sections come the requirement N(d,t), the contract hours H(w), availability A(w,d,p), and from Part 1 a standard score z(w) = (peak score - 50) / 15, set to 0 for unrated workers and clipped to the range -2 to 2. The rush weight omega(d,t) is the forecast demand pressure in the slot, scaled to 0 to 1.

**Decision variables.** x(w,d,p) is 1 when worker w works pattern p on day d. u(d,t) and o(d,t) are the shortfall and the surplus of staff in a slot.

| Constraint | Meaning |
| --- | --- |
| One pattern per day | At most one pattern per worker per day |
| Availability | x(w,d,p) is 0 whenever the worker is unavailable or on leave |
| Coverage | Staff on duty in each slot + u - o = N(d,t) |
| Contract hours | H(w) minus tolerance <= weekly paid hours <= H(w) plus tolerance |
| Rest | Pairs of patterns on consecutive days leaving under 11 hours of rest are forbidden |
| Days off | At least one full day off per week; weekly rest follows from the rest rule |
| Skills | Each slot has at least k\_sc workers cleared for self-checkout supervision and enough cashiers for registers 1 and 4 |
| Learning | Each provisional or unrated worker gets at least one peak slot per week next to a tier A colleague, so Part 1 can collect data |

**Objective.** Coverage dominates; scores and fairness refine what is left:

```latex
\min \; \sum_{d,t}\bigl(\kappa_{d,t}\,u_{d,t} + \kappa_o\, o_{d,t}\bigr) \;-\; \eta \sum_{d,t}\omega_{d,t}\sum_{w,p} a_{p,t}\, z_w\, x_{w,d,p} \;+\; \phi\,(B_{\max}-B_{\min}) \;+\; \psi\,\text{Change}
```

The shortfall penalty kappa grows with the rush weight, so an understaffed rush slot costs far more than an understaffed calm one. The third term rewards matching a high score z(w) with a high rush weight omega(d,t). Because the number of people on duty is already fixed by coverage, the solver does not add or remove staff to chase scores; it only decides who goes where. B(w) is each worker's burden, the sum of b(p) over this week's patterns plus the previous 8 weeks, so the spread B\_max - B\_min pushes late, weekend and split shifts to rotate. Change counts differences from last week's roster, because people prefer a stable routine.

**Fairness guard.** The score term must never turn into a punishment. The solver therefore adds a constraint that the average burden of each score tier stays within a tolerance epsilon of the overall average, so low-rated workers do not end up with all the worst shifts. A monthly audit also reports the correlation between score and burden.

**Uncertainty.** Part 1 supplies a confidence interval. Unrated workers get z = 0, and workers with a wide interval are shrunk toward 0, so a lucky streak does not win them the best slots.

**Solving.** With about 25 workers, 7 days and about 60 patterns there are roughly 10,000 binary variables, which CP-SAT handles within a time limit of 60 seconds (configurable). The run is made deterministic with a fixed seed and fixed number of workers, and returns the best plan with its optimality gap. If no feasible plan exists, the solver relaxes the constraints one group at a time in a fixed priority order and reports exactly which rule had to give way, instead of silently producing an illegal roster. A greedy plus local search heuristic serves as a fallback and as a baseline for comparison.

**Explanations.** Every assignment gets reason codes such as top available peak score for a rush slot, contract hours to fill, or fewer weekend shifts than average over the last 8 weeks, taken from the objective terms that drove it.

## Daily plan: posts, breaks and relief chain

The day before, the planner takes the roster and a refreshed forecast and produces a timeline for every worker on a 5-minute grid: which post they hold, when their break is, and who covers them. It works in two steps, breaks first, then posts, so each step stays explainable.

**Posts.** A post is a register (1 to 15), a self-checkout supervision spot, or floater. A floater is on duty, holds no required post and is not on break. Required posts at time t come in priority order: registers 1 and 4, one self-checkout supervisor, the further registers up to c(t) in the opening order, then any second supervisor.

**Step 1: break scheduling.** Each worker has a break length of 20 minutes and a window, for example from 2.5 to 3.5 hours after the start of a continuous shift, and one window per block in a split shift. A break may only start when someone can be spared, which is a capacity rule:

```latex
\text{OnDuty}(t) - \text{OnBreak}(t) \;\ge\; R(t) = c(t) + k_{\text{sc}}
```

This is solved as a small interval-scheduling problem (CP-SAT) whose objective spreads breaks evenly across the window, keeps breaks out of rush slots whenever the window allows, and never lets two workers from the same register zone leave at the same moment. In the weekly roster the slack beta(t) was reserved for exactly this, so Step 1 is normally feasible.

**Step 2: post assignment.** For every slot the planner assigns present, non-break workers to the required posts, minimising:

```latex
\sum_{t}\sum_{w,r}\bigl(m_{w,r} - \theta\,\omega_t\, z_w\, \pi_r\bigr)\, y_{w,r,t} \;+\; \sigma \sum_{t,w}\mathbf{1}\bigl[\text{post}(w,t) \ne \text{post}(w,t-1)\bigr]
```

Here y(w,r,t) = 1 when worker w holds post r in slot t, m penalises mismatches (a worker not cleared for self-checkout, or the supervisors with priority for that post), pi(r) is the importance of the post (registers 1 to 7 and the priority lane weigh more), and sigma is the cost of a handover. A worker keeps their register by default, which keeps changes small. A first version solves one slot at a time with the Hungarian algorithm and a continuity bonus; a later version solves the whole day jointly with CP-SAT and is compared with the first.

**The relief chain as a rule.** When worker A is due for a break from register r at time tau:

1. If a floater f exists, f takes register r at tau. The floater with the highest score is chosen when r is in a rush-weighted zone, otherwise the one waiting longest.
2. A leaves for 20 minutes. If A had priority for self-checkout, f or the next floater covers that post instead.
3. When A returns, A becomes a floater, or goes straight to the register of the next worker B who is due to leave. f keeps register r until f's own break.

| Time | Event | Result |
| --- | --- | --- |
| 11:50 | Floater W03 takes register 2 from W07 | W07 free, register 2 still open |
| 11:50 | W07 starts a 20-minute break | One person on break |
| 12:10 | W07 returns | W07 is a floater again |
| 12:10 | W12 is due for a break from register 3 | W07 takes register 3, W12 leaves |

Worker ids are illustrative. The floater at the start of the chain comes from the roster itself: a worker who starts about 10 minutes before the first break is due, as is already done on the floor.

**Output.** A board of worker timelines, a list of handovers with times and the reason for each, and an alert list for any slot whose coverage falls below R(t). Any manual change by the manager is revalidated by the plan checker immediately.

## Live re-planning

During the day the planner watches for problems and **proposes** fixes. It never changes the floor on its own: every proposal is a card that the manager accepts or rejects.

**Triggers.**

| Trigger | Detection | Typical proposal |
| --- | --- | --- |
| Absence or lateness | Worker not logged in 10 minutes after the start | Cover from a floater, shift a break, or ask a volunteer who is off duty |
| Rush starting early | Register busyness above 90% for 10 minutes | Open the next register in the opening order with a floater |
| Calm period | Busyness below 55% for 20 minutes with more than one floater | Bring a break forward, or close the last register opened |
| Manager event | Delivery, system outage, special promotion entered by hand | Re-plan the rest of the day with the new requirement |

The percentages and minutes are starting values to tune. The thresholds use hysteresis, with different levels for opening and closing, so the plan does not flip back and forth, and the same type of proposal is limited to one per 10 minutes.

**The busyness signal.** Part 1's register events provide, without any worker identity, the time each open register spends serving customers. Over a window of length Delta minutes:

```latex
\hat{\rho}(t) = \frac{\sum_{r\,\in\,\text{open}} \text{busy}_r(t-\Delta,\,t)}{|\text{open}|\cdot \Delta}
```

A value near 1 means every open register has been serving customers continuously, which is when queues form. The arrival rate seen live is blended with the forecast, lambda\_live = a times observed + (1 - a) times forecast, and the Erlang C section's formulas then give the expected wait with and without the proposed change, shown on the card.

**Stability rule.** A re-plan freezes the past and the next 10 minutes, then re-solves the daily plan for the rest of the day with an extra penalty mu for every assignment or break that differs from the current plan. The manager sees few, small changes instead of a reshuffled day.

**Boundaries.** The planner never proposes more than the labour rules allow, never asks someone to stay past their contracted shift without their agreement, and treats calling in an off-duty colleague as a request to a volunteer, not an order. Each proposal, decision and outcome is stored in the audit log, so the system can later report how often its advice was accepted and whether it helped.

## Fairness and human control

The planner uses scores to decide **where and when** inside a contract, and a manager keeps the final word. The line between acceptable and unacceptable uses is written down so it can be tested.

| Scores may influence | Scores may not influence |
| --- | --- |
| Which register or post a worker holds, especially in rush slots | Contract type or weekly contract hours |
| Which slots, within the same hours, a worker is placed on | Pay, bonuses, hiring, firing or discipline |
| Who is placed next to a provisional colleague as a mentor | The total burden of unpopular shifts a worker carries |
| Which floater covers a rush-weighted register | Leave approval, swaps or availability decisions |

**Rotation of unpopular shifts.** Burden is tracked over a rolling 8 weeks and balanced in the weekly roster, within each score tier as well as overall. A worker with a low score still gets Saturday afternoons, and a worker with a high score still gets quiet weekdays and Sundays in turn.

**Fairness metrics reported every week.**

| Metric | What it shows |
| --- | --- |
| Burden spread, max minus min over 8 weeks | Whether unpopular shifts rotate |
| Tier burden gap | Whether any score tier carries more than its share |
| Correlation between score and weekend, late or split frequency | Whether scores leak into shift quality |
| Hours deviation from contract | Whether anyone is over- or under-scheduled |
| Share of requested swaps accepted | Whether workers have real flexibility |
| Plan stability week to week | Whether people can organise their lives around the roster |

**Human control.** The manager can lock any assignment, forbid a pairing, and override any suggestion with a short reason that is stored. Workers see their own plan with the reason codes behind it, can propose swaps that the plan checker validates, and can contest a score through the appeal route built in Part 1.

**Availability data.** Students declare when they cannot work. The planner stores only blocked time, never the reason, and a limited availability is never penalised in a score or in the objective. A short availability window can look like a neutral attribute while correlating with studies, caring duties or health, so the fairness audit checks that workers with restricted availability are not systematically left with worse shifts.

**Process.** As for Part 1, workers are informed beforehand, and the works council is informed and consulted before the planner is used on real rosters. The audit log keeps every plan version, override and live proposal with its outcome.

## Architecture

Part 2 extends the Part 1 repository as a new package. It reads Part 1 only through the read-only score interface and the aggregated busyness signal, so Part 1 can improve without touching the planner. The plan checker is a separate module that every solver output and every manual edit must pass, and the evaluation harness is the only code allowed to read hidden true skill.

| Module | Responsibility |
| --- | --- |
| forecast and need | Customers per slot and registers needed per slot |
| patterns and rules | Shift pattern table and labour-rule parameters, loaded from configuration |
| roster solver | Weekly CP-SAT model, fairness tracker, explanations |
| daily planner | Break scheduling, post assignment, relief chain |
| live monitor | Busyness signal, triggers, proposals, stability |
| plan checker | Independent validation of any plan against the rule table |
| store and audit | SQLite storage of plans, versions, overrides and outcomes |
| API and screens | FastAPI service, manager screen and worker view in Streamlit |
| evaluation | Queue simulator, plan comparison, reports and figures |

## Evaluation plan

The planner is judged by running the same store day under different plans and comparing what customers experience. This needs a **queue simulator** added to the Part 1 simulator: customers arrive following the true demand, join a register line, and are served at a speed set by the cashier's hidden true skill. The planner can never read that hidden skill; only the evaluation harness can.

**Plans compared**, always with the same staff, contract hours and customer arrivals:

| Plan | What it does | Purpose |
| --- | --- | --- |
| A. Headcount only | Same number of people each hour, breaks at the middle of the shift in a random order | Proxy for today's practice, to be calibrated against how the store really plans |
| B. Demand-aware | Weekly solver and daily plan with all scores set to zero | Value of forecasting and queueing alone |
| C. Full planner | B plus Part 1's estimated scores | The proposed system |
| D. Oracle | B plus the true hidden speeds | Upper bound on what scores can add |
| E. Placebo | B plus scores shuffled between workers | Proves that gains come from score information, not from the code path |

**Metrics.** Customer wait (mean and 90th percentile, overall and in rush slots), share of customers who wait over 3 minutes, abandonment when a wait exceeds a patience limit, throughput per hour, understaffed and overstaffed minutes, breaks outside their window, handovers per day, plan stability, solve time, and the fairness metrics of the previous section. Constraint violations are not a metric but a requirement: every plan produced must pass the plan checker with zero violations.

**Scenarios**, each over many seeds: a normal week; the Christmas week with a seasonal peak; a Sunday; two unplanned absences; a forecast that is wrong by 30%; a small team against a large one; and noisy scores against accurate ones. A sweep over how much cashiers differ in speed shows where the scores add value, because if everyone is equally fast, ranking people cannot help.

**Reporting.** Results are paired by seed with confidence intervals, expressed in practical units such as seconds of waiting saved per rush hour. The report must say plainly where the full planner wins, where it ties, and where it loses, including the placebo and low-dispersion cases. As with Part 1, all results come from a simulator built on stated assumptions, and a pilot with real tickets and real shift data is the real test.

## Order of construction

The plan checker comes first because every other module depends on it. The evaluation comes before live re-planning on purpose, so a measured result already exists with a weekly roster and a daily plan; it is the credibility core of the method, as in Part 1.

## Store answers and open questions

Store answers, with what each changes in the planner:

| Topic | Store answer | Effect on the planner |
| --- | --- | --- |
| Roster owner and horizon | The store manager issues the roster every 3 weeks; a second manager runs the floor all day (breaks, posts), but also covers reception, other tasks and register problems. Most changes come from students and their hours | Roster horizon is 3 weeks, not 1. The daily plan and live proposals are an add-on for the floor manager, who is often away, so they must be quick to read and easy to override |
| Ticket history | No access | Forecast starts from the simulated profile; real counts are collected by the first pilot |
| Breaks | Paid. 3 minutes per hour worked, sometimes rounded up: 20 minutes at 6 hours, 10 minutes at the 3-hour minimum shift | Break length = ceil(3 min per hour); minimum shift 3 hours; break time counts as work |
| Self-checkouts | Two zones, each supervised separately. Zone 1 needs 1 person (2 in rush); zone 2 needs 2 people (3 in rush). Floaters do other work in the store and do not wait idle. Some workers usually hold these posts because they are better there, others for medical reasons | Two self-checkout posts with their own counts (1 and 2 calm, 2 and 3 rush), so k\_sc = 3 calm and 5 rush in total. Each worker has an optional post flag (preferred or restricted) that the solver treats as a constraint; the reason is never stored. A floater is a useful extra person for relief, not a standing-by cost |
| Shift pattern | Example 9:30 to 14:00 then 15:00 to 20:00. Nobody opens and closes the same day (never 8:30 to 21:30) | Add a rule: no pattern spans both the opening and the closing |
| Clock-in versus register time | Clock in 10 minutes before the start; at the register at the start. At the end the register closes 5 minutes before the end of the shift | Two times per shift: paid time and register time. Coverage counts register time only |
| Student contracts | 15 or 21 hours a week; availability is not fixed and changes | Availability is read per 3-week roster; contract hours are exact |
| Sunday and notice | Sunday work is not voluntary; no notice for changes | Sunday enters the burden rotation; stability is a soft goal, not a legal limit |
| Registers open today | 8 to 9 of 15 at rush, both self-checkouts fully open | Baseline plan A uses 8 to 9 registers in rush |
| Other roles | No overlap with the register team | Cashiers form a separate pool |
| Service target | A queue of more than 5 customers at every register is too long | Target expressed as queue length, converted to a wait target and checked in the simulator |
| Shift length | 4-hour shifts are used | Shift patterns include 4-hour blocks, which fit rush windows exactly and suit students |
| Evening rush | 20:00 to 21:00 can be rush, especially Friday and Saturday; the last 30 minutes (21:00 to 21:30) are never rush | Rush weight omega is high on Friday and Saturday 18:00 to 21:00, low from 21:00; closing staff are not needed at full strength in the last 30 minutes |
| Break timing | The break falls in the middle of the shift (after 3 hours in a 6-hour shift). People who start together cannot break together: 2 at a time, 3 at most | Break window is centred on the middle of the shift; at most 2 workers on break at once (3 allowed in a limit case), which staggers breaks of workers who start together |
| Split shifts | One cut per day | At most 1 cut per day, one hour long, now a store rule rather than an assumption |
| Floaters | 2 floaters from 8:30 to 14:00 and 2 from 14:00 to 21:30; they do other work in the store | Fixed floater posts in both halves of the day, counted in the break capacity rule |
| Moving breaks | The floor manager may move a break when the store is busy; it is his job, but never by about 2 hours or more | Break flexibility is a soft tolerance of at most 90 minutes from the planned time; live proposals respect it and the manager can override |
| Cut length | One hour in most cases; one 1 hour 30 minute cut appears in the rosters | Cut length 1 hour by default, up to 1.5 hours allowed |
| Month-start peak | Around the 28th to the 4th of the month, shops are busier because salaries arrive; the evening rush to 21:00 can also appear on weekdays | Forecast factor S gets a month-start multiplier on days 28 to 4, estimated from data or set by the manager |
| Christmas peak | 20 December to 3 January is by far the busiest period of the year; the average line has 10 or more extra big trolleys | Seasonal multiplier on arrivals, and also a shift in the basket mix towards big baskets, which raises the service time s in Erlang C; highest rush weights, planned at a high quantile |
| Self-checkout specialists | A few workers are labelled as priority for self-checkout and never work a normal register | A per-worker role flag, self-checkout only. The solver and checker enforce it as a hard rule: these workers fill self-checkout posts and nothing else; the reason is never stored |
| Shift times | Four weekly rosters of the store (Sunday to Saturday) show that shifts start on any 5-minute mark and last from about 3 hours to 9 hours 30 minutes. Closers end at 21:35 paid time (register until 21:30); Sunday shifts run from about 9:00 to 12:55; most shifts are continuous, with a minority of split shifts | Patterns are generated from a start grid of 5 minutes and fixed anchor ends, and loaded from a configurable file, so there is no short list of fixed shifts |
| Roster format | Weeks run Sunday to Saturday. Each cell has paid time and register time (for example 09:25 to 12:55 paid and 09:35 to 12:50 on the registers), a total per shift and per week, and an absence code (sickness, excused, leaving this month, potential) | The roster format of the planner mirrors this format: paid window, register window, absence codes as availability |
| Weekly totals | Weekly paid totals cluster around 15 h 50 and 21 h 10 for the students, a few workers reach 26 h to 37 h | Targets are the weekly paid totals including the 5 and 10 minute margins; tolerance is configurable |
| Roles in the roster | Every row is a cashier; nobody works two roles | All workers in the roster form the register pool; no cross-role availability is modelled |

Still open:

- Are the 5-minute start times chosen freely by the manager, or tied to rules such as the rush windows?
- How many more customers does Christmas bring than a normal Saturday? The dates (20 December to 3 January) and the longer lines (10 or more extra big trolleys per line) are known.
- What are the other shift start times and cut lengths in use? (4-hour shifts are used.)
- Does any weekday evening in the middle of the month reach a rush from 20:00 to 21:00?
- Roughly how many more customers come in the 28th to 4th window than in the middle of the month, and how many workers are labelled self-checkout only?
