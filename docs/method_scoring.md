# Part 1: Efficiency Scoring Engine, Method

## Scope

This engine turns raw register events into a fair, explainable efficiency rating for each cashier, and does nothing else. It is Part 1 of 2: the shift planner (Part 2) only reads its scores through one interface, so each part can be built, tested and sold on its own.

|  | In scope now | Later or elsewhere |
| --- | --- | --- |
| Inputs | Register timing events, basket size class, weekly cash-discrepancy report | Attendance, manager ratings, skills matrix (added as extra inputs) |
| Outputs | Speed score, accuracy score, combined score, confidence level, per-worker explanation | Rankings used to build shifts, breaks and replacements (Part 2) |
| Data kept | Counts and timings only, never customer or item identity | Anything about customers |

One correction to the original idea: counting how many items pass through a till during a 6-hour shift measures the queue, not the person. A fast cashier on a quiet Tuesday scans fewer items than a slow one on a Saturday. So the engine measures speed only while a transaction is active, and compares each transaction with what was expected for that basket.

## Signals the plugin captures

The plugin records one row per transaction, timed from the first scanned item to the moment the cashier presses payment. Loyalty card scans and coupons fall inside that window, so they count as part of the work.

| Signal | How it is captured | Why it matters |
| --- | --- | --- |
| Start and end time | First scan event, payment-button event | Active duration of the transaction |
| Item count | Number of scan events | Basket size and basket class |
| Basket class | Small under 20 items, medium 20 to 39, big 40 and above | Compares like with like |
| Hand-keyed items | Items sold per piece, quantity typed by hand | Under a second each, so a small fixed cost; about 15% of items (fruit, vegetables, bread) |
| Tagged items | Bottles and boxed items with an anti-theft tag; a button approves the age where needed | A fixed 5 to 7 seconds per tagged item |
| Loyalty card, reduction vouchers | Events inside the window | Fixed extra seconds per transaction |
| Exceptions | Voids, price checks, supervisor calls, manual entries | Flags the transaction so it is not judged as pure speed |
| Register and lane type | Registers 1 to 7 (busiest) and 8 to 15, register 1 as the priority lane for disabled customers, register 4 as the card-only lane | Customer flow and mix differ by position in the store and by lane |
| Cashier session | Login and logout on the register | Which person owns which transactions |

Time between transactions (waiting for the next customer) is never counted. Bagging is done by the customer, so it is not part of the cashier's time.

### Pause rule for interruptions

Transactions get interrupted by a supervisor call, a cheque or ID check, a price check, or a customer searching for a card. These delays are not the cashier's speed, and nobody can type them in by hand, so the plugin handles them automatically with two rules.

Rule 1 caps every gap. Between consecutive events (scan, hand-keyed item, tag removal, reduction voucher, loyalty card) a gap g counts only up to a cap T\_idle, and the extra time beyond it is dropped:

```latex
d_i = \sum_{j} \min\left(g_{i,j},\ T_{\text{idle}}\right)
```

Rule 2 marks the whole transaction as interrupted when any gap exceeds a longer limit T\_stop. It is then left out of the speed score and still counts for accuracy. Where the register software exposes explicit events (supervisor call, cheque payment, price check), those mark the transaction as interrupted directly.

A hard stop at 5 seconds would be too strict for two reasons. Removing an anti-theft tag takes 5 to 7 seconds by itself, so normal work would be cut. Worse, a cashier who steadily waits 6 seconds between items would have most of their time deleted and look like the fastest. Capping keeps the slowness inside the score and removes only the unusual delays. Starting values are T\_idle = 15 s and T\_stop = 60 s, and the engine tunes them from the store's own gap distribution, putting the cap where the long tail begins.

## Speed metric

Speed is the ratio between the time a transaction should have taken and the time it did take. Raw items per minute is not used, because a basket of 5 items always has a lower rate than a basket of 40 (the loyalty card and payment step cost the same fixed time).

### Version 1: compare same-sized baskets (start here)

Group transactions into bins of similar size (1 to 5 items, 6 to 10, 11 to 15, and so on), separately for each lane type and for peak or off-peak hours. For cashier w in bin b, T\_wb is the median duration of that cashier's transactions, and T\_b is the median of the whole store for the same bin. The cashier's score in that bin is the log of the ratio, and the overall speed is the average over bins, weighted by the share p\_wb of the cashier's transactions that fall in each bin:

```latex
s_{w,b} = \ln\!\left(\frac{T_b}{T_{w,b}}\right), \qquad S_w = \sum_b p_{w,b}\, s_{w,b}
```

Transactions flagged as exceptions are removed first, and the shrinkage of Step 3 below is applied to S\_w. This needs no fitted model and can be explained to a manager in one sentence: how long you take for a basket of this size, compared with your colleagues. Its weakness is that it assumes cashiers meet similar baskets. If some of them systematically get more age checks or coupons, version 2 corrects for it.

### Version 2: expected-duration baseline (added once enough data exists)

The four steps below replace the bin medians with a fitted price list for each basket, so every transaction is judged against its own expected time. The simulator runs both versions on the same planted cashiers, and the one that recovers their true order better becomes the default.

Step 1: learn the expected duration from the whole store's recent transactions, separately for each lane type and for peak or off-peak hours. With n items, m hand-keyed items, a tagged items, l a loyalty card (0 or 1) and k reduction vouchers (bons de réduction):

```latex
E_i = \alpha + \beta\, n_i + \gamma\, m_i + \delta\, a_i + \lambda\, l_i + \mu\, k_i
```

Step 2: score each transaction on a log scale, so that being 1.25 times as fast and 1.25 times as slow cancel out exactly. Here d\_i is the measured duration:

```latex
s_i = \ln\!\left(\frac{E_i}{d_i}\right)
```

Step 3: take the median of s\_i over a worker's valid transactions (flagged exceptions and durations beyond 3 median absolute deviations are removed), then shrink it toward the store average so that a worker with few transactions cannot get an extreme score by luck. With n\_w valid transactions and k about 150 (tunable):

```latex
S_w = \frac{n_w}{n_w + k}\,\operatorname{median}(s_i)
```

Step 4: convert to a comparable scale, where Z\_w is the standard score of S\_w among peers:

```latex
\text{SpeedScore}_w = 50 + 15\, Z_w \qquad Z_w = \frac{S_w - \bar S}{\sigma_S}
```

Two versions are kept for every worker, SpeedScore\_peak (10:30 to 14:00, 18:00 to 20:00, and weekend rushes) and SpeedScore\_off. The planner needs the peak one for rush slots, because some people speed up under pressure and others slow down. A worker with under 4 weeks of tenure is labelled provisional, since new cashiers are still on the learning curve.

## Accuracy metric

Accuracy combines two signals: the weekly cash-discrepancy sheet, which is slow but objective, and error events the plugin sees live, which are fast but need care. Speed without accuracy is worthless, and the accuracy score is also the main protection against gaming, for example scanning fast while skipping items.

From the weekly sheet, the sheet names both the register and the cashier, so each discrepancy goes directly to its cashier; it is split in proportion to the cash handled only when two cashiers shared a till. Small differences are normal (coins, rounding), so a tolerance tau (to be agreed with a manager, for example 1 euro per week) is ignored. Both surplus and shortage count, because both come from mistakes:

```latex
\text{ErrRate}_w = \frac{\sum_t \max\left(0,\ |\Delta_{w,t}| - \tau\right)}{\sum_t C_{w,t}}
```

Here Delta is the attributed discrepancy in week t and C the cash handled. The plugin adds a second rate, the voids and post-scan corrections per 100 transactions, called V\_w. Supervisor-requested voids and customer changes of mind are excluded when the plugin can tell them apart.

```latex
\text{AccuracyScore}_w = 50 - 15\left(0.6\, Z(\text{ErrRate}_w) + 0.4\, Z(V_w)\right)
```

The weights 0.6 and 0.4 are starting values to calibrate. Both rates use a rolling window of 12 weeks with a half-life of 4 weeks, so old mistakes fade, and the same shrinkage as for speed, so one bad week cannot define a person.

An unusually high speed combined with an unusually high rate of manual entries or voids is not punished automatically. It is flagged for a human to look at.

## Fair comparison

A score is only fair if two cashiers are compared on equal terms. Each distortion below is removed in the model rather than ignored.

| Distortion | Example | Fix in the engine |
| --- | --- | --- |
| Basket mix | One person gets big trolleys on a Saturday | Expected duration E\_i is computed per basket, so a big trolley is judged against a big trolley |
| Customer delays | Customer searching for their card or packing slowly | Median instead of mean, outlier removal, and thousands of transactions averaging it out |
| Lane type | Registers 1 to 7 are the busiest and 8 to 15 follow; register 1 (priority lane) and register 4 (card-only) see different customers | Separate model per lane type and register zone |
| Time of day | Customers rush and behave differently at 12:00 and 20:00 | Separate peak and off-peak scores |
| Few transactions | A new or part-time worker with 80 transactions | Shrinkage toward the average and a confidence level |
| Learning curve | First weeks of a new cashier | Provisional label for the first 4 weeks, compared with other new cashiers |
| Shared tills | Two workers use one till in a day | Cash discrepancy split by share of cash handled; shifts with an unclear owner are dropped |
| Seasonal effects | Christmas week, 28 December to 4 January, end of August | Expected duration refitted on a rolling window, so the whole store is compared with itself in the same period |

Seasonal busyness changes how many customers arrive, not how fast a cashier works, so it matters for the planner (Part 2) much more than for this score.

## Combined score and output

The engine produces one score per worker and per context c (peak or off-peak), because the planner asks a different question for a Saturday 11:00 slot than for a Tuesday 16:00 slot:

```latex
\text{Score}_w^{(c)} = \omega_S^{(c)}\,\text{SpeedScore}_w^{(c)} + \omega_A\,\text{AccuracyScore}_w, \qquad \omega_S^{(c)} + \omega_A = 1
```

Starting weights: speed 0.65 and accuracy 0.35 at peak, where throughput matters most; 0.5 and 0.5 off-peak. Reliability, versatility and manager ratings are added later as extra terms with their own weights, once their data exists.

The engine never hands the planner a bare rank. Each worker's record contains:

| Field | Meaning |
| --- | --- |
| score\_peak, score\_off | The two combined scores on the 0 to 100 scale |
| interval | A 90% range around the score; a wide range means little data |
| tier | A, B or C by percentile, shown only when the interval is narrow enough; otherwise unrated |
| n\_transactions, weeks\_of\_data | The sample behind the number |
| drivers | The top reasons, for example faster than expected on big baskets, high void rate |
| model\_version | Which formulas and weights produced it, for audit |

The planner can still sort workers from best to worst for rush slots. A worker marked unrated is placed as average and given more peak shifts to collect data, so newcomers are not buried.

## The score interface

`ree.api.provider.ScoreProvider` is the only coupling between the engine and the shift planner:

```python
class ScoreProvider(Protocol):
    def get_ranking(
        self, context: Context, as_of: datetime | None = None, min_confidence: float = 0.0
    ) -> list[RankedScore]: ...

    def get_score(self, worker_id: str) -> CashierScoreRecord | None: ...

    def list_scores(self) -> list[CashierScoreRecord]: ...

    def latest_as_of(self) -> datetime | None: ...
```

- `context` is `"peak"` or `"off"`: a Saturday 11:00 slot asks for the peak ranking, a Tuesday 16:00 slot for the off-peak one.
- `get_ranking` uses the latest scoring run at or before `as_of`, so a plan can be rebuilt as it would have been on a past date. Rank 1 is the best.
- Workers whose tier is `unrated`, or whose confidence is below `min_confidence`, are placed at the cohort average (`unrated = True`, `effective_score` is the average). They stay in the list so that a planner can give them peak slots and their scores gain data.
- Every record carries `interval`, `confidence`, `tier`, `provisional` and `review_flag`. A review flag is for a human to look at; it is never a reason to lower a worker's rank.
- The same contract is served over HTTP (`GET /scores/ranking?context=peak&min_confidence=0.5`, `GET /scores`, `GET /scores/{worker_id}`) so the planner can run as a separate service. The OpenAPI schema is committed in `docs/openapi.json`.

The planner is decision support on top of decision support: nothing in Part 1 or Part 2 should change a contract, a pay element or a disciplinary file without human review (see `docs/fairness.md`).

## Architecture

The plugin sends only timings and counts to a store kept inside the retailer's own environment. The engine runs overnight, adds the weekly cash sheet, and writes scores to a score store. The planner and a separate review view for managers and workers both read that store, and nothing else, so Part 1 can be replaced or improved without touching Part 2.

## Legal and fairness guardrails

These rules are built into the design from the first version. They are not legal advice, and a lawyer or data protection officer should confirm them before any real deployment.

- Workers are told in advance what is measured, and the works council (CSE) is informed and consulted before the tool goes live.
- The score is decision support: a manager validates every plan built from it, and any worker can see their own score, its drivers, and ask for a review.
- The engine stores pseudonymous worker IDs, counts and timings only; no customer data and no item identities.
- Sick leave, maternity and other protected absences never enter any score. Attendance, if used later, only counts unjustified absences.
- The model is tested for indirect bias, for example by checking that scores do not differ systematically by contract type, tenure or schedule, and the formulas and weights are documented for audit.
- A data protection impact assessment is prepared, and the EU AI Act status of workplace evaluation systems is checked against the dates in force at launch.

These same points are what a retailer's HR and legal teams will ask for, so they are part of the product, not a cost.

## Validation on simulated data

The method is validated on simulated data first: the simulator creates cashiers whose true speed and error habits are set in advance, and the engine is judged on whether it recovers that hidden order despite noisy baskets. Real data enters only after the legal and consultation steps above.

## Open questions

Settled so far:

| Question | Answer |
| --- | --- |
| Personal login | Yes, code and password, so every transaction has an owner |
| Weekly paper | Names both the register and the person |
| Big trolley | 40 items and above |
| Bagging | Done by the customer |
| Hand-keyed items | About 15% of items: fruit, vegetables, bread |
| Tagged items | Bottles and boxed items need the anti-theft tag removed, 5 to 7 seconds each; no tobacco is sold |
| Coupons | Reduction vouchers (bons de réduction) entered before payment |
| Special lanes | Register 1 is the priority lane, register 4 is card-only |
| Interruptions | Handled by the pause rule above, no manual input |
| Register position | Registers 1 to 7 are busiest, 8 to 15 next, because the store layout ends at register 1 |
| Self-checkouts (2) | Supervised only, not scored by this engine |
| Register software | Unknown, ignored for now |
| First prototype | A generated store with 25 simulated coworkers and a smoke test |

Still open:

- [ ] What is a normal discrepancy on the weekly paper, in euros? Ask a head cashier or manager; it sets the tolerance.
- [ ] Parked until the register software is known: which events does it log (supervisor call, cheque payment, price check, void), and can it export them? Until then the gap rule handles interruptions.

