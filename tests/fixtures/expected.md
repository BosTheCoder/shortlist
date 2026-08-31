# Hand-computed expectation for `tests/fixtures/dining.json` + `dining.yaml`

Worked by hand so the end-to-end test asserts an independently derived answer
rather than whatever the code happens to produce.

## 1. Hard filter

`price_gbp_pp <= 60` removes **foxtrot** (90).
`tags contains_none [shellfish]` removes **golf**.

Survivors: alpha, bravo, charlie, delta, echo.

## 2. `constraint` — soft rules, min-max normalised over the survivors

price range 20..55 (spread 35), rating range 3.9..4.8 (spread 0.9).
`penalty = (1.0 * price_deviation + 1.5 * rating_deviation) / 2.5`

| id | price dev | rating dev | penalty | score |
|----|-----------|------------|---------|-------|
| alpha   | 5/35 = 0.142857 | 1 - 0.9/0.9 = 0     | 0.142857/2.5 = 0.057143 | -0.057143 |
| bravo   | 10/35 = 0.285714 | 1 - 0.7/0.9 = 0.222222 | 0.619047/2.5 = 0.247619 | -0.247619 |
| charlie | 24/35 = 0.685714 | 1 - 0.6/0.9 = 0.333333 | 1.185714/2.5 = 0.474286 | -0.474286 |
| delta   | 35/35 = 1.0      | 1 - 0.3/0.9 = 0.666667 | 2.0/2.5 = 0.8           | -0.8      |
| echo    | 0                | 1                      | 1.5/2.5 = 0.6           | -0.6      |

Order: **alpha, bravo, charlie, echo, delta**

## 3. `lexical` — BM25, k1 = 1.5, b = 0.75, over `title + text`

Query terms: quiet, vegetarian, dinner. Each appears in 3 of the 5 survivors, so
all three share `idf = ln(1 + (5 - 3 + 0.5) / (3 + 0.5)) = ln(1.714286) = 0.539130`.

Document lengths (tokens): alpha 11, bravo 10, charlie 6, delta 6, echo 7. Sum 40, avgdl 8.0.
`L = 1.5 * (0.25 + 0.75 * dl / 8)`; every matching term has tf = 1, so each contributes
`idf * 2.5 / (1 + L)`.

| id | dl | L | 2.5/(1+L) | matching terms | score |
|----|----|---|-----------|----------------|-------|
| alpha   | 11 | 1.921875 | 0.855615 | 3 | 3 * 0.461290 = 1.383870 |
| bravo   | 10 | 1.781250 | 0.898876 | 3 | 3 * 0.484606 = 1.453818 |
| charlie | 6  | 1.218750 | 1.126761 | 2 | 2 * 0.607468 = 1.214936 |
| delta   | 6  | 1.218750 | 1.126761 | 0 | 0 |
| echo    | 7  | 1.359375 | 1.059603 | 1 | 0.571262 |

Order: **bravo, alpha, charlie, echo, delta**

## 4. `popularity` — `(v*R + m*C) / (v + m)`, m = 25

C = mean rating over survivors = (4.8 + 4.6 + 4.5 + 4.2 + 3.9) / 5 = 4.4

| id | v | R | shrunk |
|----|---|---|--------|
| alpha   | 600 | 4.8 | (2880 + 110) / 625 = 4.784000 |
| bravo   | 400 | 4.6 | (1840 + 110) / 425 = 4.588235 |
| charlie | 30  | 4.5 | (135 + 110) / 55   = 4.454545 |
| delta   | 900 | 4.2 | (3780 + 110) / 925 = 4.205405 |
| echo    | 120 | 3.9 | (468 + 110) / 145  = 3.986207 |

Order: **alpha, bravo, charlie, delta, echo**

`semantic` abstains (NullEmbedder) and `learned` abstains (no fitted weights),
so three rankers reach the fusion.

## 5. RRF, k = 60, all weights 1.0

| id | constraint | lexical | popularity | score |
|----|-----------|---------|------------|-------|
| alpha   | 1 | 2 | 1 | 1/61 + 1/62 + 1/61 = 0.04891591 |
| bravo   | 2 | 1 | 2 | 1/62 + 1/61 + 1/62 = 0.04865150 |
| charlie | 3 | 3 | 3 | 3/63             = 0.04761905 |
| echo    | 4 | 4 | 5 | 1/64 + 1/64 + 1/65 = 0.04663462 |
| delta   | 5 | 5 | 4 | 1/65 + 1/65 + 1/64 = 0.04639424 |

**Final: alpha, bravo, charlie, echo, delta. Top 3 = alpha, bravo, charlie.**
