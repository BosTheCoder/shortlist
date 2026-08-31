# shortlist

Rank a set of options against a written preference profile, using several independent
signals fused into one ranking you can pull apart afterwards.

You have 40 candidate restaurants, backpacks or hotels and a set of things you care
about. Some are absolute ("under £60 a head", "nothing with shellfish") and some are
preferences ("closer is better", "well reviewed matters more than cheap"). Sorting by
one column throws away everything else. `shortlist` runs several rankers over the same
candidates, fuses their orderings, and tells you where each one placed every option and
which of them disagreed.

**Live demo:** https://bos-shortlist.fly.dev

## 60 seconds

```bash
git clone https://github.com/BosTheCoder/shortlist
cd shortlist
uv sync

uv run shortlist rank src/shortlist/data/london-restaurants.json \
  --profile profiles/dinner.yaml --top 5 --explain
```

```
                                   dinner  (e2ba88aca178)
┏━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ # ┃ option              ┃   score ┃ spread ┃ constraint ┃ lexical ┃ semantic ┃ popularity ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1 │ Ottermill Kitchen   │ 0.06337 │   2.86 │          3 │       1 │        1 │          8 │
│ 2 │ Quill & Fern        │ 0.06240 │   3.11 │          1 │       9 │        2 │          5 │
│ 3 │ Green Lantern Rooms │ 0.06108 │   4.21 │         12 │       3 │        7 │          1 │
│ 4 │ Vellum Cafe         │ 0.06046 │   4.92 │          4 │       4 │        3 │         15 │
│ 5 │ Harlow Fish Bar     │ 0.06006 │   3.11 │         10 │       2 │        6 │          9 │
└───┴─────────────────────┴─────────┴────────┴────────────┴─────────┴──────────┴────────────┘
abstained: learned
8 filtered out by hard rules
```

The last four columns are the interesting part. Green Lantern Rooms is the most popular
place on the list and the twelfth best fit for the stated constraints; that gap, not the
fused score, is the thing worth looking at.

`just demo` serves the same thing as a web page on http://localhost:8080.

## The profile

One YAML file, written once and reused.

```yaml
name: dinner
query: "relaxed dinner spot, good vegetarian and fish, not too loud"

hard:
  - field: price_gbp_pp
    max: 60
  - field: cuisine_tags
    contains_none: [shellfish]

soft:
  - field: price_gbp_pp
    prefer: low
    weight: 1.0
  - field: rating
    prefer: high
    weight: 1.5
```

Hard rules are absolute. They are applied before any ranker sees the options, so no
ranker weighting and no combination of switches can bring a violating option back.
A missing field fails a hard rule rather than passing it quietly.

Soft rules are min-max normalised across whatever survived the hard rules, so a penalty
means "how far from the best option here", not "how big is this number".

## The rankers

| Ranker | Signal | Abstains when |
|---|---|---|
| `constraint` | Weighted soft rules, normalised across the candidates | the profile has no soft rules |
| `lexical` | BM25 of the profile query against the option text | there is no query |
| `semantic` | Cosine of embedded query against embedded text | the embedder returns nothing |
| `popularity` | Bayesian-shrunk rating, `(v·R + m·C) / (v + m)` | no option carries a rating |
| `learned` | Pairwise logistic model over past picks | nothing has been trained yet |

Abstaining is not the same as scoring everything equally. An abstaining ranker is
dropped from the fusion, so it cannot flatten the result by voting for everything.

### Fusion

Reciprocal Rank Fusion: `score = Σ wᵢ / (k + rankᵢ)`, with `k = 60`.

The rankers produce a BM25 score, a cosine, a shrunk star rating and a log-odds. Those
are not on comparable scales and averaging them would just be a statement about their
units. RRF throws the magnitudes away and fuses the positions.

Each result also carries a **spread**: the standard deviation of the ranks the rankers
gave it. A high spread means the signals disagreed, which is usually the option worth
looking at yourself.

### Adding a ranker

A ranker is a class with a name and a `rank` method, plus one line in the registry.
As a worked example, an LLM judge (this is not built, and deliberately so, because it
would make the hosted demo need a key and a network round trip):

```python
# src/shortlist/rankers/judge.py
class JudgeRanker:
    name = "judge"

    def __init__(self, client: Judge, prompt: str) -> None:
        self.client = client
        self.prompt = prompt

    def rank(self, options, profile) -> list[ScoredOption] | None:
        if not profile.query.strip():
            return None
        verdicts = self.client.score(self.prompt, profile.query, options)
        return [ScoredOption(option_id=o.id, score=verdicts[o.id]) for o in options]
```

```python
# src/shortlist/rankers/__init__.py
REGISTRY["judge"] = lambda judge=None, **_: JudgeRanker(judge, JUDGE_PROMPT)
```

Nothing else changes. Fusion, the explanation panel, the CLI `--only` flag and the
demo checkboxes all pick it up from the registry.

## Learning from what you actually chose

```bash
uv run shortlist rank options.json --profile profiles/dinner.yaml   # prints a run id
uv run shortlist feedback e2ba88aca178 --picked ottermill-kitchen
uv run shortlist train profiles/dinner.yaml                          # writes dinner.weights.json
```

Every pick becomes one training row per option it was chosen over: the feature delta
between the two. Those rows fit a logistic model by gradient descent, and the `learned`
ranker scores options with it on the next run. With no feedback recorded, it fits
nothing and abstains, so a cold profile is never guessed at.

This is a small personalisation loop, honestly. It learns weights over the numeric
fields you already have; it does not learn features.

## Using it as a library

```python
from shortlist import HashingEmbedder, load_options, load_profile, rank

options = load_options("options.json")  # JSON, CSV or YAML
profile = load_profile("profiles/dinner.yaml")
result = rank(options, profile, embedder=HashingEmbedder(), top=5)

for item in result.results:
    places = {view.ranker: view.rank for view in item.per_ranker}
    print(item.rank, item.option.title, round(item.score, 5), places)
```

Or over HTTP:

```bash
curl -s localhost:8080/rank -H 'content-type: application/json' -d '{
  "options": [{"id": "a", "title": "A", "text": "quiet vegetarian", "price": 20, "rating": 4.6, "review_count": 300}],
  "profile": {"name": "d", "query": "quiet vegetarian"}
}'
```

`POST /rank` returns exactly what the library returns, and there is a test that fails
if the two ever drift apart.

## Embedders

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- `NullEmbedder` returns zeros, which makes the semantic ranker abstain.
- `HashingEmbedder` is a signed hashing projection with sublinear term frequency. It
  needs no model, no key and no network, which is what lets the hosted demo embed a
  query typed a second ago. Two texts are close when they share vocabulary, not when
  they share meaning; it is not a learned model and does not pretend to be.
- `OpenAICompatibleEmbedder` posts to any `/embeddings` endpoint that speaks the
  OpenAI shape. Set `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY`.

The bundled datasets ship their option vectors in a `.npz`, regenerated by `just embed`,
so a demo request only embeds the query. A test fails if a dataset is edited without
rerunning it.

## Measured

On one core of a WSL2 laptop, Python 3.12, median of repeated runs:

| Workload | Rankers | Median | p95 |
|---|---|---|---|
| 24 options, precomputed vectors (the demo path) | 4 | 0.97 ms | 1.18 ms |
| 5,000 options, embedded on the spot | 4 | 269 ms | |

Reproduce these yourself; there is no benchmark harness hiding the method. The 5,000
figure was 1,950 ms before the ranker positions were put in a dict instead of being
found with `list.index` per option per ranker.

Size: 1,495 lines of source, largest file 128 lines, 69 tests.

## The demo

Three bundled datasets: London restaurants, travel backpacks, city-break hotels. Every
record is invented. There are no real businesses, people or addresses in them.

Tick a ranker off and the table re-sorts. Click a row to see which hard rules it passed
and where each ranker put it. Pick any two options to see, ranker by ranker, what moved
them apart.

The demo sleeps when nobody is using it, so the first load can take a few seconds. The
page says so, server-rendered, before anything else has to happen.

## Development

```
just check      # ruff, pyright, pytest
just test       # pytest
just fmt        # ruff format and fix
just demo       # serve on localhost:8080
just embed      # regenerate the bundled vectors
just deploy     # flyctl deploy
```

The tests are integration-first. The end-to-end expectation is hand-computed in
[`tests/fixtures/expected.md`](tests/fixtures/expected.md) so the suite asserts an
independently derived answer rather than whatever the code happened to produce. There
are no tests asserting on wording; break the ranking and the suite fails, rename a
column heading and it does not.

## Licence

MIT. See [LICENSE](LICENSE).
