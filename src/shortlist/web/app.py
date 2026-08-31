"""The demo server: an HTMX page and a JSON `/rank` endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from shortlist.data import DATASETS
from shortlist.embedding import HashingEmbedder, PrecomputedEmbedder
from shortlist.loaders import option_from_record, profile_from_dict
from shortlist.models import Result, Shortlist
from shortlist.pipeline import rank
from shortlist.rankers import DEFAULT_RANKERS
from shortlist.serialisation import to_dict
from shortlist.web.forms import DemoRequest, Values, parse, starting_values

HERE = Path(__file__).parent

app = FastAPI(title="shortlist", docs_url="/api")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

_hashing = HashingEmbedder()
_embedders: dict[str, PrecomputedEmbedder] = {
    key: PrecomputedEmbedder(dataset.vectors(), _hashing) for key, dataset in DATASETS.items()
}


class RankRequest(BaseModel):
    options: list[dict[str, Any]]
    profile: dict[str, Any]
    rankers: list[str] | None = None
    top: int | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rank")
def rank_endpoint(request: RankRequest) -> dict[str, Any]:
    """The same ranking the library produces, over the wire."""
    unknown = sorted(set(request.rankers or []) - set(DEFAULT_RANKERS))
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown rankers: {unknown}")
    shortlist = rank(
        [option_from_record(record) for record in request.options],
        profile_from_dict(request.profile),
        rankers=request.rankers,
        embedder=_hashing,
        top=request.top,
    )
    return to_dict(shortlist)


def _run(demo: DemoRequest) -> Shortlist:
    return rank(
        demo.dataset.options,
        demo.profile,
        rankers=demo.rankers,
        embedder=_embedders[demo.dataset.key],
    )


def _comparison(first: Result | None, second: Result | None) -> list[dict[str, Any]]:
    """Per ranker, how much each option gained over the other. Biggest gap first."""
    if first is None or second is None:
        return []
    left = {view.ranker: view for view in first.per_ranker}
    right = {view.ranker: view for view in second.per_ranker}
    rows = [
        {
            "ranker": name,
            "left": left[name],
            "right": right[name],
            "delta": left[name].contribution - right[name].contribution,
        }
        for name in left
        if name in right
    ]
    return sorted(rows, key=lambda row: abs(float(row["delta"])), reverse=True)


def _context(demo: DemoRequest) -> dict[str, Any]:
    shortlist = _run(demo)
    by_id = {result.option.id: result for result in shortlist.results}
    pair = [by_id.get(demo.compare[0]), by_id.get(demo.compare[1])]
    return {
        "datasets": list(DATASETS.values()),
        "dataset": demo.dataset,
        "demo": demo,
        "shortlist": shortlist,
        "all_rankers": DEFAULT_RANKERS,
        "active": demo.rankers,
        "focused": by_id.get(demo.focus),
        "compare": pair if all(pair) and pair[0] is not pair[1] else None,
        "comparison": _comparison(pair[0], pair[1]) if all(pair) and pair[0] is not pair[1] else [],
        "scores": [result.score for result in shortlist.results],
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request, dataset: str = "london-restaurants") -> HTMLResponse:
    chosen = DATASETS.get(dataset, DATASETS["london-restaurants"])
    demo = parse(starting_values(chosen))
    return templates.TemplateResponse(request, "index.html", _context(demo))


@app.post("/demo/rank", response_class=HTMLResponse)
async def demo_rank(request: Request, focus: str | None = None) -> HTMLResponse:
    form = await request.form()
    values: Values = {key: [str(value) for value in form.getlist(key)] for key in form}
    return templates.TemplateResponse(request, "_results.html", _context(parse(values, focus)))
