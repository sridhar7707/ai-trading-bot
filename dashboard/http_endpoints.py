"""FastAPI extra routes attached to the Gradio app object."""
from __future__ import annotations

import threading
from pathlib import Path

import gradio as gr
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from gradio.routes import App as _GradioApp
from starlette.routing import Mount


def build_app(demo: gr.Blocks) -> _GradioApp:
    """Wrap the Gradio Blocks demo in a FastAPI app, add custom routes, and return the app.

    Gradio 5.9.0 does not register a handler for /_app/immutable/* — those requests
    fall through to FastAPI's default 404. Fix: insert a StaticFiles Mount at index 0
    of the router (checked before every other route/mount) so all /_app/* requests are
    served from Gradio's compiled SvelteKit frontend package.

    NOTE: do NOT mount /static — Gradio registers APIRoute /static/{path:path}
    (static_resource) to serve its own assets. An overlapping StaticFiles mount at
    index 0 would shadow that route and break Gradio's static resource serving. The
    /static/fonts/*.woff2 404s from Gradio 5's CSS are benign — those are OS-level
    system fonts (ui-sans-serif, system-ui) that browsers provide natively.
    """
    app = _GradioApp.create_app(demo, app_kwargs={"docs_url": None, "redoc_url": None})

    _GR_APP_DIR = Path(gr.__file__).parent / "templates" / "frontend" / "_app"
    if _GR_APP_DIR.is_dir():
        app.router.routes.insert(0, Mount("/_app", app=StaticFiles(directory=str(_GR_APP_DIR))))

    @app.get("/run/cron")
    async def _cron_endpoint():
        from scheduler.dispatcher import main as _dispatch
        threading.Thread(target=_dispatch, daemon=True, name="cron-dispatcher").start()
        return JSONResponse({"status": "accepted"})

    @app.get("/debug/charts")
    async def _debug_charts():
        """Call each chart render function and report success/failure + timing."""
        import time as _t
        import traceback as _tb
        from dashboard.charts import (
            render_equity_chart, render_allocation_chart, render_pnl_chart,
            render_feature_importance_chart, render_returns_histogram, render_winloss_chart,
        )
        from dashboard.components.capital import render_capital_chart
        from dashboard.components.market_mood import render_market_mood
        from dashboard.components.news import render_news_feed

        fns = [
            ("equity",       render_equity_chart),
            ("allocation",   render_allocation_chart),
            ("pnl",          render_pnl_chart),
            ("capital",      render_capital_chart),
            ("returns_hist", render_returns_histogram),
            ("winloss",      render_winloss_chart),
            ("feat_imp",     render_feature_importance_chart),
            ("market_mood",  render_market_mood),
            ("news",         render_news_feed),
        ]
        results = {}
        for name, fn in fns:
            t0 = _t.time()
            try:
                out = fn()
                results[name] = {"ok": True, "type": type(out).__name__, "ms": round((_t.time() - t0) * 1000)}
            except Exception:
                results[name] = {"ok": False, "error": _tb.format_exc(), "ms": round((_t.time() - t0) * 1000)}
        return JSONResponse(results)

    return app
