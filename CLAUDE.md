# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run locally
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501

# Docker production deploy
./scripts/deploy.sh
./scripts/deploy.sh --ssl yourdomain.com

# Systemd deploy (no Docker)
./scripts/deploy.sh --no-docker

# Update running deployment (auto-detects Docker vs systemd)
./scripts/update.sh

# Logs
docker compose logs -f rdl          # Docker
sudo journalctl -u rdl -f           # systemd
```

There are no automated tests, linting, or CI pipelines in this project.

## Architecture

**Ryan's Data Lab (RDL)** is a Streamlit-based statistical analysis platform. All state lives in `st.session_state` (key `"df"` for the current dataframe, `"data_name"` for its label). There is no database.

**`app.py`** — Single entry point (~1,368 lines). Handles:
- Full CSS design system (custom properties, sidebar dark theme, card styles)
- Sidebar: data upload, sample dataset selector, module navigation radio buttons
- Module routing: imports each module's `render_*()` function and calls it inside a try/except error boundary
- Landing page with hero section and feature cards

**`modules/`** — 16 standalone analysis modules. Each exports a `render_<name>(df)` function called from app.py. Modules are stateless; they read/write `st.session_state` for data changes.

**`modules/ui_helpers.py`** — Shared UI components and the Plotly theme. Registers a `"plotly+rdl"` template at import time with a 10-color indigo-based palette. Provides: `section_header()`, `empty_state()`, `significance_result()`, `help_tip()`, `grouped_chart_selector()`.

**`scripts/`** — `deploy.sh` (Docker/systemd setup, nginx, SSL) and `update.sh` (pull, rebuild, restart, health check).

**`.streamlit/config.toml`** — Port 8501, 200MB upload limit, XSRF protection, primary color #6366f1.

## Conventions

- Use `section_header(title, help_text)` not `st.markdown("#### ...")`
- Use `empty_state(message, suggestion)` not `st.warning()` for missing-data states
- Use `significance_result()` for hypothesis test result cards
- All Plotly charts inherit from `"plotly+rdl"` template — never hardcode colors like "steelblue"
- Use `st.spinner()` for compute-heavy operations
- Use `@st.cache_data` strategically for expensive computations (model fits, sample datasets)
- Every module render is wrapped in try/except in app.py; modules should handle edge cases (division by zero, small groups, type conversion NaNs) gracefully with warnings rather than crashes
- Primary color: `#6366f1` (indigo). Font: Plus Jakarta Sans.

## Deployment

- Docker container runs as non-root `appuser`, 2GB memory / 2 CPU limit
- Nginx reverse proxy with WebSocket support and 300s timeouts for long-running analyses
- Health check endpoint: `/_stcore/health`
- SSH deploy shortcut: `ssh rdl "cd /opt/RDL && git pull origin main && docker compose up -d --build rdl"`
