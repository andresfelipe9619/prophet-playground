# Deployment and the phone layout

The dashboard is the end-user product, and it is read on a phone. This page
covers the two halves of making that true: the layout rules that make a 375px
screen a first-class target, and getting the app onto a URL.

Both are about the same object — `dashboard/app.py`, a **long-running Streamlit
server**. That one fact decides every hosting question below, so it is worth
stating before anything else: Streamlit is not a request/response app. It serves
a JavaScript bundle, holds a **websocket open for each viewer**, re-runs the
script top to bottom on every interaction, and keeps that viewer's session state
in the server process's memory between runs. A host that can run a process and
keep it running can host it. A host built around per-request functions cannot,
whatever language it speaks.

---

## 1. The phone layout

Everything lives in [`dashboard/mobile.py`](../dashboard/mobile.py), installed
once by the shell before any page renders. It is one module for the same reason
all the explanatory copy is one module: three page files each patching their own
CSS cannot be reviewed as a whole, and "no surface is unreadable on a phone" is
only checkable when there is one surface to check.

### What it does

| Rule | Why |
| --- | --- |
| Columns collapse to full width **below 640px** | A two-column comparison at 375px is two unreadable halves; stacked, it reads top to bottom. |
| ...**except** columns holding a metric, which go two-up | `st.columns(4)` of metric tiles is the most common row here. One-up turns four short numbers into four screens of scrolling. |
| Metric labels wrap instead of truncating | The labels here are sentences ("Premio mayor necesario para que…"), not words. |
| The tab strip scrolls horizontally, scrollbar hidden | Baloto has ten tabs. They were never going to fit. |
| Every text input is 16px | Below 16px iOS zooms the page on focus and does not zoom back out. |
| Buttons go full width | Thumb-sized targets. |
| The sidebar resize handle is hidden | It is a mouse gesture, and on touch it is only an 8px target next to the sidebar's edge. |
| Plotly legends move from the right-hand side to above the plot | This is the big one — see below. |
| Plotly drag and scroll-zoom are off, mode bar hidden | On a touch screen, the gestures that drive a chart are the gestures that scroll the page. The page has to win. |

Every CSS rule is **inside a `max-width: 640px` media query**, so a rule that
turns out to be wrong is wrong only on phones. The desktop layout is untouched.

### The legend is the whole chart

Plotly puts a legend in a vertical column on the right. On a 390px screen that
leaves the plot itself about two thirds of an already narrow width, which is the
single largest reason these charts were unreadable on a phone. `mobile.responsive`
moves it above the plot, where it costs one line of height instead.

That function is applied by [`ui.chart`](../dashboard/ui.py), not by the call
sites. No page module changed, and a chart added tomorrow cannot miss it — the
same reasoning that puts `PLAIN` and `READ` behind `section()` and `chart()`
rather than at each call site.

### Two things a phone layout cannot fix

The **sidebar is behind the ☰ button** on a phone, and every data control lives
there — the domain selector, the CSV upload, the format checkbox. `app.py` prints
a phone-only line naming the active domain and pointing at that button, because
a reader who does not know the controls exist will conclude the app does not have
any. The line is hidden on desktop, where the sidebar is visible and it would be
noise.

**Wide dataframes still scroll sideways inside their own box.** That is the
correct behaviour — the alternative is a page that scrolls sideways, which puts
every other column out of reach — but a 20-column backtest table is a table you
read by scrolling, on any phone.

### Verifying a layout change

Launch Streamlit headless and drive it with Playwright at a phone viewport
(`p.devices["iPhone 13"]`); Chromium is under `/opt/pw-browsers/`. An HTTP 200 on
`/` proves nothing, because Streamlit only executes the script once a client
connects over the websocket. Three things are worth asserting, and all three
caught a real bug while this page was being written:

1. **No page-level horizontal overflow**: `document.documentElement.scrollWidth`
   must equal `window.innerWidth`.
2. **The ☰ button is actually tappable**: `elementFromPoint` at its centre must
   return the button, not something on top of it. An earlier draft of
   `mobile.py` set the sidebar to `85vw`; Streamlit collapses the sidebar by
   translating it a **fixed number of pixels**, not by its own width, so widening
   it parked a 31px strip of sidebar directly over the button — with every
   control on the phone behind it. The layout looked fine in a screenshot.
3. **No tab renders an error**: all tab panels stay mounted, so scope locators
   to `get_by_role("tabpanel", name=...)` and check the text for `Traceback` and
   "This app has encountered an error".

---

## 2. Hosting: what works and what does not

### Vercel does not host this

Vercel deploys Python — it has a Python runtime, and it now supports WebSockets
in Functions — but neither of those makes it a host for Streamlit:

* Its Python runtime expects you to **export an ASGI or WSGI `app` object** that
  it invokes per request. Streamlit has no such object. It is a CLI that starts
  its own Tornado server and owns the process; there is nothing to hand over.
* Functions are **ephemeral and duration-capped**. Streamlit keeps each viewer's
  session state in process memory across the life of a websocket, and a backtest
  here runs for tens of seconds. Both assume a process that outlives a request.

This is not a Python limitation. A FastAPI or Flask API from this repo would
deploy to Vercel normally. It is a Streamlit-shaped limitation, and the same one
applies to Netlify, Cloudflare Pages and Lambda-style hosting generally.

If Vercel specifically is a requirement, the shape that works is a split: a
FastAPI service on Vercel exposing the model endpoints, and a separately hosted
front end. That is a rewrite of the dashboard, not a deployment of it.

### What does work

| Host | Fit | Cost |
| --- | --- | --- |
| **Streamlit Community Cloud** | Purpose-built: point it at the GitHub repo and the entrypoint, no config | Free; sleeps when idle, wakes on a visit |
| **Render / Railway** | `Dockerfile` in this repo, `$PORT` already wired | Free tier sleeps; a small paid instance does not |
| **Fly.io / Cloud Run** | Same Dockerfile; Cloud Run scales to zero | Pay per use |
| **Hugging Face Spaces** | Native Streamlit SDK | Free |

**Streamlit Community Cloud is the recommendation**, and not only because it is
free: it is the only one that needs no infrastructure work at all, and the app
sleeping after a period of no visitors is exactly right for something one person
opens on a phone a few times a week.

### Deploying to Streamlit Community Cloud

1. Push this branch, sign in at `share.streamlit.io` with GitHub.
2. New app → this repo → entrypoint `dashboard/app.py`.
3. That is the whole thing. It reads
   [`.streamlit/config.toml`](../.streamlit/config.toml) and
   [`dashboard/requirements.txt`](../dashboard/requirements.txt) from the repo.

Community Cloud looks for a dependency file **in the entrypoint's directory
first, then the repository root**, which is why `dashboard/requirements.txt`
exists: it is one line pointing at `requirements-deploy.txt`, and it exists
precisely so the root `requirements.txt` — which installs Prophet and a compiler
toolchain — is *not* what gets installed. Without it a free-tier build spends
most of its time compiling a Stan model for an option that ships switched off.

Community Cloud instances are memory-capped (roughly 1–2.7 GB, and the figure
changes without notice). statsforecast and xgboost fit; the thing most likely to
exceed it is a backtest over a very long history with every model selected.

### Deploying with the Dockerfile

```bash
docker build -t baloto-dashboard .
docker run -p 8501:8501 baloto-dashboard      # then http://localhost:8501
```

On Render: New → Web Service → this repo → Docker. Railway and Fly.io detect the
Dockerfile the same way. `$PORT` is already read in the `CMD`, and the bind
address is `0.0.0.0` — bound to localhost, the container answers only itself,
which is the classic "deploy succeeded, URL times out" failure.

---

## 3. What the deployed app runs on

**No real data ships.** `exported_data/` is gitignored and stays that way — a
deployed instance falls back to the synthetic generator in
`lottery/utils/sample_data.py` and **says so on screen**. That is the correct
default: a public URL serving synthetic draws labelled as synthetic is honest,
and one that silently served a scraped history would not be.

To look at real data on the phone, use the sidebar's CSV upload. It is per
session and nothing is stored server-side, which is also why the hosted instance
cannot become a shared dataset by accident. Committing a real
`exported_data/final-final.csv` would work and is a deliberate decision to
publish that data, not a deployment step.

**Prophet is not installed.** [`requirements-deploy.txt`](../requirements-deploy.txt)
leaves out prophet, cmdstanpy, matplotlib and seaborn: the first two are most of
the build for a model that ships switched off, and the last two are used only by
`scripts/summary_charts.py`, which the dashboard superseded.
`dashboard/baloto_page.py:PROPHET_AVAILABLE` checks with `importlib.util.find_spec`
(which does not import it), drops the option from the model selector, disables
the backtest checkbox, and explains why. The alternative — leaving the option
there to raise an `ImportError` from inside a spinner after someone taps it — is
the failure mode this project avoids everywhere else.

Add `prophet>=1.1.6` to `requirements-deploy.txt` to get it back. Expect the
build to take several times longer.

**There are three requirements files** and they answer three different questions:

| File | Question |
| --- | --- |
| `requirements.txt` | Everything: dashboard, scripts, Prophet. The local dev install. |
| `requirements-test.txt` | What `pytest` imports. What CI installs. |
| `requirements-deploy.txt` | What `dashboard/app.py` reaches at runtime. |

**The registry writes to disk.** `lottery/analysis/registry.py` appends to
`predictions.csv` at the repository root. On every host here that filesystem is
ephemeral — a restart or redeploy loses anything recorded through the deployed
app. Pre-registration works by being **committed to version control**, which is
what dates a prediction; a row written into a container that will be discarded
proves nothing. Record predictions from a local checkout and commit the file.
