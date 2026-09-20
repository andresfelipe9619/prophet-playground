# Local setup and the phone layout

**This project runs on your machine. There is no deployment and no hosted
instance**, and that is a design decision rather than an unfinished one — it is
what lets every model here run at full strength, with your real data, on your own
hardware, under no memory cap.

The dashboard is still read on a phone: you run the server on your laptop and
open it from your phone over your own network ([§3](#3-reading-it-on-your-phone)).
So the layout work below is unchanged by any of this — it was never about hosting.

What running locally buys, concretely:

| | Locally | On a free hosted tier |
| --- | --- | --- |
| TimesFM checkpoint | **3.0**, the strongest | 2.5 — a public URL is production, and 3.0's weights are not licensed for that |
| TimesFM context | the **full history** (2048 draws) | cut to 512 to stay inside the memory cap |
| GPU | used when present | none exists |
| Memory | whatever the machine has | ~1–2.7 GB, which torch alone does not fit in |
| Your real draws | read from disk | re-uploaded every session, because the CSV is gitignored |

If you ever do want it on a URL, the short version is that Streamlit is a
long-running server holding a websocket per viewer, not a request/response app —
so Vercel, Netlify and Lambda-style hosts cannot run it, while anything that can
keep a process alive (Render, Railway, Fly, a VPS) can. That is the whole of what
this page used to say about hosting, and `git log` has the rest.

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

## 2. Installing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

One list, because there is one place this runs. A second list that *can* differ
from what the app needs eventually does — that happened here, and a hosted build
silently offered five models where the real one offers six.
`requirements-test.txt` stays separate because it answers a genuinely different
question: what `pytest` imports, read by CI rather than by the app.

**If you have no NVIDIA GPU, install the CPU wheel of torch.** Plain
`pip install` resolves to the CUDA build and pulls ~3.2 GB of NVIDIA libraries on
top of torch's own 1.2 GB — measured, on a machine with no GPU at all:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

TimesFM's weights are a separate download, fetched from HuggingFace the first
time you use the model and cached under `~/.cache/huggingface`. Everything else
installs from PyPI and needs no account, no API key and no network at runtime.

### Your data

`exported_data/final-final.csv` is **gitignored and stays that way** — it is
yours, and nothing here publishes it. With the file present the dashboard reads
it directly and the sidebar upload becomes unnecessary; without it, every page
falls back to its synthetic generator and says so on screen.

```bash
python -m lottery.utils.scraper --years 2024 --dry-run   # eyeball what it parses
python -m lottery.utils.scraper --years 2020-2025        # write the CSV
```

**Pre-registration now works properly.** `lottery/analysis/registry.py` appends
to `predictions.csv` at the repository root, and a local checkout is the only
place that means anything: the file persists, and committing it is what dates a
prediction in version control. On an ephemeral host the same row was written into
a container that would be discarded, which proves nothing. See
[Registry](registry.md).

## 3. Reading it on your phone

The layout in §1 is for a real phone, and you do not need a hosted URL to get one
— you need the two machines on the same network:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

Streamlit prints a **Network URL** (`http://192.168.x.x:8501`). Open that on your
phone. The laptop keeps the data, the models and the GPU; the phone is only a
screen.

Two things worth knowing before you do:

* `.streamlit/config.toml` sets `headless = false` so a normal local run opens
  your browser. That is the right default for working on it and irrelevant to the
  LAN case, where you open the app by hand on the phone.
* This binds the server to every interface on your machine. On your own home
  network that is fine; on a café's wifi, anyone on that network can open it.
  There is no authentication in front of it, because there was never meant to be
  a stranger on the other end.

## 4. What the models do with a local machine

**Every model is installed. TimesFM is the one that is not on by default**, and
the reason is worth reading before you change it back.

The rule this project fixed once for Prophet is that a comparison table must not
*quietly* leave out an installed model. An unticked box with a sentence under it
saying why is not quiet — it is the reader's choice, made before the cost — and
the Bonferroni correction counts only the models that actually ran, so the table
stays honest either way.

What TimesFM ticking itself produced was an ambush. A reader clicking **Ejecutar
backtest** to compare the other five models paid a forward pass per window and,
on a first run, a checkpoint download of hundreds of megabytes. On a machine
without a GPU that is several minutes with nothing on screen, which is
indistinguishable from a hung app — and it is what the report "running the
Google model is slow and then the dashboard says it disconnected" sits on top
of.

Two related fixes went in with it. The Pronóstico tab now warns **before** the
wait rather than captioning it under the model picker, its spinner no longer
says "Entrenando…" for the one model here that fits nothing, and the result is
held in session state — so a forecast that took minutes survives the next widget
click, and a dropped connection costs you the wait once rather than twice.

**On the disconnection itself: it could not be reproduced on a direct localhost
connection, and that is worth recording.** A Streamlit script that blocks its
thread for three minutes — pure Python holding the GIL, and separately torch
saturating every core — kept its websocket open throughout, measured. Streamlit
already pings every 30 seconds by default (`server.websocketPingInterval`, whose
default is 30 with a matching timeout), so the "silent socket times out" theory
does not survive contact with the code. What remains, and what the fixes above
are aimed at, is that the run is long enough for *something* between the browser
and the server to give up, or for the process itself to be killed. Setting
`websocketPingInterval` lower is **not** a fix to reach for blind: it sets the
ping *timeout* to the same value, so a shorter interval tightens the deadline
the server itself enforces.

**TimesFM runs at full strength**, which is what changed when hosting stopped
being a consideration:

| | Now | Before, and why |
| --- | --- | --- |
| Checkpoint | **3.0** (`timesfm_model.CHECKPOINT`) | 2.5, because a public dashboard is production and 3.0's weights are not licensed for production |
| Context | **2048 draws** — a full 2010-2026 history is ~1035, so nothing is truncated | 512, to stay inside a free tier's memory |
| Device | **GPU when present** (`best_device()`: cuda, then Apple `mps`, then cpu) | cpu, because a free tier has no GPU |
| Installed | yes, in `requirements.txt` | a separate opt-in file, to keep the hosted image under the cap |

**The licence still matters, and it is the one restriction local use does not
remove.** Upstream distributes TimesFM weights **up to 2.5 under Apache-2.0** and
**3.0's under `timesfm-non-commercial-license-v1.0`, restricted to
non-commercial, non-production use**. Personal research on your own machine is
squarely inside that, which is why 3.0 is the default now. If this ever goes
commercial or back onto a public URL, switch one constant:

```python
CHECKPOINT = CHECKPOINT_APACHE      # lottery/models/timesfm_model.py
```

The `timesfm` package itself is Apache-2.0, and nothing in this repository
redistributes any weights — they are downloaded from Google's HuggingFace repo by
you, under whichever licence that repo states.

Everything else here is permissively licensed and costs nothing: Prophet (MIT),
statsforecast, xgboost, streamlit (Apache-2.0), scikit-learn, pandas (BSD-3),
plotly (MIT). No model calls out to an API, so nothing you run leaves your
machine.
