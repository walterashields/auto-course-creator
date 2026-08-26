# Pre-Render QA Checklist

This is the checklist a reviewer — human or automated — runs against an
actual rendered lesson video before it's considered done. It exists because
this project's own history shows that automated checks alone were
repeatedly insufficient: several of the items below were real, shipped
defects that a green "N/N events succeeded" run did not catch. Be honest
about that when running this list — passing the automated half is not the
same as the lesson being done.

Every item is phrased as a yes/no check against the **rendered output**
(the actual mp4, its audio track, its audit log), not against the code or
the lesson script. "Is `pre_actions` implemented?" is not a QA question;
"does the typed filter value actually appear on screen, highlighted, for
as long as narration discusses it?" is.

Each item traces back to a specific, real problem found and fixed on this
project, in the order found. That history is worth keeping attached to
each item — it's the evidence that the check is worth running, not a
hypothetical.

## How to run this

0. **Before rendering anything**, run item 10 — the one pre-render item
   on this list. Everything else below assumes a rendered video already
   exists to check; item 10 exists specifically to stop a lesson with a
   real data-correctness defect from ever getting that far.
1. Render the lesson (`automation/metabase_driver.py`, then
   `narration/audit_narrator.py`).
2. Run every item marked **Automated** below against the output files —
   these are cheap and fast, run them every time, no excuses.
3. Watch the actual rendered video for every item marked **Human**. There
   is no substitute step for this. A passing automated run does not imply
   these pass.
4. Only call the lesson done when every item passes. An item that fails
   is a real defect, found the same way the original eight were: by
   actually checking, not by assuming.

---

### 1. No dead, unnarrated setup/login time in the recording

**Check:** Does the recording open already on the first real instructional
screen — no visible login form, setup wizard, or blank loading state
before anything narrated begins?

**Human.** There's no reliable automated proxy for "does the opening feel
dead" — a login screen that renders in 0.3s and one that hangs for 8s both
just look like "some frames" to a script. Watch the first ~5 seconds.

**History:** the first cut of `video_1_1` had 20+ seconds of dead
login-screen time before any teaching started. Fixed by moving
setup/login into an unrecorded browser context whose authenticated
session is carried into the recorded one (see
`automation/metabase_driver.py`'s `run_lesson`).

---

### 2. Every notable action carries narration, or silence is a deliberate, documented choice

**Check:** Does every event that does something the learner needs to
understand have narration — and for every event that's silent, is there a
comment in the lesson script explaining why it's genuinely administrative
(nothing to teach), not just an oversight?

**Both.** Automated: count narrated vs. total non-pause events in the
lesson script and flag any silent event lacking an adjacent comment —
cheap, catches unexamined gaps, but can't judge whether a *given* silence
is actually defensible. Human: read the silent events' surrounding
context and judge whether "administrative, nothing to teach" actually
holds for each one.

**History:** the first cut only narrated `highlight_section`/`show_result`
events. Every other action (`click_new_question`, `select_table`,
`add_filter`, `save_question`, `add_to_dashboard`) was silent. Fixed by
adding a documented two-tier narration convention (short "here's what I'm
doing" lines on action events, longer "why this matters" lines on
highlight/result events) — see `AGENTS.md`.

---

### 3. Every element about to be interacted with is visibly highlighted first

**Check:** For every click in the audit log, was there a highlight overlay
visibly drawn on the correct element before that click fired?

**Both.** Automated: cross-reference the audit log — every commit-type
event (`click_new_question`, `select_table`, `add_filter`, `click_option`,
`save_question`, `add_to_dashboard`) should be preceded, positionally, by
a `highlight_target`/`highlight_targets` event targeting the same element.
Human: pull a frame at each highlight event's timestamp and confirm the
overlay is drawn on the actual right element, not just that *a* highlight
exists somewhere.

**History:** the first cut had no visual highlight at all — narration
described an action while the screen gave no indication of where it was
happening. Fixed with pixel-accurate overlays drawn from Playwright's
`bounding_box()` (`automation/metabase_driver.py`'s `_draw_highlight_box`).

---

### 4. The click fires only after the highlight, hold, and narration have played — never simultaneously

**Check:** For every highlighted action, does the actual click happen only
*after* its narrated pause completes — highlight → hold → narration →
click → hold → clear — not at the same time narration starts?

**Automated.** Fully mechanical: in the audit log, the commit event's
`started_at_ms` must be greater than or equal to the preceding pause
event's `completed_at_ms`. If it's earlier, the action fired while
narration was still describing it, or before it started.

**History:** the first re-cut had the highlight and the click firing in
the same event, so the click and its narration were simultaneous rather
than sequenced. Fixed by splitting every notable action into a
`highlight_target` event (locates, draws the overlay, carries narration)
and a separate commit event (performs the click only after the pause).

---

### 5. Audio is present, audible, and spans the actual full recording — not just on average

**Check:** Does the rendered mp4 have an audio stream; is that stream's
mean volume above a silence threshold across the *entire* duration
(check the closing segment specifically, not just an overall average);
and does the mp4's duration match the audit log's real recorded time?

**Automated**, and already wired in: `narration/audit_narrator.py`'s
`_verify_audio` checks stream presence and mean volume before reporting
success. Still worth running the duration cross-check by hand
(`ffprobe` the mp4 vs. the audit log's last event's `completed_at_ms`) —
see the honesty note below for why.

**Be honest about this one — this is the clearest case of "N/N events
passed" being insufficient.** The real root cause of a "no audio" defect
on this project was never TTS or muxing at all: `automation/
metabase_driver.py` was silently transcoding the *wrong source video* —
`glob.glob()` on an ever-growing directory of past recordings kept
returning a stale file from hours earlier, regardless of what had just
been recorded. Every automated check that existed at the time (event
count, audio-present-and-non-silent on the resulting file) stayed green,
because the wrong video genuinely did have valid, non-silent audio — it
just wasn't the audio for the run being checked. The defect was only
caught by manually cross-referencing the mp4's real duration against the
audit log's real recorded time, which is why that specific cross-check
is called out here as its own thing, not folded silently into "audio
present." Fixed by getting the exact recorded file from Playwright's own
`page.video().path()` instead of a directory scan.

---

### 6. Narration explains reasoning, not just mechanics

**Check:** For every narrated step, is there a stated reason *why* the
step happens — not just a description of *what* is being clicked or
typed? (Test: delete every "click X" / "select Y" clause and see if a
reason sentence is still left. If not, it fails.)

**Human.** This is a judgment call about whether an explanation is
actually a reason or just a rephrased description of the click. No
keyword-matching heuristic reliably tells the difference between "we
click Save, since that's the Save button" (fails) and "we click Save,
since saving means we can reopen this without rebuilding it" (passes).

**History:** the first content pass had lines like "now we select
Orders" and "we'll create a new dashboard to hold it" — mechanical
description with no reasoning attached. Fixed by rewriting every
narration line against `LESSON_CONTENT_STANDARD.md`'s rule 2.

---

### 7. Every value, column, or field named in narration is visible and highlighted at that moment

**Check:** When narration says a specific number, column name, or field,
is that exact thing on screen, highlighted, for as long as it's being
discussed — not just the click that eventually leads to it?

**Mostly human, partially automatable.** A frame extracted at the
narrated moment is the reliable check — pull a frame at the event's
timestamp and confirm the named thing is actually visible and
highlighted. A cheaper partial automated proxy exists for numbers
specifically: extract digit/spelled-number mentions from the narration
text (the same technique `narration/check_numbers.py` and
`narration/sync_guard.py` already use for the SQL pipeline) and confirm a
`highlight_target`/`highlight_targets` event with a matching literal
value is active at that timestamp. No automated check currently exists
for the Metabase path specifically — this would be worth building,
modeled on `narration/sync_guard.py`, rather than relying on frame checks
forever.

**History:** filter values (50 and 1000) were narrated while only the
closed filter *button* was highlighted, and column names were narrated
over a highlight on the generic table title, with no columns shown at
all — and named columns that didn't even match the table's real headers.
Fixed with `pre_actions` (reveal data before it's narrated) and
`highlight_targets` (highlight several named things at once). Also
discovered live that the first fix attempt for columns was itself wrong —
the highlight timed out because Metabase's query-builder editor shows no
data grid at all until `visualize` runs — caught only by pulling a frame,
not by any automated signal.

---

### 8. First-time concept introductions get more time and deeper narration than repeated actions

**Check:** For every step introducing a concept for the first time in
this video (not a repeated action type already shown earlier in the same
lesson) — does it hold noticeably longer than a repeated action, and does
the narration explain what the resulting object or state actually *is*
and *why it matters*, not just that the action completed?

**Both.** Automated: if the lesson script marks a step as concept-intro
(`lead_ms`/`post_hold_ms` set above the defaults), confirm the audit
log's actual measured duration reflects that configured value — cheap and
exact. Judging *which* steps should have been marked concept-intro in the
first place, and whether the narration actually explains the "what is
this" and "why does it matter," is a human read.

**History:** save-question and add-to-dashboard were paced identically to
a repeated click, despite introducing two brand-new object types (what a
saved question is, what a dashboard is), and the narration only confirmed
the action completed ("now we save this") rather than explaining what a
saved question or dashboard actually is. Fixed with per-event
`lead_ms`/`post_hold_ms` overrides and a documented
`CONCEPT_INTRO_HOLD_MS` convention in `automation/metabase_driver.py`.

---

### 9. Any artifact a lesson's `requires_state` declares actually matches reality after recording

**Check:** For a lesson with a `requires_state` block (see
`automation/state_seed.py`), does the declared artifact (a named
question, a named dashboard, what it should contain) actually exist in
that real shape in Metabase after recording — not just that the driver
reported success?

**Automated**, and the most important item to actually run, not skip,
on any lesson using `requires_state`: query the same artifacts by name
via Metabase's API (the same calls `state_seed.py` itself makes) and
check the real result, e.g. `GET /api/dashboard/:id` and confirm its
`dashcards` actually contain the expected card names.

**Be honest about this one too — it's the most direct proof yet that
"the driver didn't raise an exception" and "the audit log says N/N
events succeeded" are not the same claim as "the result is correct."**
Multi-video continuity work found a click that picked the wrong option
in a dashboard picker (clicked "New dashboard" instead of selecting the
already-seeded target with the same name) and silently created a
duplicate dashboard with an orphaned card on it. No exception. No failed
event. The audit log reported a clean 32/32. The only way this was ever
caught was querying the actual Metabase state afterward and finding two
active dashboards with the same name instead of one — exactly the
discipline this item exists to make routine rather than incidental.

---

### 10. Every filter/aggregation a lesson performs actually produces a real, non-empty result — checked BEFORE render, not after

**This item is different from every one above it: it is a pre-render
gate, not a post-render check.** Items 1 through 9 all verify something
against a rendered video that already exists — this one exists
specifically so a video with this defect never gets rendered in the
first place. Run it as step 0, before `automation/metabase_driver.py`
ever starts recording, not as one more thing to check afterward.

**Check:** For every filter, aggregation, or other data-producing step a
lesson script performs, does the equivalent query — run live against
Metabase's real `/api/dataset`, against whatever state will actually be
seeded for the real recording — return a real, meaningful, non-empty
result? A pure aggregation with no breakout (e.g. a bare `count`/`sum`)
additionally must not return a zero/null value when the narration
implies a real, non-trivial number.

**Automated, and a hard gate, not a soft warning: a lesson that fails
this must not proceed to render.** Either it's corrected (regenerated
with real data-grounded values, or hand-edited) and re-checked, or it's
blocked outright — never rendered with the defect still present, and
never silently downgraded to a post-render note for a human to notice
later. The whole point of running this before rendering is that a
video with this defect showing up in the final output is not an
acceptable outcome to catch after the fact.

**History:** the most serious defect found on this project so far. A
generated lesson filtered `Orders.Total` to a `$500–$5000` range —
Metabase's real data for that field tops out at `$159.35`. The query
returned zero rows, always. The video recorded anyway: a long dead-air
gap where the result should have been, while the narration continued
describing the (nonexistent) results as if they were on screen. Every
earlier item on this list checks for an *omission* — something true
that wasn't shown or said. This is the system stating something
**false**, in the course's own teaching voice, about data it never
actually checked. Confirmed reproducible directly: POSTing that exact
filter to Metabase's own `/api/dataset` returns 0 rows immediately, no
ambiguity — there was nothing subtle to have caught here, only nobody
had checked before rendering. Fixed in `wsda-pipeline-console` (the
separate console UI that drives this pipeline's generation): every
generated script now declares a `validations` block describing its own
filter/aggregation as a structured query, `console/validator.py` runs
each one live before a render is ever offered, and a script that fails
gets one automatic corrected-regeneration attempt before being blocked
outright. Real per-field data ranges are now also given to the
generator up front (`console/metabase_schema.py`), so a plausible-
sounding but physically-impossible value is less likely to be chosen in
the first place — the gate is the hard backstop, not the only defense.

---

## What this checklist does not cover

This list is delivery/content-defect verification for a single rendered
video. It does not check:

- Whether the lesson's *overall structure* matches the proven
  chapter-opener → lesson(s) → challenge pattern documented in
  `LESSON_CONTENT_STANDARD.md`'s "Grounded in SQL Essential Training"
  section — that's a judgment about the course as a whole, not a single
  render.
- Whether the underlying SQL/data adapter path (`generator/prompts.py` →
  production cards) has equivalent gaps — this checklist was written
  against the Metabase automation path's specific history; the SQL
  pipeline has its own partially-automated checks
  (`verify.py`'s `check_engagement`, `narration/qa.py`,
  `narration/sync_guard.py`) and would need its own pass to confirm this
  same checklist's items hold there too.
