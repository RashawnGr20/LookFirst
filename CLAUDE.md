# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DriveSim ("LookFirst") is a desktop driver's-test prep tool: a webcam-driven head/eye-tracking
simulator (pygame + MediaPipe) that scores whether a user performs the correct mirror/blind-spot
checks during scripted driving scenarios, plus a FastAPI/Postgres backend for accounts and
session/gaze history.

The repo is really **two independent Python apps** sharing no code, each with its own venv and
dependency file:

- `sim/` — the pygame desktop client (computer vision + rendering + scenario logic). Deps in
  `requirements.txt` (repo root), venv in `drivesim_env/`.
- `backend/` — a FastAPI + SQLAlchemy/Postgres API for auth and session/gaze data. Deps in
  `backend/requirements.txt`, venv in `backend_env/`.

## Commands

Backend (run from repo root so the `backend.*` absolute imports resolve):
```
backend_env\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```
Requires `backend/.env` (copy `backend/.env.example`). Startup aborts if `DATABASE_URL` or
`SECRET_KEY` is missing. `ANTHROPIC_API_KEY` is only needed for the coaching endpoint;
set `TRUST_PROXY_HEADER=true` only when running behind a trusted tunnel/reverse proxy so
`slowapi` keys rate limits on the real client IP. `.env` is gitignored.

For the two-machine dev setup (backend + Postgres on the main computer, sim on the laptop),
expose the backend via a tunnel (Cloudflare Tunnel / ngrok) and set `TRUST_PROXY_HEADER=true`.
Run uvicorn with `--host 0.0.0.0` in that case.

Sim client:
```
drivesim_env\Scripts\activate
pip install -r requirements.txt
python sim/mainlogic.py
```
Needs a working webcam. The auth/login screens require a reachable backend; the URL is read
from `sim/api_config.json` (gitignored; copy `sim/api_config.json.example`) via `sim/config.py`
and falls back to `http://127.0.0.1:8000` if the file is absent. Without a reachable backend
the sim can still be used in guest mode.

There is no test runner, linter, or build step configured for either half. `sim/test_auth.py`
is an ad hoc manual script (not pytest) that exercises signup/login against the live backend
with a throwaway account — not part of any CI.

## Sim architecture (`sim/`)

`mainlogic.py` is the entry point: a single `while running` loop that drives the `SceneGen`
state machine and — only while `scene.state` is `"calibration"` or `"simulation"` — pulls
camera frames, runs face tracking, fuses head pose and gaze into an observed zone, evaluates
scenario progress, and renders. When state is anything else (home, scene_select, etc.) the
loop just calls `scene.update()` with no tracking args and releases the camera.

Pipeline for a tracked frame:
1. `HeadTracker` (`headtracking.py`) runs MediaPipe face mesh, smooths landmark positions, and
   derives pitch/yaw/roll (`pitch_vectors`) plus normalized iris/gaze offsets
   (`normalized_gaze`) in head-relative (iris-in-eye-socket) coordinates. Before classification
   the head-relative gaze is projected to *world-relative* via
   `GazeZoneClassifier.to_world_gaze(norm_x, norm_y, rel_yaw, rel_pitch)`, which adds a
   per-user `k_yaw × rel_yaw` term to `norm_x` and `k_pitch × rel_pitch` term to `norm_y`
   (first-order additive model). Anchors and gaze both live in world-relative coords.
2. `feedBackEngine.assign_pose()` (`feedback.py`) maps pitch/yaw onto a fixed set of pose
   labels (`FORWARD`, `LEFT MIRROR`, `LEFT BLINDSPOT`, `RIGHT MIRROR`, `RIGHT BLINDSPOT`,
   `TOP MIRROR`, `LOOKING DOWN`) via hardcoded degree thresholds, debounced by a `pose_counter`
   (a pose must hold for 5 frames to become the `confirmed_pose`).
3. `ObservationEngine` (`observation.py`) + `GazeZoneClassifier` (`gaze_zones.py`) fuse the
   confirmed head pose with per-user gaze evidence into an `observed_zone`. Head-only zones
   (`LEFT/RIGHT BLINDSPOT`) trust head pose. Gaze-anchored zones (`TOP MIRROR`, `LEFT MIRROR`,
   `RIGHT MIRROR`, `LOOKING DOWN` in `ZONE_CATALOG`) require the classifier to agree — a head
   turn with eyes at the forward baseline downgrades to `FORWARD`, catching the "looked
   through the mirror" case. The engine also debounces its output over 5 frames independent of
   `feedBackEngine`'s pose debounce.
4. `Scene` / `SequenceScene` / `coverageScene` (`scenes.py`) evaluate the `observed_zone`
   against a scenario's `expected_sequence`. Note the counter passed for the `min_glance` gate
   is `observation_engine.zone_counter`, not `feedback.pose_counter`, so the gate applies
   uniformly to head- and gaze-triggered zones.
5. `SceneGen` (`scenegen.py`) + `UI` (`UI.py`) own the pygame window, screen state machine
   (`home`, `scene_select`, `about`, `features`, `auth`, `calibration`, `simulation`,
   `results`), click routing, and all drawing. `SceneGen.update()` dispatches to an
   `update_<state>` method per state.

Calibration (inside the `mainlogic.py` main loop, driven by `scene.state == "calibration"`)
runs four sequential sub-phases, tracked with module-level state, not an object:
- **Head baseline**: buffer ~60 frames of pitch/yaw/roll while the user holds still facing
  forward (rejecting the buffer and restarting if yaw spread is too high) to compute
  `baseline_angles`. All subsequent pitch/yaw/roll are reported relative to this baseline
  via `angle_diff_deg`.
- **Center reference (`center_ref`)**: 50 frames of "look at the center dot" — captures both
  `HeadTracker.eye_height_ref` (per-eye lid-opening baseline used to correct `norm_y` for
  eye-openness changes) and the classifier's forward-gaze baseline (mean `(norm_x, norm_y)`
  when the user is head-forward-eyes-forward).
- **Zone anchor walk (`zone_anchor`)**: iterates `ZONE_CATALOG`, showing a dot at each zone's
  `screen_pos` and collecting ~30 samples of `(world_x, world_y)`. Each zone gets a per-user
  anchor stored as a single point (no radius). After the last zone, `finalize_anchors()`
  caps the forward baseline's dead-zone radius so it can't swallow the nearest anchor, and
  emits a diagnostic dump of anchor positions and pairwise distances for future debugging.
- **Head scale (`head_scale`)**: ~150 frames of "keep eyes on the center dot, slowly move
  your head around." Exploits VOR (vestibulo-ocular reflex) — if the eyes stay on target,
  every sample obeys `0 = norm + K × rel_head`. Two independent OLS regressions fit
  `k_yaw` and `k_pitch` on the classifier; three quality checks (head stddev ≥ 3°,
  R² ≥ 0.4, K in [0.005, 0.05]) gate the fit, otherwise the axis falls back to the
  default 0.02 with a logged reason.

`GazeZoneClassifier` uses a Voronoi model rather than per-anchor discs: `classify(x, y)`
returns `None` when the gaze is inside the forward baseline's dead-zone (at rest), else the
name of the nearest anchor by Euclidean distance. `ObservationEngine.update` applies a light
EMA smoothing (`α = 0.4`) to the gaze before classification to damp MediaPipe iris jitter at
cell boundaries; the 5-frame `zone_counter` debouncer still filters transient flips.

Adding a new gaze-observable zone is a one-line data edit: append a `ZoneSpec` to
`ZONE_CATALOG` in `sim/gaze_zones.py`. `UI.get_calibration_target_pos` builds its target
table from `ZONE_CATALOG` at import, so no UI edit is needed.

Scenario metadata is defined in **two places that must be kept in sync**: `scenes.py` defines
the evaluation logic and `expected_sequence` per scenario key, while `SceneGen.scene_info` in
`scenegen.py` separately holds the display title/description/required-checks/images for the
same scenario keys (used by the scene-select and scene-intro screens).

## Backend architecture (`backend/`)

Standard FastAPI layout: `main.py` loads `backend/.env`, aborts at startup if `DATABASE_URL`
or `SECRET_KEY` is missing, wires the `slowapi` rate limiter, calls
`Base.metadata.create_all(engine)`, and mounts the `auth`, `sessions`, `gaze`, and `coaching`
routers. Data model (`backend/database/models.py`): `User` 1:N `Session` 1:N `GazeEvent`, plus
`Session` 1:1 `CoachingReport`; plain SQLAlchemy declarative against Postgres.

Auth (`backend/auth/`) is JWT-based: `security.py` hashes/verifies passwords with
passlib/bcrypt, `tokens.py` issues a `python-jose` JWT carrying `user_id` + expiry,
`dependencies.py` `get_current_user` decodes the bearer token and raises 401 on a
bad/expired token or unknown user. `/auth/login`, `/auth/signup`, and
`/coaching/{id}/report` are per-IP rate-limited.

The `sessions`, `gaze`, and `coaching` routers all gate access with
`session.user_id != current_user.id`. `backend/coaching/service.py` calls an Anthropic model
via LangChain to generate a per-session coaching report, cached in the `coaching_reports`
table.

`sim/auth_client.py` (login/signup) and `sim/api_client.py` (session/gaze/coaching calls) are
the sim's backend consumers; both take `BASE_URL` from `sim/config.py`. The session-logging
path is only half-wired: `mainlogic.py` calls `api_client.complete_session(...)` but never
`create_session`, so `backend_session_id` stays `None` and nothing is actually posted yet.
