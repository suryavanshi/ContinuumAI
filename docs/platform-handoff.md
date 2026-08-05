# Continuum Platform Handoff

## Handoff Snapshot

Date: 2026-08-04  
Repository: `ContinuumAI`  
Working directory: `/Users/kb/Documents/proj/git_projs/ContinuumAI`  
Deployed application: [Continuum training console](https://suryavanshi--continuum-training-console-console.modal.run)  
Modal dashboard: [continuum-training-console](https://modal.com/apps/suryavanshi/main/deployed/continuum-training-console)

This document records the implemented state and the operational context needed
to continue development. It deliberately contains no password, session secret,
Modal token, Hugging Face token, or other credential.

## What Was Built

### Training console

- Static HTML/CSS/JavaScript frontend served by the Python backend.
- Responsive training-run dashboard with a light-blue Continuum theme.
- Run selector, run status, core metrics, loss chart, trajectory table, token
  preference heatmap, logs, and normalized config/command views.
- New-run dialog populated from a server-side algorithm/model catalog.
- Explicit launch confirmation for billable Modal GPU work.
- Polling of an active run until it reaches a terminal state.

### API and persistence

- `GET /api/health`
- `GET /api/auth/status`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/catalog`
- `GET /api/runs`
- `POST /api/runs`
- `GET /api/runs/:id`
- `POST /api/runs/:id/launch`
- `POST /api/runs/:id/ingest`
- Atomic JSON-file writes through `RunStore`.
- Local state at `.continuum/runs.json`.
- Deployed state at `/data/runs.json` on the `continuum-console-data` Modal
  Volume.

### Launch integration

- Catalog and command generation for:
  - SDPO via `SDPO/modal_verl_sdpo.py::main`;
  - OPD via `OPD/modal_verl_qwen35_opd.py`;
  - Harvey LAB SDPO via `SDPO/modal_verl_sdpo.py::harvey_main`;
  - Kimi SDPO via `SDPO/modal_verl_kimi_k26_sdpo.py`.
- Commands are argv lists and do not use a shell.
- Launching is disabled unless `CONTINUUM_ENABLE_LAUNCH=1`.
- Run ID must be echoed exactly in the launch request.
- Model catalog and training step/dataset-size limits are enforced server-side.
- Deployed job submission uses Modal workload identity. Modal account tokens are
  not copied into the image or attached as app secrets.

### Authentication and deployment

- Public deployment requires authentication.
- A single admin username/password is loaded from the
  `continuum-console-auth` Modal Secret.
- Sessions are HMAC-signed, expiring, HTTP-only, SameSite=Strict cookies.
- Deployed cookies use the Secure flag.
- State-changing cross-origin requests are rejected.
- The application is deployed by `continuum_console/modal_app.py` with one
  control-plane container and a persistent Modal Volume.

## Real Training Smoke Result

The local GUI successfully triggered a real one-step SDPO training run:

- Run ID: `run_4486983a77e0`
- Name: `qwen35_08b_gui_smoke`
- Model: `Qwen/Qwen3.5-0.8B`
- Dataset: `gsm8k_sdpo`
- Compute: H100 x2 smoke profile
- Result: completed one of one training steps
- Recorded distillation loss: approximately `-0.115183`
- Modal app ID: `ap-G0pcFOUC8m4a6OU1jgZP6w` (stopped after completion)
- Checkpoint:
  `checkpoints/gsm8k_sdpo-sdpo-20260804T071253Z/global_step_1`
- Actor checkpoint file size: approximately 2.1 GiB
- Warning: `grad_norm:nan` was observed and treated as non-fatal for this smoke
  gate. It should be investigated before longer training.

The deployed run store was seeded with the real Qwen smoke run so that the
hosted UI shows real training output. The older
`run_sdpo_airline_001` entry is demonstration data and should remain visibly
labelled as such if retained.

## Important Files

### Console

- `continuum_console/api.py` — HTTP server, API routing, launch process,
  ingestion, and Verl log metric parsing.
- `continuum_console/auth.py` — shared-admin credential validation and signed
  session cookies.
- `continuum_console/store.py` — JSON persistence and run config validation.
- `continuum_console/catalog.py` — approved algorithms, models, defaults, and
  topology metadata.
- `continuum_console/commands.py` — argv-safe mapping from run config to Modal
  launchers.
- `continuum_console/modal_app.py` — authenticated Modal web deployment, image,
  secret, and volume bindings.
- `continuum_console/static/index.html` — console shell and dialogs.
- `continuum_console/static/app.js` — client API calls, rendering, auth, launch,
  and polling behavior.
- `continuum_console/static/styles.css` — responsive light-blue visual system.

### Training launchers and guidance

- `SDPO/modal_verl_sdpo.py` — Qwen/Harvey SDPO Modal/Verl launcher.
- `SDPO/modal_verl_kimi_k26_sdpo.py` — Kimi SDPO topology launcher.
- `OPD/modal_verl_qwen35_opd.py` — Qwen OPD launcher.
- `MODAL_SKILL.md` — Modal execution and debugging runbook.
- `POST_TRAINING_PLAN.md` — broader training framework strategy.
- `docs/sdpo-learnings.md` — SDPO bridge findings and caveats.
- `docs/scaling-sdpo-replication.md` — SDPO smoke gates and scaling ladder.
- `docs/platform-roadmap.md` — prioritized production roadmap produced with
  this handoff.

### Tests

- `tests/test_continuum_console.py` — command construction, auth, metric
  parsing, store, and API behavior.
- `tests/test_modal_verl_sdpo_args.py` — SDPO argument/config generation.
- `tests/test_inspect_sdpo_log.py` — smoke-log classification.

At this handoff, all 41 tests pass. The palette-only change also passed
`node --check` and `git diff --check` and was visually verified on localhost
and the Modal deployment with no browser console errors.

## Run Locally

Unauthenticated development mode:

```bash
python3 -m continuum_console.api --port 8787
```

Authenticated local mode:

```bash
CONTINUUM_REQUIRE_AUTH=1 \
CONTINUUM_ADMIN_USER=admin \
CONTINUUM_ADMIN_PASSWORD='<12+ character password>' \
CONTINUUM_SESSION_SECRET='<32+ random characters>' \
python3 -m continuum_console.api --port 8787
```

State is written to `.continuum/runs.json` unless `--data` is supplied.

## Deploy on Modal

Create or rotate the application auth secret without committing its values:

```bash
modal secret create continuum-console-auth \
  CONTINUUM_ADMIN_USER=admin \
  CONTINUUM_ADMIN_PASSWORD='<strong rotated password>' \
  CONTINUUM_SESSION_SECRET='<new random 32+ character value>'
```

Deploy:

```bash
modal deploy continuum_console/modal_app.py
```

Basic post-deploy checks:

```bash
curl -fsS https://suryavanshi--continuum-training-console-console.modal.run/api/health
modal app list
```

Then verify login, run selection, real Qwen metrics, and logout in a browser.

## Configuration Reference

### Authentication

- `CONTINUUM_REQUIRE_AUTH=1` — require authentication.
- `CONTINUUM_ADMIN_USER` — shared admin username.
- `CONTINUUM_ADMIN_PASSWORD` — shared admin password, minimum 12 characters.
- `CONTINUUM_SESSION_SECRET` — HMAC secret, minimum 32 characters.
- `CONTINUUM_SECURE_COOKIE=1` — add Secure to the session cookie.

### Launch and safety

- `CONTINUUM_ENABLE_LAUNCH=1` — allow billable Modal submissions.
- `CONTINUUM_MODAL_BIN` — Modal CLI executable; deployed value is `modal`.
- `CONTINUUM_SMOKE_VERL_IMAGE_TAG` — image override for guarded smoke jobs.
- `CONTINUUM_MAX_STEPS` — maximum accepted training steps; default 100.
- `CONTINUUM_MAX_TRAIN_ROWS` — maximum accepted training rows; default 2048.
- `CONTINUUM_MAX_VAL_ROWS` — maximum accepted validation rows; default 512.

Training launcher variables such as Hugging Face and W&B credentials must be
managed with provider-native secrets and must not be added to this repository.

## Current Feature Matrix

| Surface | Status | Notes |
|---|---|---|
| Login/logout | Prototype working | One shared admin; no OIDC, MFA, RBAC, rate limiting, or server-side revocation |
| Training Runs | Working prototype | Draft, launch, poll, metrics, logs, config, trace inspector |
| SDPO launch | Smoke-tested | Real Qwen3.5-0.8B one-step run completed |
| OPD launch | Wired | Command/API tests exist; no GUI-triggered smoke recorded in this handoff |
| Harvey SDPO | Wired | Catalog and command mapping exist |
| Kimi SDPO | Wired/experimental | Large-model topology risk remains |
| Evaluations | Placeholder | Sidebar button only |
| Checkpoints | Placeholder | Checkpoint exists in volume, but no resource/API/UI |
| Jobs | Placeholder | Job metadata is nested in a run; no list/cancel/retry/reconciliation |
| Compute | Placeholder | Form has two labels, not a compute inventory or policy model |
| Settings | Placeholder | Sidebar button only |
| GRPO family | Not exposed | Current SDPO/OPD launchers use GRPO internally, but no standalone typed recipes/UI |

## Known Architectural Limitations

### Security

- Shared admin credential rather than individual users.
- No login throttling, lockout, MFA, OIDC, RBAC, audit log, or API-key model.
- No explicit CSRF token or Content Security Policy.
- No log secret-redaction layer.
- The ingest endpoint uses the same browser/admin authentication boundary as
  human control-plane operations.

### Jobs

- Submission and log supervision run in a daemon thread in the web process.
- A web-container restart can lose supervision even if external work continues.
- No durable attempt model, heartbeat, idempotency key, cancellation, retry, or
  Modal reconciliation.
- Only the latest 1,000 captured output lines are stored on the run record.
- Metric parsing is regex-based and recognizes only a small subset of Verl
  metrics.

### Storage

- All metadata, logs, metrics, and traces share one JSON file.
- `max_containers=1` prevents concurrent writers but also makes the service a
  single control-plane replica.
- No database constraints, pagination, ownership, append-only events, backup
  job, or schema migration mechanism.
- The store auto-seeds demo data when the file is absent, which is useful for
  development but undesirable in a production workspace.

### Frontend

- No router or view model for sidebar sections.
- Evaluations, Checkpoints, Jobs, Compute, and Settings buttons have no event
  handlers.
- Polling is fixed at five seconds for up to one hour and does not recover from
  transient fetch failures.
- No explicit loading skeletons, global error boundary, pagination, filters, or
  accessibility test suite.

### Algorithms

- Algorithm configuration is a flat form and hard-coded branch structure.
- Catalog validation allowlists models globally rather than validating
  model/algorithm/topology compatibility as a typed matrix.
- Standalone GRPO, Dr. GRPO, DAPO, GSPO, RLOO, ReMax, and other Verl recipes are
  not first-class platform options.
- Framework/image commits are not stored in an immutable run manifest.

## Verl Context for the Next Engineer

The sibling local Verl checkout was inspected at commit `cb821109`. Its current
algorithm surface includes:

- advantage estimators: GAE, GRPO, GRPO Pass@K, vectorized GRPO, RLOO,
  vectorized RLOO, REINFORCE++, REINFORCE++ baseline, ReMax, OPO, GPG, GDPO,
  and optimal-token-baseline variants;
- policy-loss recipes/examples: GSPO, SAPO, CISPO, DPPO-TV/KL, GMPO, and GPG;
- DAPO as a GRPO-based recipe with decoupled clipping and dynamic sampling;
- experimental one-step-off-policy and fully asynchronous variants;
- OPD as GRPO plus a distillation loss path.

Do not expose every upstream option immediately. Implement the typed recipe
adapter and smoke-test GRPO, Dr. GRPO, DAPO, and GSPO first, as specified in the
platform roadmap.

## Recommended Next Change

The next implementation should be a durable `Job`/`JobAttempt`/`JobEvent`
model with idempotent launch and Modal reconciliation, followed immediately by
a real Jobs page. This resolves the most visible broken tab while replacing the
most fragile backend behavior. Security headers, CSRF tokens, login throttling,
audit events, and log redaction should be included in the same production-
foundation milestone.

## Credential and Secret Handling

- The admin password has been shared out-of-band during development. Rotate it
  before granting access to another person or treating the deployment as
  production.
- Do not place the password in this document, shell history, source, tests, or
  screenshots.
- Rotate the session secret together with the password to invalidate existing
  sessions when handing ownership to a new operator.
- Continue using Modal workload identity for backend submission; do not attach
  Modal account-token environment variables to the application.

## Working Tree State at Handoff

The repository contains uncommitted work from the platform implementation.
The known status at handoff includes modifications to `.gitignore`, `README.md`,
`SDPO/modal_verl_sdpo.py`, and `tests/test_modal_verl_sdpo_args.py`, plus new
`continuum_console/` and `tests/test_continuum_console.py` paths. These changes
belong to the current implementation and must not be discarded during cleanup.
Review the complete diff, remove generated `__pycache__` files from consideration,
then commit intentionally when the user requests publication.
