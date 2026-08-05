# Continuum Platform Roadmap

## Purpose

This roadmap turns the current authenticated training-run console into a durable,
multi-user control plane for Verl-based post-training on Modal. It is based on
the implementation in `continuum_console/`, the current Modal deployment, and
the local Verl checkout at commit `cb821109`.

The ordering is intentional: durable identity and jobs come before adding more
GPU recipes. Without those foundations, every new algorithm multiplies the
number of unsafe and unrecoverable states.

## Current Baseline

The platform currently provides:

- a Python HTTP API and static single-page console with no frontend build step;
- a signed-cookie, single-admin login for public Modal deployment;
- draft creation and argv-safe launch commands for SDPO, OPD, Harvey SDPO, and
  Kimi SDPO;
- detached Modal submission from the control-plane container;
- JSON run persistence on a Modal Volume;
- log capture, limited Verl metric parsing, trace inspection, and launch safety
  limits;
- a deployed authenticated console and one completed real Qwen3.5-0.8B SDPO
  smoke run.

The sidebar entries for Evaluations, Checkpoints, Jobs, Compute, and Settings
are placeholders. They have neither routing nor backend resources. Authentication
is a shared admin credential, and a running job is supervised by a background
thread in the web process.

## Target Product Shape

The production platform should have four explicit planes:

1. **Identity plane**: users, organizations/workspaces, roles, sessions, API
   keys, audit events, and resource ownership.
2. **Control plane**: validated experiment specs, jobs, retries, cancellation,
   schedules, budgets, compute policies, and backend adapters.
3. **Data plane**: immutable run manifests, append-only events, metrics, logs,
   traces, artifacts, checkpoints, and evaluation results.
4. **Execution plane**: Modal functions/apps that execute Verl recipes and
   report progress through scoped job credentials rather than browser sessions.

## Priority 0: Stabilize the Existing Deployment

**Goal:** eliminate immediate operational and security risks before expanding
the feature surface.

### Security hardening

- Rotate the currently shared admin password and session secret; keep both only
  in the `continuum-console-auth` Modal Secret.
- Add login rate limiting, exponential backoff, and structured failed-login
  audit events. Use a shared store so limits survive container replacement.
- Add `Content-Security-Policy`, `Strict-Transport-Security`,
  `X-Frame-Options`/`frame-ancestors`, and `Permissions-Policy` headers.
- Add an explicit CSRF token for state-changing browser requests. SameSite and
  origin validation remain useful defense-in-depth but should not be the only
  CSRF control.
- Separate browser authentication from worker ingestion. A training worker
  should receive a short-lived, run-scoped credential that can only append
  events to its own job.
- Redact secrets, bearer tokens, signed URLs, and common credential patterns
  before persisting or returning logs.
- Put a reverse-proxy/request limit in front of login and ingestion paths; add
  per-endpoint body and event-count limits.
- Add a dependency lock and automated vulnerability, secret, and static checks
  in CI.

### Correctness and reliability

- Stop using a daemon thread plus child `modal run` process as the source of
  truth. A web-container restart currently loses supervision of active work.
- Introduce a job ID and append-only job events before submission. Store the
  Modal app/function call ID as soon as it is known.
- Reconcile non-terminal jobs against Modal on startup and periodically.
- Make launch idempotent with an idempotency key and a uniqueness constraint on
  active submissions.
- Add explicit terminal states: `succeeded`, `failed`, `cancelled`, `timed_out`,
  and `orphaned`; reserve run status for the experiment lifecycle.
- Back up the current Modal Volume before the storage migration.

### Exit criteria

- A control-plane restart does not lose, duplicate, or incorrectly complete a
  GPU job.
- Brute-force login attempts are throttled and visible in an audit log.
- No browser credential can ingest data for an unrelated run.
- Existing SDPO smoke creation, launch, monitoring, and completed-run viewing
  remain covered by an end-to-end test.

## Priority 1: Multi-User Login and Workspace Authorization

**Goal:** replace the shared admin password with real user identity and scoped
access.

### Recommended first implementation

Use an external OIDC provider for authentication and keep authorization in the
application. Avoid building password reset, email verification, and MFA from
scratch. For an internal deployment, Google Workspace, GitHub, or another OIDC
provider is sufficient; for a customer-facing product, use a managed identity
provider with organization and MFA support.

### Data model

- `users`: provider subject, email, display name, state, created/last-login time.
- `workspaces`: name, slug, owner, budget policy, Modal environment mapping.
- `memberships`: user, workspace, role (`owner`, `admin`, `operator`, `viewer`).
- `sessions`: hashed session ID, user, expiry, rotation/revocation metadata.
- `api_keys`: hashed key, scopes, workspace, creator, expiry, last-used time.
- `audit_events`: actor, action, resource, workspace, request ID, IP hash,
  timestamp, and redacted metadata.

### Authorization rules

- Viewers can read runs, jobs, metrics, evaluations, and artifacts.
- Operators can create and cancel jobs but cannot change credentials, compute
  allowlists, or budgets.
- Admins manage members, recipes, datasets, and workspace configuration.
- Owners manage billing/budgets, destructive retention changes, and ownership.
- Every run, job, checkpoint, evaluation, and API key belongs to a workspace.
- Server-side authorization is checked on every resource lookup; the UI is not
  an authorization boundary.

### Session requirements

- Secure, HTTP-only, SameSite cookies with server-side revocation.
- Session rotation after login and privilege change.
- Short inactivity timeout plus a bounded absolute lifetime.
- Logout-all-sessions and administrator revocation.
- Optional enforced MFA claim for launch, cancel, member, and secret actions.

### Exit criteria

- Two workspaces cannot read or mutate each other's resources, including by
  guessing IDs.
- Role behavior is integration-tested for every state-changing endpoint.
- Launch and cancel events contain a durable actor identity and audit record.

## Priority 2: Durable Jobs Tab

**Goal:** make Jobs the operational source of truth rather than a projection of
the selected run.

### Backend resources and API

- `POST /api/jobs`: validate a frozen experiment revision and enqueue once.
- `GET /api/jobs`: filter by status, owner, algorithm, model, time, and backend.
- `GET /api/jobs/:id`: manifest, backend identity, status, attempts, timing,
  cost estimate, and recent events.
- `POST /api/jobs/:id/cancel`: request cancellation and reconcile the result.
- `POST /api/jobs/:id/retry`: create a linked attempt from an immutable spec.
- `GET /api/jobs/:id/events`: cursor-based event/log stream.
- `POST /internal/jobs/:id/events`: scoped worker ingestion with sequence IDs.

### Job state machine

`draft -> queued -> provisioning -> running -> checkpointing -> succeeded`

Terminal alternatives are `failed`, `cancelled`, `timed_out`, and `orphaned`.
Retries create new attempts rather than rewinding a terminal job. State changes
must be monotonic and recorded as events.

### Modal adapter

- Submit a dedicated Modal function call or app per attempt; do not spawn a CLI
  subprocess inside the web server for the long-term design.
- Store Modal call/app/function IDs and deep links.
- Support cancel, timeout, retry policy, preemption signal, and termination
  reason mapping.
- Have the worker heartbeat and append structured progress events.
- Reconcile active jobs against Modal after control-plane startup.

### Jobs UI

- Paginated/filterable jobs table with status, algorithm, model, compute, owner,
  elapsed time, estimated cost, and last heartbeat.
- Detail page with timeline, live logs, metrics, manifest, attempts, artifacts,
  and clear cancel/retry actions.
- Empty, loading, stale, disconnected, and terminal failure states.
- Server-sent events for logs/status initially; WebSockets are unnecessary
  unless interactive terminals are later added.

### Exit criteria

- Jobs survive web deployment and container replacement.
- Cancel is reflected both in Modal and in the platform within a bounded time.
- Duplicate launch requests create one job.
- Logs can be resumed from a cursor without polling the whole run record.

## Priority 3: Compute Tab and Policies

**Goal:** expose available execution profiles and capacity without pretending
the platform is a cloud scheduler.

### Compute model

- `compute_profiles`: GPU type/count, node count, CPU/memory, timeout, region,
  training backend, rollout backend, FSDP/FSDP2/Megatron, TP/PP/EP, and image.
- `compute_policies`: allowed profiles by workspace/role, maximum parallel jobs,
  maximum runtime, and spend ceilings.
- `capacity_snapshots`: availability/queue observations when Modal exposes them;
  otherwise label capacity as unknown rather than fabricate real-time values.
- `cost_rates`: versioned estimates with an observed-cost override when billing
  data becomes available.

### Compute UI

- Cards/table for approved presets such as H100 x2 smoke, H100 x8, H200 LoRA,
  H200 full, and B200 full.
- Compatibility matrix: model, algorithm, training backend, rollout backend,
  minimum GPUs, tested status, and last successful run.
- Preflight estimator for GPU-hours, memory risk, token budget, and rough cost.
- Admin-only editing of profile allowlists and spend limits.
- Links from a compute profile to active/recent jobs using it.

### Guardrails

- Enforce limits on the server from immutable profile IDs; never accept a raw
  arbitrary Modal GPU specification from the browser.
- Require an explicit cost summary and confirmation for launch.
- Support workspace concurrency and daily/monthly estimated-spend ceilings.
- Add an emergency launch-disable switch that does not block reads or cancels.

### Exit criteria

- Every job references a versioned compute profile.
- Unsupported model/profile combinations fail preflight before GPU allocation.
- Estimated and observed runtime/cost are retained for later calibration.

## Priority 4: Experiment Specs and Verl GRPO Family

**Goal:** support algorithms through typed recipes instead of adding one-off
form flags and command branches.

### Schema first

Introduce a versioned experiment schema with these sections:

- model/checkpoint/tokenizer;
- dataset and preprocessing revision;
- algorithm and policy-loss configuration;
- reward/verifier and environment;
- rollout engine and sampling;
- trainer topology and optimization;
- evaluation/checkpoint schedule;
- compute profile, limits, and observability;
- exact Verl image/tag and git SHA.

Store the normalized immutable spec on each job. The UI edits a draft revision;
launch always references a frozen revision.

### Supported rollout order

The local Verl checkout exposes more algorithms than should be enabled at once.
Use a maturity matrix: `available upstream`, `adapter implemented`, `smoke
tested on Modal`, and `production approved`.

#### Wave A: GRPO baseline and closely related variants

1. **GRPO** — `adv_estimator=grpo`, grouped rollouts (`n >= 2`), configurable
   KL and loss aggregation. This becomes the reference RLVR recipe.
2. **Dr. GRPO** — GRPO with `norm_adv_by_std_in_grpo=False`, normalized
   sequence-token aggregation, and no actor KL loss per the upstream recipe.
3. **DAPO** — GRPO advantages plus decoupled clipping, dynamic sampling,
   token-level loss, and overlong filtering/reward shaping. Treat this as a
   recipe, not only an enum value.
4. **GSPO** — sequence-level policy optimization through
   `policy_loss.loss_mode=gspo`.

Wave A acceptance requires one-step and short multi-step Qwen smoke tests,
config snapshots, metric parsing, checkpoint discovery, cancellation, and a
base-vs-trained evaluation.

#### Wave B: alternative baselines and estimators

- GRPO Pass@K (`grpo_passk`)
- vectorized GRPO (`grpo_vectorized`)
- RLOO and vectorized RLOO
- REINFORCE++ and REINFORCE++ baseline
- ReMax
- OPO
- GDPO
- GPG
- optimal-token-baseline variants

Expose these only after the common reward, grouping, and evaluation contracts
are stable. Each recipe needs parameter constraints; for example, group-based
estimators require multiple samples per prompt, while ReMax requires a greedy
baseline rollout.

#### Wave C: policy-loss and specialized recipes

- SAPO, CISPO, DPPO-TV, DPPO-KL, and GMPO;
- one-step-off-policy and fully asynchronous GRPO/DAPO as experimental modes;
- multimodal GRPO and Flow-GRPO only after the text pipeline is stable;
- OPD/SDPO combinations with GRPO task rewards as a separate recipe family.

### Verl adapter design

- Keep catalog metadata separate from command construction.
- Generate Hydra overrides from typed config objects and validate constraints
  before launch.
- Pin and record the Verl commit/image; do not infer behavior from `latest`.
- Provide recipe-specific parsers for metrics, artifacts, and failure reasons.
- Maintain golden generated commands/configs in tests.
- Import known-good upstream scripts as references, not mutable runtime
  dependencies.

### Exit criteria

- A user can create, preflight, launch, monitor, cancel, and compare GRPO and
  Dr. GRPO jobs from the GUI.
- DAPO and GSPO each pass a Modal smoke gate before becoming selectable outside
  admin/experimental mode.
- The stored manifest can reproduce the exact command and framework revision.

## Priority 5: Evaluations and Checkpoints Tabs

### Evaluations

- Define evaluation suites and immutable suite revisions.
- Run evaluation as jobs linked to a base model or checkpoint.
- Store aggregate metrics and per-example outcomes separately.
- Provide base-vs-checkpoint and run-vs-run comparisons with confidence
  intervals where appropriate.
- Add promotion gates such as target improvement, maximum regression, invalid
  output rate, and cost.

### Checkpoints

- Discover checkpoints from the configured Modal Volume/path.
- Record producing job, global step, model/config revision, file manifest,
  byte size, format, and integrity hashes.
- Expose conversion/export status and Hugging Face upload as separate audited
  jobs.
- Add retention rules and protect promoted checkpoints from deletion.
- Never expose raw signed artifact URLs longer than necessary.

### Exit criteria

- A completed training job links to verified checkpoint records.
- A checkpoint can be evaluated without manually entering its path.
- Promotion decisions and retention changes are audited.

## Priority 6: Settings, Observability, and Operations

### Settings

- Workspace members and roles.
- OIDC configuration status, API keys, and session revocation.
- Modal environment/volume bindings by reference, never secret value display.
- Model/dataset/recipe allowlists.
- Compute policies, budgets, retention, and feature flags.

### Observability

- Structured JSON application logs with request/job/run IDs.
- Metrics for API latency/error rate, login failures, queued/running jobs,
  heartbeat age, launch latency, GPU runtime, token throughput, and ingestion
  lag.
- Error reporting for frontend and backend with secret scrubbing.
- Alerts for orphaned jobs, missing heartbeat, repeated launch failures, volume
  errors, and budget thresholds.
- Health endpoints split into liveness and dependency readiness.

### Disaster recovery

- Automated database backups and restore drills.
- Artifact/checkpoint inventory reconciliation.
- Documented credential rotation and emergency launch-disable procedure.
- Deployment rollback to the previous known-good Modal revision.

## Storage Migration

The single JSON file is acceptable for a one-container prototype but not for
multi-user concurrency, pagination, append-only logs, or relational ownership.

Recommended progression:

1. Define repository interfaces for users, runs, jobs, events, artifacts, and
   evaluations.
2. Add SQLite only for local development if useful.
3. Use managed Postgres for the deployed control-plane metadata.
4. Keep large logs/traces/artifacts in object or volume storage, referenced by
   immutable metadata records.
5. Migrate existing JSON runs with a versioned, idempotent importer and retain
   the source backup.

## Testing and Delivery Gates

- Unit tests: validation, state transitions, auth/session behavior, redaction,
  command generation, and metric parsing.
- API integration tests: ownership, role matrix, idempotency, cancel/retry,
  event sequencing, and pagination.
- Adapter contract tests: generated Verl configs against pinned upstream
  schemas and golden commands.
- Browser tests: login/logout, navigation, run creation, launch confirmation,
  jobs live state, compute preflight, and responsive layouts.
- Modal smoke tests: one-step algorithm gates with strict cost/step limits.
- Security tests: CSRF, IDOR, session fixation/expiry, rate limiting, injection,
  log redaction, and secret scanning.
- Deployment gate: migrations, backup, health check, smoke login, and rollback
  instructions must all succeed.

## Suggested Delivery Sequence

### Milestone 1 — Production foundation

- Postgres repository layer and JSON importer.
- OIDC users/workspaces/RBAC.
- security headers, CSRF, rate limiting, audit log, and redaction.
- durable job/event schema and Modal reconciliation.

### Milestone 2 — Working operations UI

- real Jobs tab with cancel/retry and event streaming;
- real Compute tab with approved profiles and preflight;
- route-aware frontend shell with loading/error/empty states;
- end-to-end launch test through Modal.

### Milestone 3 — Verl RLVR

- typed experiment spec and Verl adapter;
- GRPO and Dr. GRPO;
- DAPO and GSPO behind experimental flags;
- Qwen smoke ladder and evaluation comparison.

### Milestone 4 — Artifact lifecycle

- Checkpoints, Evaluations, promotion gates, export, and retention;
- settings/admin surfaces;
- cost and performance dashboards.

### Milestone 5 — Algorithm expansion

- Wave B estimators, then Wave C loss modes;
- asynchronous and multimodal recipes only after operational SLOs are met.

## Immediate Next Sprint

1. Freeze and document the current JSON schema and export a backup.
2. Add `Job`, `JobAttempt`, and `JobEvent` models plus a repository interface.
3. Change launch to create an idempotent job record before invoking Modal.
4. Add Modal status reconciliation and cancel support.
5. Turn the Jobs sidebar entry into a real route backed by the new API.
6. Add CSP/HSTS/frame policy, CSRF tokens, login throttling, and audit events.
7. Draft the versioned experiment schema and implement canonical GRPO command
   generation from the pinned local Verl checkout.
8. Add a guarded one-step Qwen GRPO smoke profile, but do not expose it to all
   users until cancellation and reconciliation are proven.

