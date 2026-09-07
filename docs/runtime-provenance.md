# Runtime provenance and passive diagnostics

`ApplicationRuntime` captures read-only provenance at construction and passes
that same snapshot to the dashboard. `/api/status` exposes it under
`runtime_provenance`. Dashboard-only composition takes its own construction
snapshot. Repeated GETs do not hash files again, inspect newer Git refs, reload
configuration, load a model, or issue control commands.

Schema 1 reports:

- Git revision and branch, including a detached revision, or explicit unknown
  values when Git is unavailable. Dirty status is scoped to application source,
  packaging metadata and the default configuration; it is not whole-repo status.
- SHA-256 manifest and aggregate digest for package Python, HTML, JavaScript and
  CSS files. Static images, dependencies/native libraries and device state are
  outside that source digest.
- The exact configuration-file bytes parsed by `load_config`, a separate
  construction-time file hash and comparison, and the effective configuration
  digest after in-memory/CLI overrides. A comment-only file edit affects the file
  hash without changing effective values.
- Selected legacy MobileNet definition/weights hashes and whether its execution
  is configured enabled. Missing, unreadable or changing files remain explicit;
  hashing does not invoke model inference.
- Process ID, UTC snapshot time, and the process-start observation made when the
  provenance module was imported. The latter is explicitly not the OS process
  creation timestamp. Schema version is independent of older package version
  constants.

Only identities are exposed: no configuration values, model bytes, environment,
absolute model/config paths or Git remote URLs. Failed file reads return an error
type without private path/error text. This diagnostic neither arms nor disables
the application and is not an attestation service. Unexpected diagnostic
failures produce `status: unavailable` with an error type and cannot abort
application construction or bypass its normal control lifecycle.

The source manifest records a set of file observations, not an atomic release
snapshot. It does not prove Python's already-loaded bytecode or native-library
identity. Git branch names alone never establish process bytes. A future
deployment should retain the intended release commit and expected config/model
hashes, start from a controlled immutable checkout, and compare the startup
diagnostic. No such deployment was performed during Phase 0/1.

## Runtime health decision

Existing `/api/health`, camera/motion status, classifier activity/progress ages,
and storage/recording failure/backpressure telemetry remain available. The pure
`runtime_health.py` evaluator and its 73 existing tests are retained.

Full evaluator integration is deliberately deferred. Its `ActiveWorkProgress`
contract needs the start of the current active/queued period to avoid treating
new work after a long idle interval as stalled. Current public statuses expose
several progress ages and active flags but not a consistent active-period start
for every required classifier/decision/writer/storage stage. Supplying zero,
startup time, or `expected=False` for missing stages would misrepresent health.
Correct integration needs producer instrumentation and a coherent composition
snapshot across those lifecycles; that belongs with later worker/recorder work.

No restart policy, automatic recovery, or hardware action has been added. The
existing `critical_worker_failure` behavior is unchanged. Later integration must
also account for startup, night-mode pause, legitimately idle workers, and
bounded queues independently from camera health.

## Camera publication generations

`FramePacket.generation`, `BufferedFrameInfo.generation`, and
`CameraStatus.frame_generation` identify a new capture handle's first successful
frame or a native frame-shape change. Source sequence continues across reconnect
and stop/start on the same `CameraService` object; receipt timestamps are not
sensor exposure times. Latest-frame reads and retained raw pre-roll preserve
the original identity, including a frame's original generation.

This is a provenance contract in Phase 1. Legacy annotations, target state and
recorders do not yet use generation to invalidate their state. Old pre-roll is
retained across transitions with its original shape/metadata. Generation-aware
consumer invalidation and mixed-resolution recording policy are explicit future
work; current calibrated frame-size checks remain intact. Do not interpret the
new field as complete stream-reset protection.
