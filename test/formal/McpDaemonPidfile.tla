-------------------------- MODULE McpDaemonPidfile --------------------------
EXTENDS Naturals, FiniteSets
\* Check:
\*   bunx everystate test/formal/McpDaemonPidfile.tla                       exit 0
\* Mutations:
\*   bunx everystate -config test/formal/McpDaemonPidfileUnsafe.cfg \
\*        test/formal/McpDaemonPidfile.tla                               exit 12
\*   bunx everystate -config test/formal/McpDaemonPidfileUnverified.cfg \
\*        test/formal/McpDaemonPidfile.tla                               exit 12
\*   bunx everystate -config test/formal/McpDaemonPidfileBlind.cfg \
\*        test/formal/McpDaemonPidfile.tla                               exit 12
\*   bunx everystate -config test/formal/McpDaemonPidfileTimeout.cfg \
\*        test/formal/McpDaemonPidfile.tla                               exit 12
\*
\* src/cli/mcp-pid.ts and the managed-daemon branch of src/cli/qmd.ts. Two
\* contenders (Procs), each a `qmd mcp --http --daemon` starter, its spawned
\* child (same name as its PID) and, once it serves, the daemon. One pidfile
\* (`pf`) persists across the episode; OldPid is a recycled PID left in a
\* pre-existing file. One `stop` caller is modelled by stopPc/stopPid.
\*
\* Action -> symbol:
\*   Spawn               qmd.ts daemon pre-check (readMcpDaemonPidState),
\*                       spawn() and QMD_MCP_DAEMON_CHILD=1
\*   ChildExecs          the child finishing exec: /proc/<pid>/cmdline now
\*                       looks like qmd (looksLikeQmdMcpCommand)
\*   ChildDiesBeforeExec spawn failure / exit before identity
\*   Claim               waitForDaemonChildIdentity + claimMcpDaemonPidfile:
\*                       BEGIN EXCLUSIVE, readMcpDaemonPidState,
\*                       publishPidfileAtomic
\*   StarterCrashes      the starter dies between spawn() and the claim
\*   Serve               waitForDaemonPidfileOwnership "owned", then
\*                       startMcpHttpServer binds the port
\*   Abandon             waitForDaemonPidfileOwnership "taken"/"timeout", or
\*                       the starter's child.kill after a refused claim
\*   DaemonExits         graceful SIGTERM/SIGINT teardown; the exit handler's
\*                       pidfileRecordsPid + removeMcpDaemonPidfileIfRecords
\*   DaemonCrashes       SIGKILL / crash: no exit handler, file left behind
\*   StopBegin           stopMcpDaemon's first readMcpDaemonPidState and
\*                       stopRunningDaemon's process.kill(SIGTERM)
\*   StopCleanup         removeStaleMcpDaemonPidfile: re-classify inside the
\*                       sidecar section, then unlink only if still stale
\*   StopFinish          stopRunningDaemon's poll sees the PID gone, then
\*                       removeMcpDaemonPidfileIfRecords
\*   StopTimeout         stopRunningDaemon's deadline -> {outcome:"timeout"}
\*
\* Promises:
\*   AtMostOneServing          at most one daemon has bound the port.
\*   AtMostOneOwner            at most one live claim exists.
\*   ClaimedOwnerIsNamed       every live claim is the PID the pidfile names,
\*                             so "Started (PID n)" is the daemon a later
\*                             stop/status will find.
\*   NoUnverifiedDeletion      no unlink was made on an earlier stale hint.
\*   NoUnverifiedSignal        no SIGTERM was sent to a PID the probe did not
\*                             classify as a live qmd process.
\*   TimeoutPreservesTracking  a stop that times out leaves the pidfile naming
\*                             the daemon it was waiting for.
\*
\* Assumptions:
\*   A1 Identity probe is sound: classifyProcessIdentity(pid) reports the
\*      PID's existence and command line for the whole transition in which it
\*      is used, and a PID is not reused between that probe and the signal or
\*      unlink it authorises. An unreadable cmdline is `not-qmd`, never
\*      `running`.
\*   A2 Same-version participants: every qmd process uses this sidecar/identity
\*      protocol; no older binary writes, moves or links the pidfile.
\*   A3 SQLite lock: BEGIN EXCLUSIVE on `<pidfile>.lock.sqlite` is atomic and
\*      mutually exclusive, is released by ROLLBACK/close or by process death,
\*      and the sidecar is never unlinked. Claim, StopCleanup, StopFinish and
\*      the exit unlink are therefore one atomic step each; a crashed holder
\*      leaves no lock state to recover. The sidecar itself is not a variable.
\*   A4 Publication is write-to-private-staging then rename; a lock-free
\*      reader sees a complete old file, a complete new file, or no file.
\*      `unreadable` is fail-safe and never repaired here.
\*   A5 spawn() returns before exec, so the child is `warming` (not-qmd) until
\*      ChildExecs. SIGTERM runs the exit handler; SIGKILL/crash does not. The
\*      exit unlink is best-effort: skipping it leaves a stale file that the
\*      next transition reclassifies.
\*   A6 stop waits outside the section, so the daemon's exit handler can take
\*      it; the wait observes the daemon leaving `qmd`.
\*   A7 One stop invocation per episode; each PID spawns once; the code's
\*      STOP_DECISION_ATTEMPTS loop is collapsed to "kept, decide again";
\*      EPERM/unsignalable and the identity-wait timeout are omitted.
\*
\* Excluded:
\*   - A stop whose lock-free first read happens before any starter has
\*     claimed: it can return not-running/cleaned-stale while a child that was
\*     spawned around the same time starts serving afterwards. The model
\*     reaches that state (the safety invariants still hold); it deliberately
\*     makes no liveness claim that such a stop stops that daemon. Stop/start
\*     ordering is the caller's, not the pidfile protocol's.
\*   - Mixed old-version binaries (see A2).
\*   - Port binding/EADDRINUSE details, log files, per-index path selection,
\*     status's separate call to removeStaleMcpDaemonPidfile, and multiple
\*     concurrent stops.
\* Bounds: Procs = {s1, s2}, one OldPid, one stop. A model-check pass covers
\* these bounds; it is not a proof of the code.

CONSTANTS
    Procs,          \* the two contenders (starter, child and daemon share a name)
    OldPid,         \* recycled PID recorded by a pre-existing pidfile
    IdentityGate,   \* CONTROL: claim publishes only an identified qmd child
    RecheckStale,   \* CONTROL: stale removal re-classifies in the section
    SignalGate,     \* CONTROL: stop signals only a classification `running`
    KeepTrackingOnTimeout  \* CONTROL: a timed-out stop leaves the pidfile

Pids == Procs \cup {OldPid}
Contents == {"absent", "malformed", "unreadable"} \cup Pids
Idents == {"none", "warming", "qmd", "foreign", "exited"}
StarterPhases == {"idle", "identity", "done", "abandoned", "crashed"}
ChildPhases == {"none", "warming", "waiting", "serving", "exited"}
StopPhases == {"idle", "cleanup", "waiting", "finished"}
Outcomes == {"none", "not-running", "unreadable", "cleaned-stale", "stopped", "timeout"}

VARIABLES
    pf,              \* pidfile: absent, malformed, unreadable, or a recorded PID
    ident,           \* per PID: process-table identity (never "running" for `foreign`)
    sp,              \* per contender: its starter's phase
    ch,              \* per contender: its child's phase
    serving,         \* daemons that bound the port
    granted,         \* claims whose starter was told it owns the pidfile
    stopPc,          \* the stop caller's phase
    stopPid,         \* PID the stop caller decided to wait for
    stopOutcome,     \* the stop caller's result
    timeoutKeptFile, \* did the pidfile still name stopPid when the deadline hit?
    unsafeDelete,    \* ghost: an unlink used a stale hint instead of a re-check
    unsafeSignal     \* ghost: SIGTERM went to a PID not classified `running`

vars == <<pf, ident, sp, ch, serving, granted, stopPc, stopPid, stopOutcome,
          timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* readMcpDaemonPidState + classifyProcessIdentity. `warming` (pre-exec) and
\* `foreign` (recycled PID) are both `not-qmd`, hence both `stale`; the starter
\* waits out the first and never signals either.
ClassifyPid(x) ==
    IF x \in {"absent", "malformed", "unreadable"} THEN x
    ELSE IF ident[x] = "qmd" THEN "running"
    ELSE "stale"

PidState == ClassifyPid(pf)

Init ==
    /\ pf \in {"absent", "malformed", "unreadable", OldPid}
    /\ ident = [p \in Pids |-> IF p = OldPid THEN "foreign" ELSE "none"]
    /\ sp = [p \in Procs |-> "idle"]
    /\ ch = [p \in Procs |-> "none"]
    /\ serving = {}
    /\ granted = {}
    /\ stopPc = "idle"
    /\ stopPid = "none"
    /\ stopOutcome = "none"
    /\ timeoutKeptFile = FALSE
    /\ unsafeDelete = FALSE
    /\ unsafeSignal = FALSE

\* qmd.ts daemon branch: the first readMcpDaemonPidState must not see a live
\* daemon or an unreadable file; then spawn(). Both starters can pass this
\* lock-free check before either claims.
Spawn(p) ==
    /\ sp[p] = "idle"
    /\ ch[p] = "none"
    /\ ident[p] = "none"
    /\ PidState # "running"
    /\ pf # "unreadable"
    /\ sp' = [sp EXCEPT ![p] = "identity"]
    /\ ch' = [ch EXCEPT ![p] = "warming"]
    /\ ident' = [ident EXCEPT ![p] = "warming"]
    /\ UNCHANGED <<pf, serving, granted, stopPc, stopPid, stopOutcome,
                   timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* The child exec'd. Its cmdline now identifies qmd and, because the parent
\* spawned it with QMD_MCP_DAEMON_CHILD=1, it enters
\* waitForDaemonPidfileOwnership; whether the claim has happened yet is
\* decided by the interleaving, so the wait starts independently of `sp`.
ChildExecs(p) ==
    /\ ch[p] = "warming"
    /\ ident[p] = "warming"
    /\ ident' = [ident EXCEPT ![p] = "qmd"]
    /\ ch' = [ch EXCEPT ![p] = "waiting"]
    /\ UNCHANGED <<pf, sp, serving, granted, stopPc, stopPid, stopOutcome,
                   timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* Failed exec, or the child exited before it could identify itself. The
\* starter's identity wait observes hasExited and reports failure. A pre-exec
\* child has no exit handler, so the file is left as it was.
ChildDiesBeforeExec(p) ==
    /\ ch[p] = "warming"
    /\ ident' = [ident EXCEPT ![p] = "exited"]
    /\ ch' = [ch EXCEPT ![p] = "exited"]
    /\ granted' = granted \ {p}
    /\ sp' = [sp EXCEPT ![p] = IF sp[p] = "identity" THEN "abandoned" ELSE sp[p]]
    /\ UNCHANGED <<pf, serving, stopPc, stopPid, stopOutcome,
                   timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* waitForDaemonChildIdentity succeeded, then claimMcpDaemonPidfile takes the
\* sidecar section, re-classifies the file and publishes this PID (staging +
\* rename). `running` means another identified daemon owns the file: the
\* starter must give its child up. `unreadable` fails safe. Every other state
\* (absent/malformed/stale) may be replaced. IdentityGate=FALSE is the ablation
\* that publishes the PID while it is still `warming` (not-qmd).
Claim(p) ==
    /\ sp[p] = "identity"
    /\ ident[p] \in {"warming", "qmd"}
    /\ (IdentityGate => ident[p] = "qmd")
    /\ \/ /\ PidState = "running"
          /\ pf # p
          /\ sp' = [sp EXCEPT ![p] = "abandoned"]
          /\ UNCHANGED <<pf, ident, ch, serving, granted, stopPc, stopPid,
                         stopOutcome, timeoutKeptFile, unsafeDelete, unsafeSignal>>
       \/ /\ PidState = "unreadable"
          /\ sp' = [sp EXCEPT ![p] = "abandoned"]
          /\ UNCHANGED <<pf, ident, ch, serving, granted, stopPc, stopPid,
                         stopOutcome, timeoutKeptFile, unsafeDelete, unsafeSignal>>
       \/ /\ PidState \in {"absent", "malformed", "stale"}
          /\ pf' = p
          /\ granted' = granted \cup {p}
          /\ sp' = [sp EXCEPT ![p] = "done"]
          /\ UNCHANGED <<ident, ch, serving, stopPc, stopPid, stopOutcome,
                         timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* The starter dies before it can claim (the detached child survives).
StarterCrashes(p) ==
    /\ sp[p] = "identity"
    /\ sp' = [sp EXCEPT ![p] = "crashed"]
    /\ UNCHANGED <<pf, ident, ch, serving, granted, stopPc, stopPid,
                   stopOutcome, timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* waitForDaemonPidfileOwnership returns "owned": the file records this PID,
\* so the child may bind the port. The pidfile is the ownership token.
Serve(p) ==
    /\ ch[p] = "waiting"
    /\ pf = p
    /\ ident[p] = "qmd"
    /\ ch' = [ch EXCEPT ![p] = "serving"]
    /\ serving' = serving \cup {p}
    /\ UNCHANGED <<pf, ident, sp, granted, stopPc, stopPid, stopOutcome,
                   timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* The wait sees another live daemon ("taken"), times out, or the starter
\* killed this child after its own claim was refused. A child that was never
\* named never serves.
Abandon(p) ==
    /\ ch[p] = "waiting"
    /\ sp[p] \in {"done", "crashed", "abandoned"}
    /\ pf # p
    /\ ident' = [ident EXCEPT ![p] = "exited"]
    /\ ch' = [ch EXCEPT ![p] = "exited"]
    /\ granted' = granted \ {p}
    /\ UNCHANGED <<pf, sp, serving, stopPc, stopPid, stopOutcome,
                   timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* Graceful exit: termination tears the server down, then process.on("exit")
\* calls pidfileRecordsPid and removeMcpDaemonPidfileIfRecords with a zero
\* section budget. The unlink is checked against the PID it records; another
\* daemon's file is untouched.
DaemonExits(p) ==
    /\ ch[p] \in {"waiting", "serving"}
    /\ ident' = [ident EXCEPT ![p] = "exited"]
    /\ ch' = [ch EXCEPT ![p] = "exited"]
    /\ serving' = serving \ {p}
    /\ granted' = granted \ {p}
    /\ pf' = IF pf = p THEN "absent" ELSE pf
    /\ sp' = [sp EXCEPT ![p] = IF sp[p] = "identity" THEN "abandoned" ELSE sp[p]]
    /\ UNCHANGED <<stopPc, stopPid, stopOutcome, timeoutKeptFile,
                   unsafeDelete, unsafeSignal>>

\* SIGKILL / crash: the kernel drops the SQLite lock but no exit handler runs,
\* so the pidfile stays and becomes stale. Recovery is the next Claim or
\* StopCleanup reclassifying it. The port socket dies with the process.
DaemonCrashes(p) ==
    /\ ch[p] \in {"waiting", "serving"}
    /\ ident' = [ident EXCEPT ![p] = "exited"]
    /\ ch' = [ch EXCEPT ![p] = "exited"]
    /\ serving' = serving \ {p}
    /\ granted' = granted \ {p}
    /\ sp' = [sp EXCEPT ![p] = IF sp[p] = "identity" THEN "abandoned" ELSE sp[p]]
    /\ UNCHANGED <<pf, stopPc, stopPid, stopOutcome, timeoutKeptFile,
                   unsafeDelete, unsafeSignal>>

\* The stop caller's first, lock-free read, followed by
\* stopRunningDaemon's SIGTERM. `running` is the only state it signals; stale
\* and malformed go to the re-checking cleanup; unreadable is left alone.
\* SignalGate=FALSE is the ablation that signals whatever PID the file records
\* (a recycled PID or a pre-exec child included).
StopBegin ==
    /\ stopPc = "idle"
    /\ \/ /\ PidState = "absent"
          /\ stopOutcome' = "not-running"
          /\ stopPc' = "finished"
          /\ UNCHANGED <<stopPid, unsafeSignal>>
       \/ /\ PidState = "unreadable"
          /\ stopOutcome' = "unreadable"
          /\ stopPc' = "finished"
          /\ UNCHANGED <<stopPid, unsafeSignal>>
       \/ /\ PidState = "malformed"
          /\ stopPc' = "cleanup"
          /\ UNCHANGED <<stopPid, stopOutcome, unsafeSignal>>
       \/ /\ PidState = "stale" /\ SignalGate
          /\ stopPc' = "cleanup"
          /\ UNCHANGED <<stopPid, stopOutcome, unsafeSignal>>
       \/ /\ PidState = "running"
          /\ stopPid' = pf
          /\ stopPc' = "waiting"
          /\ unsafeSignal' = (unsafeSignal \/ ident[pf] # "qmd")
          /\ UNCHANGED stopOutcome
       \/ /\ PidState = "stale" /\ ~SignalGate
          /\ stopPid' = pf
          /\ stopPc' = "waiting"
          /\ unsafeSignal' = TRUE
          /\ UNCHANGED stopOutcome
    /\ UNCHANGED <<pf, ident, sp, ch, serving, granted, timeoutKeptFile,
                   unsafeDelete>>

\* removeStaleMcpDaemonPidfile: the earlier classification was only a hint.
\* With RecheckStale it re-classifies inside the section and either unlinks a
\* still stale/malformed file or reports "kept" so the stop loop decides again
\* from the new contents. RecheckStale=FALSE is the "the caller already
\* looked" ablation: it unlinks whatever is there now.
StopCleanup ==
    /\ stopPc = "cleanup"
    /\ \/ /\ RecheckStale /\ PidState \in {"stale", "malformed"}
          /\ pf' = "absent"
          /\ stopOutcome' = "cleaned-stale"
          /\ stopPc' = "finished"
          /\ UNCHANGED <<stopPid, unsafeDelete>>
       \/ /\ RecheckStale /\ PidState \notin {"stale", "malformed"}
          /\ stopPc' = "idle"
          /\ UNCHANGED <<pf, stopOutcome, stopPid, unsafeDelete>>
       \/ /\ ~RecheckStale
          /\ pf' = "absent"
          /\ stopOutcome' = "cleaned-stale"
          /\ stopPc' = "finished"
          /\ unsafeDelete' = (unsafeDelete \/ PidState \in {"running", "unreadable"})
          /\ UNCHANGED stopPid
    /\ UNCHANGED <<ident, sp, ch, serving, granted, timeoutKeptFile, unsafeSignal>>

\* The poll loop sees the signalled PID leave `qmd`. The daemon's own exit
\* handler normally unlinked the file already; removeMcpDaemonPidfileIfRecords
\* re-checks that it still records that PID before removing it, so a file
\* another daemon republished is left alone.
StopFinish ==
    /\ stopPc = "waiting"
    /\ stopOutcome = "none"
    /\ ident[stopPid] = "exited"
    /\ \/ /\ pf = stopPid /\ pf' = "absent"
       \/ /\ pf # stopPid /\ pf' = pf
    /\ stopOutcome' = "stopped"
    /\ stopPc' = "finished"
    /\ UNCHANGED <<ident, sp, ch, serving, granted, stopPid,
                   timeoutKeptFile, unsafeDelete, unsafeSignal>>

\* The deadline expires while the PID is still alive and qmd. The pidfile is
\* deliberately kept: the daemon remains tracked and `qmd mcp stop` can be
\* retried, instead of the process being orphaned as untracked.
\* KeepTrackingOnTimeout=FALSE is the "the deadline failed, so drop the file"
\* ablation; it leaves a live, serving daemon unnamed.
StopTimeout ==
    /\ stopPc = "waiting"
    /\ stopOutcome = "none"
    /\ ident[stopPid] = "qmd"
    /\ pf' = IF KeepTrackingOnTimeout THEN pf ELSE "absent"
    /\ stopOutcome' = "timeout"
    /\ stopPc' = "finished"
    /\ timeoutKeptFile' = (KeepTrackingOnTimeout /\ pf = stopPid)
    /\ UNCHANGED <<ident, sp, ch, serving, granted, stopPid,
                   unsafeDelete, unsafeSignal>>

Next ==
    \/ \E p \in Procs :
           Spawn(p) \/ ChildExecs(p) \/ ChildDiesBeforeExec(p) \/ Claim(p)
           \/ StarterCrashes(p) \/ Serve(p) \/ Abandon(p)
           \/ DaemonExits(p) \/ DaemonCrashes(p)
    \/ StopBegin \/ StopCleanup \/ StopFinish \/ StopTimeout

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pf \in Contents
    /\ ident \in [Pids -> Idents]
    /\ sp \in [Procs -> StarterPhases]
    /\ ch \in [Procs -> ChildPhases]
    /\ serving \subseteq Procs
    /\ granted \subseteq Procs
    /\ stopPc \in StopPhases
    /\ stopPid \in {"none"} \cup Pids
    /\ stopOutcome \in Outcomes
    /\ timeoutKeptFile \in BOOLEAN
    /\ unsafeDelete \in BOOLEAN
    /\ unsafeSignal \in BOOLEAN

\* Until the file changes, the child that bound the port is the file's PID.
AtMostOneServing == Cardinality(serving) <= 1

\* A claim is a live ownership token: replacing a stale file is fine, adding a
\* second live claim is not.
AtMostOneOwner == Cardinality(granted) <= 1

\* Every granted starter is still discoverable by stop/status.
ClaimedOwnerIsNamed == \A p \in granted : pf = p

NoUnverifiedDeletion == ~unsafeDelete
NoUnverifiedSignal == ~unsafeSignal
TimeoutPreservesTracking == stopOutcome # "timeout" \/ timeoutKeptFile
=============================================================================
