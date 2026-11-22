-------------------------- MODULE ServerShutdown ---------------------------
EXTENDS Naturals
\* Shipped default (5 s deadline, reported timeout, safe background cleanup,
\* standalone CLI exits non-zero):
\*   bunx everystate -config test/formal/ServerShutdown.cfg \
\*        test/formal/ServerShutdown.tla                               exit 0
\* Embedded host (same deadline, but the host is never exited):
\*   bunx everystate -config test/formal/ServerShutdownEmbedded.cfg \
\*        test/formal/ServerShutdown.tla                               exit 0
\* Mutant: the deadline tears the store down instead of abandoning safely:
\*   bunx everystate -config test/formal/ServerShutdownForced.cfg \
\*        test/formal/ServerShutdown.tla                               exit 12
\* Pre-fix order (transport closed before the drain):
\*   bunx everystate -config test/formal/ServerShutdownPreFix.cfg \
\*        test/formal/ServerShutdown.tla                               exit 12
\* Legacy (before the lifecycle refactor):
\*   bunx everystate -config test/formal/ServerShutdownLegacy.cfg \
\*        test/formal/ServerShutdown.tla                               exit 12
\*
\* src/mcp/server.ts: shutdown of the MCP transports racing one admitted
\* request. The shipped code (createMcpLifecycle + runMcpTeardown, used by
\* startMcpHttpServer.stop and, in its own order, registerStdioEofShutdown)
\* closes admission, stops the listener, drains admitted work until ONE
\* deadline shared by every step (MCP_SHUTDOWN_TIMEOUT_MS = 5000 by default),
\* then closes the MCP transport, finishes the listener, closes the store and
\* only then marks the lifecycle `stopped`. A second stop() awaits the same
\* memoised promise.
\*
\* Two outcomes exist at the deadline, and the model keeps them apart:
\*
\*   reported  -- stop() rejects with McpShutdownTimeoutError; the admitted
\*                requests are aborted and their sockets severed. This is a
\*                *report*, not a teardown.
\*   abandoned -- the cleanup that could not finish in time keeps running in
\*                the background in the SAME safe order: unbounded wait for
\*                idle, transport close, listener close, store close, then
\*                `draining -> stopped`. Nothing is closed underneath live
\*                work; `closed` resolves only after the store closed.
\*
\* The standalone CLI exits 1 immediately on the report (EarlyExitOnTimeout):
\* work remains and the process is the only thing that can end it. That exit is
\* modelled separately and does NOT claim a clean teardown. An embedded host
\* (EarlyExitOnTimeout = FALSE) keeps running and awaits `closed`.
\*
\* Promises this module checks:
\*   NoTransportCloseUnderHandler: no admitted request is still executing when
\*     the MCP transport closes. mcpHandler.close() aborts the exchanges it
\*     still tracks, so closing it before the drain turns an in-flight /mcp
\*     request into a 500 ("handler has been closed").
\*   NoStoreCloseUnderHandler: no admitted request is still executing when the
\*     store starts closing -- in the grace window AND after a reported
\*     timeout. The deadline may abort work, but it may not close the store
\*     under it; the background cleanup waits for idle first.
\*   StoppedImpliesStoreClosed: `stopped` (adm) is claimed only once
\*     store.close() actually resolved.
\*   ReportedTimeoutImpliesAbandoned: a reported timeout is a background
\*     cleanup, not a teardown.
\*   CleanExitImpliesTeardown: exit code 0 implies the listener and the store
\*     are closed.
\*   TimeoutExitImpliesWork: exit code 1 is only the standalone early exit
\*     after a reported timeout, never a skipped clean shutdown.
\*
\* Negative controls assert the concrete safety properties, not
\* SaneShutdownOrder: that operator is a meta-ordering sanity check for the
\* shipped sequence, so asserting it in a control would mask the real safety
\* failure the control exists to witness. ServerShutdownForced.cfg therefore
\* fails NoStoreCloseUnderHandler and ServerShutdownLegacy.cfg fails whichever
\* of NoStoreCloseUnderHandler / NoTransportCloseUnderHandler /
\* CleanExitImpliesTeardown the exploration reaches first. TypeOK stays on
\* everywhere.
\*
\* Assumptions: httpServer.close(cb) fires once every connection ended;
\* closeIdleConnections() ends sockets with nothing left to write and leaves an
\* unfinished response alone; closeAllConnections() destroys the rest so that
\* callback arrives; aborting an admitted request releases it (the web Request
\* signal closes the SDK exchange and cancels the REST search); a store call
\* that resumes on a closed db / disposed llama.cpp rejects. Excluded: the /mcp
\* route's 400/403/404 branches, response bodies, a throwing store.close()
\* (the coordinator records it and stops claiming `stopped`), SIGKILL, and the
\* *content* of a refused response. The report is not modelled as a reply to a
\* client -- the caller of stop() is the process itself.
\* Bounds: 1 REST/MCP request, the 2 signal handlers the standalone entry
\* registers.

CONSTANTS
    Reqs,                 \* requests this server admits
    Callers,              \* signal handlers that call stop(): {"sigterm","sigint"}
    AdmissionGate,        \* current: lifecycle.admit() at entry, 503 once draining
    SharedStop,           \* current: stop() memoises one teardown promise
    DrainAtAll,           \* current: wait for admitted handlers before teardown
    BoundedDrain,         \* current: one deadline bounds the whole teardown
    AbandonAtDeadline,    \* current: report + finish in the background, never
                          \* close the store under live work
    CloseTransportAtStop, \* TRUE = PRE-FIX order: mcpHandler.close() is issued
                          \* before the drain instead of after it
    EarlyExitOnTimeout    \* TRUE = standalone CLI exits 1 at the report;
                          \* FALSE = embedded host keeps running

VARIABLES
    proc,                 \* "running", then "exited" for good
    exitCode,             \* "none" until exit: "clean" (0) or "abandoned" (1)
    adm,                  \* lifecycle: "serving" -> "draining" -> "stopped"
    pc,                   \* per request: where the HTTP handler is
    srv,                  \* node listener: "listening" -> "closing" -> "closed"
    handlerSt,            \* mcpHandler.close(): "open" -> "closing" -> "closed"
    storeSt,              \* store.close(): "open" -> "disposing" -> "closed"
    deadline,             \* the shared timer fired
    abandoned,            \* the report happened and cleanup continues behind it
    stopOutcome,          \* the promise stop() hands its caller
    stage,                \* where the one teardown chain is
    spc                   \* per signal handler: idle / in stop / waiting / exiting

vars == <<proc, exitCode, adm, pc, srv, handlerSt, storeSt,
          deadline, abandoned, stopOutcome, stage, spc>>

\* An admitted request is executing (or about to execute) store work: exactly
\* the phases in which closing its transport or the store is unsafe.
ActiveRequest(r) == pc[r] \in {"handler", "body", "searching"}

NoActiveRequest == \A r \in Reqs : ~ActiveRequest(r)
SomeActiveRequest == \E r \in Reqs : ActiveRequest(r)

Init ==
    /\ proc = "running"
    /\ exitCode = "none"
    /\ adm = "serving"
    /\ srv = "listening"
    /\ handlerSt = "open"
    /\ storeSt = "open"
    /\ deadline = FALSE
    /\ abandoned = FALSE
    /\ stopOutcome = "pending"
    /\ stage = "idle"
    /\ spc = [s \in Callers |-> "idle"]
    /\ pc = [r \in Reqs |-> "none"]

\* node dispatches the handler; the admission gate runs before the origin
\* screen and route branch, which are synchronous with it. A socket the
\* listener accepted before it closed can still deliver a request ("closing").
Arrive(r) ==
    /\ proc = "running"
    /\ srv \in {"listening", "closing"}
    /\ pc[r] = "none"
    /\ (AdmissionGate => adm = "serving")
    /\ pc' = [pc EXCEPT ![r] = "handler"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* Admission refused: 503 + Connection: close, so the store is never touched.
Refuse(r) ==
    /\ proc = "running"
    /\ srv \in {"listening", "closing"}
    /\ AdmissionGate
    /\ adm # "serving"
    /\ pc[r] = "none"
    /\ pc' = [pc EXCEPT ![r] = "refused"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* `await collectBody(nodeReq)` resumes; validation up to store.search entry is
\* synchronous.
ReadBody(r) ==
    /\ proc = "running"
    /\ pc[r] = "handler"
    /\ pc' = [pc EXCEPT ![r] = "body"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* store.search is entered with the store still open.
BeginSearch(r) ==
    /\ proc = "running"
    /\ pc[r] = "body"
    /\ storeSt = "open"
    /\ pc' = [pc EXCEPT ![r] = "searching"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* store.search is entered after the store already left "open": it does not
\* complete and the handler answers 500 (guarded by headersSent).
SearchRejects(r) ==
    /\ proc = "running"
    /\ pc[r] = "body"
    /\ storeSt # "open"
    /\ pc' = [pc EXCEPT ![r] = "failed"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* The awaited search resolves and the response is written: the handler's work
\* (the admission ticket) is done, whether or not the bytes have flushed.
SearchDone(r) ==
    /\ proc = "running"
    /\ pc[r] = "searching"
    /\ storeSt = "open"
    /\ pc' = [pc EXCEPT ![r] = "responded"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* The awaited search does not complete because the store left "open" while it
\* ran; the handler answers 500.
SearchFails(r) ==
    /\ proc = "running"
    /\ pc[r] = "searching"
    /\ storeSt # "open"
    /\ pc' = [pc EXCEPT ![r] = "failed"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* The client is gone or the deadline aborted this request: its response can
\* never be delivered, so the admission ticket is released. This is what lets an
\* abandoned cleanup reach idle without closing the store under live work.
AbandonWork(r) ==
    /\ proc = "running"
    /\ abandoned
    /\ ActiveRequest(r)
    /\ pc' = [pc EXCEPT ![r] = "failed"]
    /\ UNCHANGED <<proc, exitCode, adm, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* First signal: admission closes, then either the listener (shipped) or the
\* MCP transport (pre-fix) is closed first. Legacy has neither a gate nor a
\* drain: it goes straight at the store once the transport closes.
EnterStop(s) ==
    /\ proc = "running"
    /\ spc[s] = "idle"
    /\ stage = "idle"
    /\ spc' = [spc EXCEPT ![s] = "inStop"]
    /\ adm' = IF AdmissionGate THEN "draining" ELSE adm
    /\ \/ /\ CloseTransportAtStop        \* PRE-FIX: transport torn down first
          /\ handlerSt' = "closing"
          /\ stage' = "transport"
          /\ UNCHANGED <<srv, storeSt>>
       \/ /\ DrainAtAll /\ ~CloseTransportAtStop   \* SHIPPED: gate + listener first
          /\ srv' = "closing"
          /\ stage' = "drain"
          /\ UNCHANGED <<handlerSt, storeSt>>
       \/ /\ ~DrainAtAll                  \* LEGACY: no drain, straight to store
          /\ srv' = "closing"
          /\ stage' = "store"
          /\ storeSt' = "disposing"
          /\ UNCHANGED handlerSt
    /\ UNCHANGED <<proc, exitCode, pc, deadline, abandoned, stopOutcome>>

\* A later signal. Legacy: `if (stopping) return` and that handler proceeds to
\* its own process.exit(0). Current: it awaits the memoised promise.
LaterSignal(s) ==
    /\ proc = "running"
    /\ spc[s] = "idle"
    /\ stage # "idle"
    /\ \/ /\ SharedStop
          /\ \/ /\ stopOutcome # "pending"
                /\ spc' = [spc EXCEPT ![s] = "exit"]
             \/ /\ stopOutcome = "pending"
                /\ spc' = [spc EXCEPT ![s] = "waitShared"]
       \/ /\ ~SharedStop
          /\ spc' = [spc EXCEPT ![s] = "returned"]
    /\ UNCHANGED <<proc, exitCode, adm, pc, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage>>

\* The drain resolved TRUE: every admitted handler released, so the transport
\* (and later the store) close with nothing running through them. The pre-fix
\* order has already closed the transport, so it resumes at the listener.
DrainIdle ==
    /\ proc = "running"
    /\ DrainAtAll
    /\ stage = "drain"
    /\ NoActiveRequest
    /\ \/ /\ handlerSt = "open"          \* shipped: close the transport next
          /\ stage' = "transport"
          /\ handlerSt' = "closing"
       \/ /\ handlerSt # "open"          \* pre-fix: transport already gone
          /\ stage' = "listener"
          /\ UNCHANGED handlerSt
    /\ UNCHANGED <<proc, exitCode, adm, pc, srv, storeSt,
                    deadline, abandoned, stopOutcome, spc>>

\* The one deadline expired with an admitted request still active. stop()
\* reports it now (stopOutcome = "timeout").
DrainDeadline ==
    /\ proc = "running"
    /\ DrainAtAll
    /\ BoundedDrain
    /\ stage = "drain"
    /\ SomeActiveRequest
    /\ deadline' = TRUE
    /\ stopOutcome' = "timeout"
    /\ \/ /\ AbandonAtDeadline        \* SHIPPED: report, then wait for work
          /\ abandoned' = TRUE
          /\ stage' = "reportWait"
          /\ UNCHANGED <<adm, srv, handlerSt, storeSt>>
       \/ /\ ~AbandonAtDeadline       \* MUTANT: close the store under the work
          /\ abandoned' = FALSE
          /\ stage' = "storeUnsafe"
          /\ storeSt' = "disposing"
          /\ UNCHANGED <<adm, srv, handlerSt>>
    /\ UNCHANGED <<proc, exitCode, pc, spc>>

\* Abandoned cleanup, step 1: the unbounded wait for idle. The store is still
\* open, and `adm` is still only draining.
ReportWaitIdle ==
    /\ proc = "running"
    /\ abandoned
    /\ stage = "reportWait"
    /\ NoActiveRequest
    /\ \/ /\ handlerSt = "open"
          /\ stage' = "transport"
          /\ handlerSt' = "closing"
       \/ /\ handlerSt # "open"
          /\ stage' = "listener"
          /\ UNCHANGED handlerSt
    /\ UNCHANGED <<proc, exitCode, adm, pc, srv, storeSt,
                    deadline, abandoned, stopOutcome, spc>>

\* `await mcpHandler.close()` resumes. Shipped: the drain already ran, so the
\* listener is next. Pre-fix: the drain runs only now, with the transport
\* already torn down under the admitted request.
TransportClosed ==
    /\ proc = "running"
    /\ stage = "transport"
    /\ handlerSt = "closing"
    /\ handlerSt' = "closed"
    /\ srv' = "closing"
    /\ stage' = IF DrainAtAll /\ CloseTransportAtStop THEN "drain" ELSE "listener"
    /\ UNCHANGED <<proc, exitCode, adm, pc, storeSt,
                    deadline, abandoned, stopOutcome, spc>>

\* closeIdleConnections() plus closeAllConnections() after a report: the
\* listener's 'close' callback fires.
ListenerClosed ==
    /\ proc = "running"
    /\ stage = "listener"
    /\ srv = "closing"
    /\ srv' = "closed"
    /\ stage' = "store"
    /\ storeSt' = "disposing"
    /\ UNCHANGED <<proc, exitCode, adm, pc, handlerSt,
                    deadline, abandoned, stopOutcome, spc>>

\* Legacy only: the node listener finishes on its own, after the store is
\* already closing.
SocketsDrained ==
    /\ proc = "running"
    /\ ~DrainAtAll
    /\ srv = "closing"
    /\ srv' = "closed"
    /\ UNCHANGED <<proc, exitCode, adm, pc, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

\* store.close() resolved: the store and the LLM it owns are gone. Only now may
\* the lifecycle become "stopped", the stop promise resolve cleanly, and the
\* signal handlers proceed to exit 0.
StoreClosed ==
    /\ proc = "running"
    /\ stage = "store"
    /\ storeSt = "disposing"
    /\ storeSt' = "closed"
    /\ stage' = "done"
    /\ adm' = IF AdmissionGate THEN "stopped" ELSE adm
    /\ stopOutcome' = IF stopOutcome = "pending" THEN "clean" ELSE stopOutcome
    /\ spc' = [s \in Callers |->
                 IF spc[s] \in {"inStop", "waitShared"} THEN "exit" ELSE spc[s]]
    /\ UNCHANGED <<proc, exitCode, pc, srv, handlerSt,
                    deadline, abandoned>>

\* The mutant's continuation: the store finishes closing even though admitted
\* work was running through it. Kept so the unsafe branch is not a dead end.
StoreClosedUnsafe ==
    /\ proc = "running"
    /\ stage = "storeUnsafe"
    /\ storeSt = "disposing"
    /\ storeSt' = "closed"
    /\ stage' = "done"
    /\ adm' = IF AdmissionGate THEN "stopped" ELSE adm
    /\ UNCHANGED <<proc, exitCode, pc, srv, handlerSt,
                    deadline, abandoned, stopOutcome, spc>>

\* The process ends. A clean stop exits 0 ("clean") only after the teardown
\* finished; the standalone entry exits 1 ("abandoned") immediately after a
\* reported timeout (work remains, and only the process can end it); legacy's
\* duplicate signal exits 0 with teardown still pending -- the bug this model
\* pins.
Exit(s) ==
    /\ proc = "running"
    /\ spc[s] # "idle"
    /\ \/ /\ stopOutcome = "clean"
          /\ stage = "done"
          /\ exitCode' = "clean"
       \/ /\ stopOutcome = "timeout"
          /\ EarlyExitOnTimeout
          /\ exitCode' = "abandoned"
       \/ /\ ~SharedStop
          /\ spc[s] = "returned"
          /\ exitCode' = "clean"
    /\ proc' = "exited"
    /\ UNCHANGED <<adm, pc, srv, handlerSt, storeSt,
                    deadline, abandoned, stopOutcome, stage, spc>>

ReqStep(r) == Arrive(r) \/ Refuse(r) \/ ReadBody(r) \/ BeginSearch(r) \/
             SearchRejects(r) \/ SearchDone(r) \/ SearchFails(r) \/ AbandonWork(r)

SigStep(s) == EnterStop(s) \/ LaterSignal(s) \/ Exit(s)

Next ==
    \/ \E r \in Reqs : ReqStep(r)
    \/ \E s \in Callers : SigStep(s)
    \/ DrainIdle \/ DrainDeadline \/ ReportWaitIdle \/ TransportClosed \/
       ListenerClosed \/ SocketsDrained \/ StoreClosed \/ StoreClosedUnsafe

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ proc \in {"running", "exited"}
    /\ exitCode \in {"none", "clean", "abandoned"}
    /\ adm \in {"serving", "draining", "stopped"}
    /\ srv \in {"listening", "closing", "closed"}
    /\ handlerSt \in {"open", "closing", "closed"}
    /\ storeSt \in {"open", "disposing", "closed"}
    /\ deadline \in BOOLEAN
    /\ abandoned \in BOOLEAN
    /\ stopOutcome \in {"pending", "clean", "timeout"}
    /\ stage \in {"idle", "drain", "reportWait", "transport",
                  "listener", "store", "storeUnsafe", "done"}
    /\ spc \in [Callers -> {"idle", "inStop", "waitShared", "returned", "exit"}]
    /\ pc \in [Reqs -> {"none", "handler", "body", "searching",
                        "responded", "failed", "refused"}]
    /\ stage = "idle" <=> (adm = "serving" /\ srv = "listening"
                            /\ handlerSt = "open" /\ storeSt = "open")

\* Sanity of the modelled ordering (catches a model that reorders teardown
\* itself instead of letting the code's order produce a race).
SaneShutdownOrder ==
    /\ (storeSt # "open") => (handlerSt = "closed")
    /\ (handlerSt = "closed") => (srv # "listening")
    /\ (storeSt = "closed") => (srv = "closed")
    /\ (adm = "stopped") => (storeSt = "closed")
    \* admission closes before any teardown step runs
    /\ ((stage # "idle") /\ AdmissionGate) => (adm # "serving")

\* The transport is not torn down under an admitted request. mcpHandler.close()
\* aborts the exchanges it still tracks, so issuing it while a handler runs
\* fails that request: the pre-fix order (ServerShutdownPreFix.cfg), the legacy
\* code (ServerShutdownLegacy.cfg) and the store-at-deadline mutant
\* (ServerShutdownForced.cfg) do that by construction.
NoTransportCloseUnderHandler ==
    (handlerSt # "open") => NoActiveRequest

\* The promise: no admitted handler is still executing once the store starts
\* closing -- in the grace window and after a reported timeout alike. The
\* deadline may abort the work, but it must not close the store underneath it.
NoStoreCloseUnderHandler ==
    (storeSt # "open") => NoActiveRequest

\* `stopped` is claimed only once the store actually closed. A reported timeout
\* therefore leaves the lifecycle draining for as long as the abandoned cleanup
\* needs; it never claims a teardown that did not happen.
StoppedImpliesStoreClosed ==
    (adm = "stopped") => (storeSt = "closed")

\* A reported timeout is a report plus a background cleanup, never a teardown:
\* the shipped branch must have abandoned the work, not closed under it.
ReportedTimeoutImpliesAbandoned ==
    (stopOutcome = "timeout" /\ AbandonAtDeadline) => (deadline /\ abandoned)

\* Exit is the last step of a finished teardown, not a way to skip one.
CleanExitImpliesTeardown ==
    (proc = "exited" /\ exitCode = "clean") =>
        (stage = "done" /\ storeSt = "closed" /\ srv = "closed")

\* The "abandoned" exit is the standalone entry abandoning a reported timeout:
\* work remains, so the process must not pretend the teardown finished.
TimeoutExitImpliesWork ==
    (proc = "exited" /\ exitCode = "abandoned") =>
        (stopOutcome = "timeout" /\ EarlyExitOnTimeout)

=============================================================================
