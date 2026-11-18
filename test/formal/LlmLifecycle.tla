-------------------------- MODULE LlmLifecycle --------------------------
EXTENDS Naturals
\* Check: bunx everystate test/formal/LlmLifecycle.tla
\* Mutation: bunx everystate -config test/formal/LlmLifecycleBroken.cfg test/formal/LlmLifecycle.tla
\* One LlamaCpp instance, one context, one session and one operation. The
\* counters are bounded to 0/1. Acquisition and lifecycle transitions in
\* LlmResourceLifecycle are synchronous; pool disposal itself may await.
\* EnforceLeases=FALSE deliberately removes the unload/drain guards and the
\* shared-disposal-completion barrier; it is not the current implementation.
\* Excludes native failures, per-resource terminal timeouts, model loading,
\* and whether a parked operation or native disposer eventually completes.
CONSTANT EnforceLeases
VARIABLES state, pool, session, operation, unloading, unsafe, secondReturned
vars == <<state, pool, session, operation, unloading, unsafe, secondReturned>>

Init == /\ state = "ready" /\ pool = "available"
        /\ session = FALSE /\ operation = FALSE /\ unloading = FALSE
        /\ unsafe = FALSE /\ secondReturned = FALSE

\* acquireLease: session and operation admissions each commit before an await.
StartSession == /\ state = "ready" /\ ~session
                /\ session' = TRUE
                /\ UNCHANGED <<state, pool, operation, unloading, unsafe, secondReturned>>
EndSession == /\ session
              /\ session' = FALSE
              /\ UNCHANGED <<state, pool, operation, unloading, unsafe, secondReturned>>
StartOperation == /\ state = "ready" /\ pool = "available" /\ ~operation
                  /\ operation' = TRUE
                  /\ UNCHANGED <<state, pool, session, unloading, unsafe, secondReturned>>
EndOperation == /\ operation
                /\ operation' = FALSE
                /\ UNCHANGED <<state, pool, session, unloading, unsafe, secondReturned>>

\* Idle timeout: canUnload, drainContextPools, then await actual disposal.
BeginUnload == /\ state = "ready" /\ pool = "available"
               /\ (~EnforceLeases \/ (~session /\ ~operation))
               /\ state' = "unloading" /\ pool' = "disposing" /\ unloading' = TRUE
               /\ UNCHANGED <<session, operation, unsafe, secondReturned>>
FinishUnload == /\ unloading
                /\ unloading' = FALSE /\ pool' = "absent"
                /\ unsafe' = (unsafe \/ operation)
                /\ state' = IF state = "unloading" THEN "ready" ELSE state
                /\ UNCHANGED <<session, operation, secondReturned>>
LoadContext == /\ state = "ready" /\ pool = "absent"
               /\ pool' = "available"
               /\ UNCHANGED <<state, session, operation, unloading, unsafe, secondReturned>>

\* dispose synchronously closes admission; final teardown waits for operations
\* and an already-started idle unload. Other callers await finishDispose.
BeginDispose == /\ state \in {"ready", "unloading"}
                /\ state' = "disposing"
                /\ UNCHANGED <<pool, session, operation, unloading, unsafe, secondReturned>>
FinishDispose == /\ state = "disposing"
                 /\ (~EnforceLeases \/ (~operation /\ ~unloading))
                 /\ state' = "disposed" /\ pool' = "absent" /\ unloading' = FALSE
                 /\ unsafe' = (unsafe \/ operation \/ unloading)
                 /\ UNCHANGED <<session, operation, secondReturned>>
ReturnSecond == /\ state \in {"disposing", "disposed"} /\ ~secondReturned
                /\ (~EnforceLeases \/ state = "disposed")
                /\ secondReturned' = TRUE
                /\ UNCHANGED <<state, pool, session, operation, unloading, unsafe>>

Next == StartSession \/ EndSession \/ StartOperation \/ EndOperation \/
        BeginUnload \/ FinishUnload \/ LoadContext \/ BeginDispose \/ FinishDispose \/ ReturnSecond
Spec == Init /\ [][Next]_vars
TypeOK == /\ state \in {"ready", "unloading", "disposing", "disposed"}
          /\ pool \in {"available", "disposing", "absent"}
          /\ session \in BOOLEAN /\ operation \in BOOLEAN /\ unloading \in BOOLEAN
          /\ unsafe \in BOOLEAN /\ secondReturned \in BOOLEAN
NoDisposalWhileInUse == ~unsafe
NoEarlyDisposeReturn == secondReturned => state = "disposed"
============================================================================
