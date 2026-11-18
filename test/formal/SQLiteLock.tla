---------------------------- MODULE SQLiteLock ----------------------------
EXTENDS Naturals, FiniteSets
\* Check: bunx everystate test/formal/SQLiteLock.tla
\* Mutation: bunx everystate -config test/formal/SQLiteLockBroken.cfg test/formal/SQLiteLock.tla
\* src/cli/embed-lock.ts: one independent SQLite sidecar connection per
\* process; BEGIN EXCLUSIVE acquires its writer lock, ROLLBACK/close or process
\* death releases it. The index database is not this sidecar.
\* AtomicAcquire=FALSE is a deliberately broken control (check then acquire).
\* Assumptions: one local SQLite database, lock/connection remains open for the
\* whole embed body, SQLite serializes BEGIN, process death releases OS locks,
\* and no older qmd version participates. Legacy PID-file cleanup only affects
\* admission and is outside the new-version safety claim.
CONSTANTS Procs, AtomicAcquire
VARIABLES phase, holders
vars == <<phase, holders>>
Init == /\ phase = [p \in Procs |-> "opening"]
        /\ holders = {}

\* BEGIN EXCLUSIVE and its SQLite single-writer guard form one atomic step.
Acquire(p) == /\ AtomicAcquire /\ phase[p] = "opening" /\ holders = {}
              /\ holders' = holders \cup {p}
              /\ phase' = [phase EXCEPT ![p] = "embedding"]
\* Control: a non-atomic check followed by an unguarded acquire.
Check(p) == /\ ~AtomicAcquire /\ phase[p] = "opening" /\ holders = {}
            /\ phase' = [phase EXCEPT ![p] = "checked"]
            /\ UNCHANGED holders
Enter(p) == /\ ~AtomicAcquire /\ phase[p] = "checked"
            /\ holders' = holders \cup {p}
            /\ phase' = [phase EXCEPT ![p] = "embedding"]
\* SQLite busy -> return without embedding.
Busy(p) == /\ phase[p] = "opening" /\ holders # {}
           /\ phase' = [phase EXCEPT ![p] = "busy"]
           /\ UNCHANGED holders
\* release(), or death while holding the SQLite connection.
Release(p) == /\ phase[p] = "embedding"
              /\ holders' = holders \ {p}
              /\ phase' = [phase EXCEPT ![p] = "released"]
Crash(p) == /\ phase[p] \in {"opening", "checked", "embedding"}
            /\ holders' = holders \ {p}
            /\ phase' = [phase EXCEPT ![p] = "crashed"]
Next == \E p \in Procs : Acquire(p) \/ Check(p) \/ Enter(p) \/ Busy(p) \/ Release(p) \/ Crash(p)
Spec == Init /\ [][Next]_vars
TypeOK == /\ phase \in [Procs -> {"opening", "checked", "embedding", "busy", "released", "crashed"}]
          /\ holders \subseteq Procs
          /\ holders = {p \in Procs : phase[p] = "embedding"}
SingleOwner == Cardinality(holders) <= 1
============================================================================
