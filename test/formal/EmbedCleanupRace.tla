------------------------ MODULE EmbedCleanupRace ------------------------
EXTENDS Naturals, FiniteSets
\* Current code (one embed domain per index, public clear included):
\*   bunx everystate -config test/formal/EmbedCleanupRace.cfg \
\*        test/formal/EmbedCleanupRace.tla                          exit 0
\* Pre-fix code (no lock inside generateEmbeddings), regression evidence:
\*   bunx everystate -config test/formal/EmbedCleanupRaceBlind.cfg \
\*        test/formal/EmbedCleanupRace.tla                          exit 12
\* Control (lock-free alternative: atomic recheck-and-delete):
\*   bunx everystate -config test/formal/EmbedCleanupRaceGuarded.cfg \
\*        test/formal/EmbedCleanupRace.tla                          exit 0
\* Pre-fix public clear (no embed lock), regression evidence:
\*   bunx everystate -config test/formal/EmbedCleanupRaceClear.cfg \
\*        test/formal/EmbedCleanupRace.tla                          exit 12
\*
\* Concurrent actors covered:
\*   1. Embed run A, cleaning up its own partial batch
\*      (removeIncompleteEmbeddings inside generateEmbeddings).
\*   2. Embed run B, publishing a complete two-chunk result for the same hash
\*      and embedding model (generateEmbeddings / SDK store.embed / CLI embed).
\*   3. The public vector clear (clearAllEmbeddings, whole-index or one
\*      collection). In the current code it takes the same per-index embed lock,
\*      so it neither drops the vec0 table under a run that is publishing into
\*      it nor commits between a run's pending-vector read and its writes.
\* Both runs and the clear are bounded to one document and one hash.
\*
\* Not covered (deliberately): writes made directly through `store.internal` or
\* any raw Database handle, which bypass the operation lifecycle and can hold no
\* embed lock, and binaries of a different version sharing the index — an older
\* build that neither takes this lock nor serialises its clear is a separate
\* deployment, not an interleaving of the code modeled here. Mixed-version and
\* direct-handle writes are outside the promise.
\*
\* src/store.ts generateEmbeddings is the single entry point for the CLI, the
\* SDK's store.embed() and direct library calls, and it takes the crash-safe
\* sidecar embed lock (src/cli/embed-lock.ts) for the whole run:
\*   - LockDomain    = TRUE  (current code): A owns the index's embed lock from
\*                             before it inspects the partition until it has
\*                             deleted or kept its chunks, so B cannot publish
\*                             between A's read and A's delete.
\*   - LockDomain    = FALSE (pre-fix code): nothing serializes the runs; A's
\*                             separate read and delete straddle B's commit and
\*                             delete the complete hash.
\*   - AtomicDelete  = TRUE  (hypothetical lock-free control): the delete
\*                             re-checks the chunk count under the writer lock,
\*                             so a hash that became complete survives.
\* ClearMode models the public clear:
\*   - "guarded"   (current code): the clear takes the embed lock, so it waits
\*                 for a run to finish (lock = FALSE) before touching vectors.
\*   - "unguarded" (pre-fix code): the clear touches vectors at any point and
\*                 can therefore interleave with a run holding the lock.
\* Clears = FALSE is the mutation control: the same cleanup model with no clear
\* actor, so the pre-fix cleanup violation cannot be blamed on a clear.
\*
\* CompletedVectorsRemain is qualified by `cleared`: an intentional clear is
\* allowed to remove a completed hash, so the promise only says that a completed
\* publication stays present until a clear actually happens — not that vectors
\* are permanent. A publish resets `cleared`, so the promise covers the latest
\* completed publication rather than being disabled forever by one clear.
\* NoClearDuringRun says the clear never runs while a run holds the lock; it is
\* the atomic-step form of "the clear and the run are one exclusion domain".
\*
\* Assumptions: all participants share one on-disk index and resolve the same
\* lock path; the crash-safe sidecar lock is taken and released as the code
\* takes it; a clear that runs while a run holds the lock is the hazard, and its
\* consequences (the run's spurious failure, or a later republish of cleared
\* content) are outside this bounded model.
\* A model-check pass covers these bounds; it is not a proof of the code.
CONSTANTS LockDomain, AtomicDelete, ClearMode, Clears

VARIABLES chunks, cleanup, completed, cleared, tornClear, lock
vars == <<chunks, cleanup, completed, cleared, tornClear, lock>>

Init ==
    /\ chunks = {0}
    /\ cleanup = "start"
    /\ completed = FALSE
    /\ cleared = FALSE
    /\ tornClear = FALSE
    /\ lock = FALSE

\* Run A starts its embed run and inspects the partition:
\* removeIncompleteEmbeddings reads one of two expected chunks. Entering the run
\* is where the index's embed lock is taken (current code).
RunARead ==
    /\ cleanup = "start"
    /\ Cardinality(chunks) = 1
    /\ lock' = IF LockDomain THEN TRUE ELSE lock
    /\ cleanup' = "delete"
    /\ UNCHANGED <<chunks, completed, cleared, tornClear>>

\* Run A, inside its lock, finds the hash already complete and deletes nothing.
RunAComplete ==
    /\ cleanup = "start"
    /\ Cardinality(chunks) = 2
    /\ lock' = IF LockDomain THEN TRUE ELSE lock
    /\ cleanup' = "done"
    /\ UNCHANGED <<chunks, completed, cleared, tornClear>>

\* Run B publishes a complete two-chunk result. It needs the index's embed lock
\* itself, so it is disabled while A holds it. A fresh publication is a new
\* subject for CompletedVectorsRemain, so it clears the `cleared` marker.
PublishComplete ==
    /\ ~completed
    /\ ~(LockDomain /\ lock)
    /\ chunks' = {0, 1}
    /\ completed' = TRUE
    /\ cleared' = FALSE
    /\ UNCHANGED <<cleanup, tornClear, lock>>

\* The public vector clear. Current code takes the same embed lock, so it waits
\* for a run that holds it; the pre-fix clear does not, which is the hazard
\* NoClearDuringRun reports.
ClearVectors ==
    /\ Clears
    /\ (\/ /\ ClearMode = "guarded"
             /\ lock = FALSE
        \/ /\ ClearMode = "unguarded")
    /\ chunks' = {}
    /\ completed' = FALSE
    /\ cleared' = TRUE
    \* Parenthesised on the right: `=` binds tighter than `\/`, so the primed
    \* assignment must cover the whole disjunction or the second disjunct is a
    \* state predicate and tornClear' is left unassigned.
    /\ tornClear' = (tornClear \/ (LockDomain /\ lock))
    /\ UNCHANGED <<cleanup, lock>>

\* removeIncompleteEmbeddings deletes by hash after its earlier read, then the
\* run releases the index lock. The control rechecks the count atomically.
DeletePartial ==
    /\ cleanup = "delete"
    /\ chunks' = IF AtomicDelete /\ Cardinality(chunks) = 2 THEN chunks ELSE {}
    /\ cleanup' = "done"
    /\ lock' = FALSE
    /\ UNCHANGED <<completed, cleared, tornClear>>

Next == RunARead \/ RunAComplete \/ PublishComplete \/ ClearVectors \/ DeletePartial
Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ chunks \subseteq {0, 1}
    /\ cleanup \in {"start", "delete", "done"}
    /\ completed \in BOOLEAN
    /\ cleared \in BOOLEAN
    /\ tornClear \in BOOLEAN
    /\ lock \in BOOLEAN

\* A completed publication stays present until a clear happens. Not permanence:
\* an intentional clear may remove it, and the next publish is a new subject.
CompletedVectorsRemain == completed /\ ~cleared => chunks = {0, 1}

\* The clear and the runs are one exclusion domain: no clear while a run holds
\* the index's embed lock.
NoClearDuringRun == ~tornClear
=============================================================================
