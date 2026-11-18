---------------------- MODULE ConfigRenameRecovery ----------------------
EXTENDS Naturals
\* Current code (evidence-gated recovery, conflicting mutations refused):
\*   bunx everystate -config test/formal/ConfigRenameRecovery.cfg \
\*        test/formal/ConfigRenameRecovery.tla                          exit 0
\* Broken control (recovery inferred from the file shape, no marker):
\*   bunx everystate -config test/formal/ConfigRenameRecoveryInferred.cfg \
\*        test/formal/ConfigRenameRecovery.tla                          exit 12
\* Broken control (a conflicting mutation may clobber the marker):
\*   bunx everystate -config test/formal/ConfigRenameRecoveryClobber.cfg \
\*        test/formal/ConfigRenameRecovery.tla                          exit 12
\*
\* src/collections.ts: a collection rename is two resources. The configuration
\* is published first (so a refused publish loses no index data), preceded by a
\* `.pending-rename.json` marker naming the exact old→new pair. The index half
\* (documents, sync rows, vector partition, processor state) moves only against
\* that marker, and the marker is removed only after the move succeeds.
\*
\* Covered here:
\*   - a crash between the marker and the publication (phase "marked"), and a
\*     crash after it (phase "published"); the retry must publish first when the
\*     publication never happened;
\*   - exact matching: the move happens only for the marker naming this pair,
\*     and into that marker's target;
\*   - unrelated mutations (a name the marker does not mention) may proceed and
\*     leave the evidence intact;
\*   - a mutation that would touch the marker's names, and any second rename,
\*     is refused while the evidence is live, so it cannot be clobbered;
\*   - a marker left by a publication that never landed is stale: the current
\*     code reconciles it against the file and drops it, so it neither blocks a
\*     later change nor authorizes a move.
\*
\* Broken controls (CONSTANTS):
\*   MoveGate     = "evidence" (current) | "inferred" (moves whenever the file
\*                  no longer lists the old name and lists another one, which is
\*                  the ownership-corruption hazard);
\*   MutationGate = "reject"   (current) | "clobber"  (a conflicting mutation
\*                  clears the marker, losing the only evidence).
\*
\* Promises:
\*   NoMoveWithoutPublication: a's documents end up under the target only when
\*     the rename's publication exists (the file reflects it while carrying the
\*     marker, or this run published it).
\*   EvidenceNotClobbered: an attempted but unfinished rename always keeps its
\*     marker while the documents are still under the old name.
\*
\* Assumptions: one config file and one index; the marker and the publication
\* happen under the config lock, so no observer sees a half-written marker; the
\* marker's sidecar is not edited by hand (an unreadable marker is not evidence);
\* inline configurations are process-local and carry the marker in memory (same
\* steps, no crash window to model). Excluded: documents keyed by more than one
\* name in the same collection, concurrent renames on different configs sharing
\* one index, and mixed old-version binaries that never wrote a marker.
\* Bounds: 3 names (a old, b target, c unrelated), one marker, one document set,
\* one crash, one retry. A model-check pass covers these bounds; it is not a
\* proof of the code.
CONSTANTS MoveGate, MutationGate

Old == "a"
Target == "b"
Other == "c"
Names == {Old, Target, Other}

VARIABLES
    file,          \* collection names in the YAML
    docs,          \* name the old collection's documents are stored under
    marker,        \* pending rename: {} or {<<Old, Target>>}
    publishedTo,   \* target names whose publication this run observed
    attempted,     \* a rename was started at least once
    phase          \* "idle" | "marked" | "published" | "crashed"

vars == <<file, docs, marker, publishedTo, attempted, phase>>

Init ==
    /\ file = {Old, Other}
    /\ docs = Old
    /\ marker = {}
    /\ publishedTo = {}
    /\ attempted = FALSE
    /\ phase = "idle"

\* The user renames a to b: the marker is written before the publication.
RenameStart ==
    /\ phase = "idle"
    /\ Old \in file
    /\ Target \notin file
    /\ marker' = {<<Old, Target>>}
    /\ attempted' = TRUE
    /\ phase' = "marked"
    /\ UNCHANGED <<file, docs, publishedTo>>

RenamePublish ==
    /\ phase = "marked"
    /\ file' = (file \ {Old}) \cup {Target}
    /\ publishedTo' = publishedTo \cup {Target}
    /\ phase' = "published"
    /\ UNCHANGED <<docs, marker, attempted>>

\* The process dies between the marker and the publication, or after it.
Crash ==
    /\ phase \in {"marked", "published"}
    /\ phase' = "crashed"
    /\ UNCHANGED <<file, docs, marker, publishedTo, attempted>>

\* A human or a git checkout makes the file look renamed without any rename
\* publication: the evidence-free shape the inferred control acts on.
ManualEdit ==
    /\ phase \in {"idle", "crashed"}
    /\ Old \in file
    /\ Target \notin file
    /\ file' = (file \ {Old}) \cup {Target}
    /\ UNCHANGED <<docs, marker, publishedTo, attempted, phase>>

\* Retrying the rename in a new process. Evidence-gated: only the exact marker
\* authorizes the move, and a publication that never happened is made first.
\* Inferred control: the file shape alone authorizes the move.
RenameRetry ==
    /\ phase \in {"idle", "crashed"}
    /\ IF MoveGate = "evidence"
       THEN /\ marker = {<<Old, Target>>}
            /\ \/ /\ Old \in file /\ Target \notin file          \* never published
                  /\ file' = (file \ {Old}) \cup {Target}
                  /\ publishedTo' = publishedTo \cup {Target}
               \/ /\ Old \notin file /\ Target \in file           \* published already
                  /\ UNCHANGED file
                  /\ publishedTo' = publishedTo \cup {Target}
       ELSE /\ Old \notin file /\ Target \in file                 \* shape only
            /\ UNCHANGED <<file, publishedTo>>
    /\ docs' = Target
    /\ marker' = {}
    /\ phase' = "idle"
    /\ UNCHANGED attempted

\* An unrelated mutation (add or remove `c`) may proceed at any time and must
\* leave the marker alone.
MutateOther ==
    /\ \/ /\ Other \in file /\ file' = file \ {Other}
       \/ /\ Other \notin file /\ file' = file \cup {Other}
    /\ UNCHANGED <<docs, marker, publishedTo, attempted, phase>>

\* A mutation that changes one of the marker's names while its evidence is live:
\* the current code refuses it, the clobber control destroys the evidence.
MutateConflict ==
    /\ marker # {}
    /\ IF MutationGate = "reject"
       THEN UNCHANGED <<file, marker>>
       ELSE /\ marker' = {}
            /\ file' = file \ {Target}
    /\ UNCHANGED <<docs, publishedTo, attempted, phase>>

Next ==
    \/ RenameStart
    \/ RenamePublish
    \/ Crash
    \/ ManualEdit
    \/ RenameRetry
    \/ MutateOther
    \/ MutateConflict

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ file \subseteq Names
    /\ docs \in Names
    /\ marker \subseteq Names \X Names
    /\ publishedTo \subseteq Names
    /\ attempted \in BOOLEAN
    /\ phase \in {"idle", "marked", "published", "crashed"}

\* The documents never end up under the target on the strength of a name alone.
NoMoveWithoutPublication == docs = Target => Target \in publishedTo

\* An unfinished rename keeps its only evidence while the documents are still
\* under the old name.
EvidenceNotClobbered == attempted /\ docs = Old => marker # {}

============================================================================
