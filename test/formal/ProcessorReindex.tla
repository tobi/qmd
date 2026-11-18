-------------------------- MODULE ProcessorReindex -------------------------
EXTENDS Naturals
\* Check:
\*   bunx everystate test/formal/ProcessorReindex.tla                     exit 0
\* Mutations:
\*   bunx everystate -config test/formal/ProcessorReindexBroken.cfg \
\*        test/formal/ProcessorReindex.tla                               exit 12
\*   bunx everystate -config test/formal/ProcessorReindexUnguarded.cfg \
\*        test/formal/ProcessorReindex.tla                               exit 12
\*   bunx everystate -config test/formal/ProcessorReindexStaleSource.cfg \
\*        test/formal/ProcessorReindex.tla                               exit 0
\*
\* src/processor-store.ts reindexProcessedFiles: who may write one output path.
\* ownershipOf() answers owned / foreign / unowned from processor_outputs;
\* publishSource() runs inside withSourceSavepoint() (SAVEPOINT / RELEASE /
\* ROLLBACK TO); retireSource() runs in the final pass over processor_sources.
\*
\* Two ownership checks protect publication: the cheap look in the loop, after
\* the awaited executeProcessor and outside the write transaction, and the same
\* question re-read inside the savepoint. The pre-check cannot see a
\* publication that lands between it and SAVEPOINT; the re-check can, and a
\* throw rolls the savepoint back.
\*
\* Promises: ContentMatchesOwner -- the document at an output path holds the
\* markdown of the source processor_outputs credits. FingerprintMatchesPublished
\* -- a trusted "unchanged" fingerprint describes a published output of that
\* source. RawRowNotClaimed -- a documents row from a plain scan (no ownership
\* row) is never turned into a processor output.
\*
\* Bounds and the two-run episode: an output path is `${source}/${doc.key}`, so
\* two sources share one only when one source name is a path prefix of the
\* other. That pair cannot coexist on one filesystem, but is reachable from two
\* enumerations on either side of a file-to-directory replacement: "a" is a
\* regular file for run 1; "a" becomes a directory holding "x" for run 2, whose
\* source is "a/x". Publication writes rows only, so the earlier run still
\* publishes into a tree that no longer exists. Each source is enumerated by
\* its own run, whose seen list holds only its own path, so each run retires the
\* other source. A document row at the path may also predate both enumerations,
\* left by a plain scan of that directory generation.
\*
\* Assumptions: (A1) a run reads only the generation its enumeration saw, so
\* NewSource may read only after OldSource's run left "idle" (Readable);
\* publication re-reads no file and is not gated. (A2) SAVEPOINT..RELEASE is
\* one critical section: the callback is synchronous and its first write takes
\* SQLite's write lock. (A3) processor_outputs is keyed (collection, path); the
\* upsert replaces hash/current but keeps the old source. (A4) no ownership row
\* means foreign, so publication may not enter it. (A5) a throw out of
\* publishSource aborts that run; the other process continues. (A6)
\* document_metadata extraction_version is always valid here. (A7) content is
\* modelled as provenance, not bytes. (A8) a crash between RELEASE and the
\* caller's COMMIT is out of scope.
\*
\* Excluded: more than one output per source, ROOT_MISSING and abort exits,
\* document_metadata / content-table bookkeeping, file_sync_state deletes, and
\* two runs on one connection (scanWriteBatch refuses that). A model-check pass
\* covers these bounds; it is not a proof of the code.

CONSTANTS
    OldSource,      \* source the earlier enumeration saw: a regular file
    NewSource,      \* source the later enumeration saw: a path beneath it
    Active,         \* Sources that have a run in this episode
    RecheckInside,  \* CONTROL: the ownership re-check inside the savepoint
    OwnershipCheck  \* CONTROL: the pre-publication "foreign" look

Sources == {OldSource, NewSource}

Phases == {"idle", "executed", "ready", "done", "skipped", "aborted"}
Who == {"none"} \cup Sources
DocRows == {"absent", "ordinary", "inactive", "active"}
Fingerprints == {"none", "current"}

VARIABLES
    phase,       \* per source: where its own run is in inspect / publish
    owner,       \* processor_outputs.source of the one output path; "none" = no row
    content,     \* which source's markdown the documents row at that path holds
    docRow,      \* the documents row at that path ("active" flag + existence)
    fingerprint, \* processor_sources.fingerprint per source; "none" = no row
    rawRow       \* ghost: this documents row predates processor mode

vars == <<phase, owner, content, docRow, fingerprint, rawRow>>

\* Rows an operator can actually start from: nothing yet; a documents row left
\* by a plain scan of a generation where "a" was a directory, active or already
\* retired by a later plain scan, with no ownership row; or one source's
\* published output with its fingerprint, written by an earlier run (that is
\* what makes a source "stale" when it is not in Active).
Init ==
    /\ phase = [s \in Sources |-> "idle"]
    /\ \/ /\ owner = "none" /\ content = "none" /\ docRow = "absent"
          /\ rawRow = FALSE
          /\ fingerprint = [s \in Sources |-> "none"]
       \/ /\ owner = "none" /\ content = "none"
          /\ docRow \in {"ordinary", "inactive"}
          /\ rawRow = TRUE
          /\ fingerprint = [s \in Sources |-> "none"]
       \/ \E s \in Sources :
            /\ owner = s /\ content = s /\ docRow = "active" /\ rawRow = FALSE
            /\ fingerprint = [t \in Sources |-> IF t = s THEN "current" ELSE "none"]

\* A1: inspectSource starts with readSource(), which rejects anything that is
\* not a regular file. OldSource's run always reads the earlier generation;
\* NewSource exists in an enumeration only because the replacement already
\* happened, so it may read only once OldSource's run left "idle". If OldSource
\* has no run this episode there is nothing to order.
Readable(s) ==
    \/ s = OldSource
    \/ OldSource \notin Active
    \/ phase[OldSource] # "idle"

\* ownershipOf(), in terms of the two rows it reads:
\*   owner = s                   "owned"    (its document row may be
\*                                          "absent" or "inactive" and is
\*                                          republished)
\*   owner = t, t # s            "foreign"
\*   owner = "none", row exists  "foreign"  (raw row or retired row)
\*   owner = "none", no row      "unowned"  (publication creates both)
Foreign(s) ==
    \/ owner # "none" /\ owner # s
    \/ owner = "none" /\ docRow # "absent"

\* inspectSource's read-only exits: aborted signal, realpath containment,
\* readSource() failure, or executeProcessor() threw. Every one is skipped(...)
\* and the loop advances without touching a row.
InspectFails(s) ==
    /\ s \in Active /\ phase[s] = "idle"
    /\ Readable(s)
    /\ phase' = [phase EXCEPT ![s] = "skipped"]
    /\ UNCHANGED <<owner, content, docRow, fingerprint, rawRow>>

\* The stored fingerprint, the recorded output_count and the `invalid` query all
\* agree, so executeProcessor() does not run.
Reuse(s) ==
    /\ s \in Active /\ phase[s] = "idle"
    /\ Readable(s)
    /\ fingerprint[s] = "current"
    /\ owner = s /\ docRow = "active" /\ content = s
    /\ phase' = [phase EXCEPT ![s] = "done"]
    /\ UNCHANGED <<owner, content, docRow, fingerprint, rawRow>>

\* executeProcessor() ran outside the write transaction and returned documents:
\* the verdict is "publishable".
Inspect(s) ==
    /\ s \in Active /\ phase[s] = "idle"
    /\ Readable(s)
    /\ phase' = [phase EXCEPT ![s] = "executed"]
    /\ UNCHANGED <<owner, content, docRow, fingerprint, rawRow>>

\* The cheap ownership look, outside the write transaction: any produced
\* document whose path is "foreign" skips the source
\* (PROCESSOR_PATH_COLLISION) and leaves every row alone.
SkipCollision(s) ==
    /\ OwnershipCheck
    /\ s \in Active /\ phase[s] = "executed"
    /\ Foreign(s)
    /\ phase' = [phase EXCEPT ![s] = "skipped"]
    /\ UNCHANGED <<owner, content, docRow, fingerprint, rawRow>>

PrecheckPasses(s) ==
    /\ s \in Active /\ phase[s] = "executed"
    /\ (~OwnershipCheck) \/ ~Foreign(s)
    /\ phase' = [phase EXCEPT ![s] = "ready"]
    /\ UNCHANGED <<owner, content, docRow, fingerprint, rawRow>>

\* ONE ACTION = SAVEPOINT .. RELEASE. Rows the block writes: the documents row
\* at `${source}/${doc.key}` becomes active and holds this source's markdown
\* (insert, or update then active = 1); the processor_outputs upsert replaces
\* hash and current but keeps the old `source` (A3); processor_sources records
\* this source's fingerprint, inside the same savepoint. The guard is the
\* re-check: a path that became foreign since the pre-check throws. No file is
\* read here, so the generation does not gate this action (A1).
Publish(s) ==
    /\ s \in Active /\ phase[s] = "ready"
    /\ (~RecheckInside) \/ ~Foreign(s)
    /\ owner' = (IF owner = "none" THEN s ELSE owner)
    /\ content' = s
    /\ docRow' = "active"
    /\ fingerprint' = [fingerprint EXCEPT ![s] = "current"]
    /\ phase' = [phase EXCEPT ![s] = "done"]
    /\ UNCHANGED rawRow

\* The re-check throws "Processor ownership changed during publication";
\* withSourceSavepoint catches it and ROLLBACK TO undoes everything the block
\* wrote, so rows are exactly what the block read: the source keeps its
\* previous outputs and its previous fingerprint, never half of each. The throw
\* is rethrown, so the rest of THIS run never happens (A5); the other run is a
\* separate process and continues.
Rollback(s) ==
    /\ s \in Active /\ phase[s] = "ready"
    /\ RecheckInside /\ Foreign(s)
    /\ phase' = [phase EXCEPT ![s] = "aborted"]
    /\ UNCHANGED <<owner, content, docRow, fingerprint, rawRow>>

\* A run's loop is over when its single enumerated source answered: skipped or
\* done. An aborted or still-running source is not "over".
LoopFinished(r) ==
    r \in Active /\ phase[r] \in {"done", "skipped"}

\* Run r lists processor_sources and retires every row it did not see. Its seen
\* set holds exactly r's own path, so the OTHER source is always retired by this
\* run, even while its own run is still working. The delete is
\* ownership-guarded: deleteOwnedOutput removes the documents row only while an
\* ownership row for (collection, t, path) exists, so a path another source now
\* owns keeps its document.
RetireSourceIn(r, t) ==
    /\ LoopFinished(r)
    /\ t # r
    /\ fingerprint[t] # "none"          \* the processor_sources row the pass lists
    /\ owner' = (IF owner = t THEN "none" ELSE owner)
    /\ content' = (IF owner = t THEN "none" ELSE content)
    /\ docRow' = (IF owner = t THEN "absent" ELSE docRow)
    /\ fingerprint' = [fingerprint EXCEPT ![t] = "none"]
    /\ UNCHANGED <<phase, rawRow>>

\* Deactivate a row left from an ordinary collection scan -- active, with no
\* processor_outputs row. The run's seen set holds source paths and the output
\* path modelled here is not a source path, so the exemption never applies.
RetireOrdinaryIn(r) ==
    /\ LoopFinished(r)
    /\ owner = "none" /\ docRow = "ordinary"
    /\ docRow' = "inactive"
    /\ UNCHANGED <<phase, owner, content, fingerprint, rawRow>>

Next ==
    \/ \E s \in Sources : InspectFails(s)
    \/ \E s \in Sources : Reuse(s)
    \/ \E s \in Sources : Inspect(s)
    \/ \E s \in Sources : SkipCollision(s)
    \/ \E s \in Sources : PrecheckPasses(s)
    \/ \E s \in Sources : Publish(s)
    \/ \E s \in Sources : Rollback(s)
    \/ \E r, t \in Sources : RetireSourceIn(r, t)
    \/ \E r \in Sources : RetireOrdinaryIn(r)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in [Sources -> Phases]
    /\ owner \in Who
    /\ content \in Who
    /\ docRow \in DocRows
    /\ fingerprint \in [Sources -> Fingerprints]
    /\ rawRow \in BOOLEAN

\* THE PROMISE. Either no source owns the path, or the path's documents row is
\* active and holds exactly the markdown of the source the ownership table
\* credits: never a document rendered by A while processor_outputs says B.
ContentMatchesOwner ==
    owner = "none" \/ (docRow = "active" /\ content = owner)

\* A documents row that predates processor mode is never turned into a
\* processor output: it is foreign, so the source is skipped or the savepoint
\* rolls back.
RawRowNotClaimed ==
    rawRow = FALSE \/ owner = "none"

\* The fingerprint a later run trusts for "unchanged" is written in the same
\* savepoint as the document, so a committed fingerprint always describes a
\* published output of that source.
FingerprintMatchesPublished ==
    \A s \in Sources :
        fingerprint[s] = "current" => (owner = s /\ docRow = "active" /\ content = s)

\* Reported but not forbidden: concurrent runs retire each other (each run's
\* seen set holds only its own path), and a source name is not stable across the
\* replacement, so "a" and "a/x" both own "a/x/y" in turn. Nothing in the code
\* ties an output path to the generation that produced it. If the code changes,
\* re-derive Foreign() from ownershipOf, Publish's guard and row effects from
\* publishSource, Rollback from withSourceSavepoint's catch, and the two
\* retirement actions from the final passes.

=============================================================================
