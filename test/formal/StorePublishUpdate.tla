----------------------- MODULE StorePublishUpdate --------------------------
EXTENDS Naturals
\* Current code (reconcile under the publish lock):
\*   bunx everystate -config test/formal/StorePublishUpdateRace.cfg \
\*        test/formal/StorePublishUpdate.tla                          exit 0
\* Pre-fix code (blind copy), kept as regression evidence:
\*   bunx everystate -config test/formal/StorePublishUpdateBlind.cfg \
\*        test/formal/StorePublishUpdate.tla                          exit 12
\* Control (lock-free alternative: hold the writer lock across the build):
\*   bunx everystate -config test/formal/StorePublishUpdateControl.cfg \
\*        test/formal/StorePublishUpdate.tla                          exit 0
\*
\* src/store.ts: the one-time CJK FTS rebuild in
\* rebuildFTSForCjkNormalization racing a concurrent `qmd update`
\* (reindexCollection) on the same index, from two processes.
\*   P: reads documents JOIN content into a per-process shadow FTS table in
\*      batches of 500 OUTSIDE any transaction and without the writer lock,
\*      recording each row's identity (hash, path, title) in a per-process
\*      snapshot table, then BEGIN IMMEDIATE. Inside that transaction it diffs
\*      the snapshot against documents: rows that changed are re-read with CJK
\*      normalization, rows of deactivated/removed documents are dropped, and
\*      documents the snapshot never saw are added. Then it copies the shadow
\*      into documents_fts, stamps fts_cjk_normalized_version and COMMITs. The
\*      rebuild returns early once the stamp is present, so a bad publish is
\*      never repaired.
\*   U: a per-file batch transaction that commits the document row and its FTS
\*      row (documents_ai/au triggers, or rebuildDocumentFTS) together.
\*
\* Promise FtsSeesActiveDocument: keyword search (searchFTS, a MATCH on
\* documents_fts joined to active documents) returns every active document with
\* its current text.
\*
\* PublishMode models the three candidate designs:
\*   "blind"     the pre-fix code: publish exactly what the shadow holds, so a
\*               document committed after the read is dropped or gets old text.
\*   "lock"      the control: BEGIN IMMEDIATE before the first batch read, so
\*               the snapshot and the publish are one atomic boundary. Correct,
\*               and it blocks every other writer for the whole rebuild.
\*   "reconcile" the current code: the publish transaction re-reads whatever
\*               changed between the snapshot and the commit, so the published
\*               index is the corpus as of the commit without holding the lock
\*               while the shadow is built. The stamp moves with the copy.
\*
\* Assumptions: separate connections (the updater waits on the writer lock);
\* one publisher, one updater -- a second process that stamps first only makes
\* the publisher skip its publish. documents.id is AUTOINCREMENT, so ids are
\* never reused. Deactivation is not modelled: reconcile drops those rows and
\* the active = 1 join masks a resurrected one. Bounds: 1 document, 1 shadow
\* table, MaxGen = 2 versions of that document.
\*
\* A model-check pass covers these bounds; it is not a proof of the code.

CONSTANTS
    MaxGen,             \* how many versions of the one document the run may produce
    PublishMode         \* "blind" | "lock" | "reconcile"

VARIABLES
    liveGen,    \* body generation committed in documents + content, 0 = none
    ftsGen,     \* generation of that document id's documents_fts row, 0 = none
    shadowGen,  \* generation the publisher copied into its private shadow table
    shadowRead, \* the publisher has finished reading (shadow is complete)
    stamped,    \* fts_cjk_normalized_version is set
    lock,       \* the publisher holds the WAL writer lock
    ppc,        \* publisher: "start" -> "publishing" -> "done"
    upc         \* updater: "start" -> "done"

vars == <<liveGen, ftsGen, shadowGen, shadowRead, stamped, lock, ppc, upc>>

Init ==
    /\ liveGen = 0
    /\ ftsGen = 0
    /\ shadowGen = 0
    /\ shadowRead = FALSE
    /\ stamped = FALSE
    /\ lock = FALSE
    /\ ppc = "start"
    /\ upc = "start"

TypeOK ==
    /\ PublishMode \in {"blind", "lock", "reconcile"}
    /\ liveGen   \in 0..MaxGen
    /\ ftsGen    \in 0..MaxGen
    /\ shadowGen \in 0..MaxGen
    /\ shadowRead \in BOOLEAN
    /\ stamped \in BOOLEAN
    /\ lock \in BOOLEAN
    /\ ppc \in {"start", "publishing", "done"}
    /\ upc \in {"start", "done"}

\* (P) The batch loop that reads documents JOIN content into the private shadow
\* table (and its identity snapshot), one implicit snapshot per batch, with no
\* transaction and no writer lock. Modelled as the generation it captured.
PublisherRead ==
    /\ ppc = "start"
    /\ shadowGen' = liveGen
    /\ shadowRead' = TRUE
    /\ ppc' = "publishing"
    \* Guard option: take the writer lock BEFORE the first read, so no other
    \* connection can commit between the snapshot and the publish.
    /\ lock' = IF PublishMode = "lock" THEN TRUE ELSE lock
    /\ UNCHANGED <<liveGen, ftsGen, stamped, upc>>

\* (P) BEGIN IMMEDIATE; double-checked version read; reconcile the snapshot's
\* changed rows against documents (so the effective published generation is the
\* live one for "reconcile" and the shadow's for "blind"); DELETE
\* documents_fts; INSERT from the shadow (plus the re-read rows); stamp; COMMIT --
\* one atomic step for every other connection.
Publish ==
    /\ ppc = "publishing"
    /\ shadowRead
    /\ ~stamped
    /\ ftsGen' = CASE PublishMode = "reconcile" -> liveGen
                 [] OTHER                    -> shadowGen
    /\ stamped' = TRUE
    /\ lock' = FALSE
    /\ ppc' = "done"
    /\ UNCHANGED <<liveGen, shadowGen, shadowRead, upc>>

\* (U) reindexCollection's per-file batch transaction: the document row and its
\* FTS row commit together. While the publisher holds the writer lock this
\* waits (busy_timeout, default 120 s), so it is disabled.
UpdaterWrite ==
    /\ upc = "start"
    /\ liveGen < MaxGen
    /\ ~lock
    /\ liveGen' = liveGen + 1
    /\ ftsGen' = liveGen + 1
    /\ UNCHANGED <<shadowGen, shadowRead, stamped, lock, ppc, upc>>

UpdaterDone ==
    /\ upc = "start"
    /\ upc' = "done"
    /\ UNCHANGED <<liveGen, ftsGen, shadowGen, shadowRead, stamped, lock, ppc>>

Next ==
    \/ PublisherRead
    \/ Publish
    \/ UpdaterWrite
    \/ UpdaterDone

\* No fairness clause: the property below is a safety invariant, so only the
\* reachable states matter, not which of them a run lingers in.
Spec == Init /\ [][Next]_vars

\* Keyword search sees every active document with its current text. A publish
\* from a stale shadow breaks it in both directions: a document committed after
\* the read keeps no row at all (liveGen = 1, ftsGen = 0), and one updated after
\* the read gets its old text back (liveGen = 2, ftsGen = 1). "reconcile" and
\* "lock" publish the live generation.
FtsSeesActiveDocument == liveGen = 0 \/ ftsGen = liveGen

\* The stamp must move with the copy: it is written in the same transaction as
\* DELETE+INSERT (or the reconciled re-reads), and the rebuild is skipped
\* forever after.

=============================================================================
