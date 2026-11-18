---------------------------- MODULE ConfigStores ----------------------------
EXTENDS Naturals
\* Check:  bunx everystate test/formal/ConfigStores.tla                     exit 0
\* Legacy: bunx everystate -config test/formal/ConfigStoresLegacy.cfg \
\*         test/formal/ConfigStores.tla                                   exit 12
\*
\* src/index.ts createStore() / addCollection() / close(), src/collections.ts:
\* two SDK stores in one process, each writing a collection configuration
\* through to its own target. Legacy code kept ONE process-global
\* `configSource`: the second createStore() replaced it, and any close() reset
\* it to the default ~/.config/qmd/<index>.yml. Current code binds the target
\* per store (CollectionConfigScope) and a closed scope never writes.
\*
\* Promise NoCrossStoreConfig: a configuration only ever receives collections
\* its own store added, and no mutation falls through to the default file.
\* NoLostMutation is the other half: while a store is open, its own target
\* holds everything its SQLite table holds (the write-through reached it).
\*
\* OwnedSource = TRUE is the current per-store binding (ConfigStores.cfg).
\* FALSE is the legacy global source (ConfigStoresLegacy.cfg): store B's
\* createStore() replaces A's target, and either close() points the shared
\* source at the default file, so a mutation lands in another owner's config.
\*
\* Assumptions: both stores are config-bound, so every mutation writes
\* through; inline mode mutates the caller's object in place; file mode is
\* load-modify-write; the SQLite upsert and the config write are one
\* synchronous step, so only their order against the other store interleaves.
\* Bounds: 2 stores, 1 added collection each, 3 targets (A's YAML file, B's
\* inline object, the default file). Excluded: DB-only stores (never write
\* through), reads, remove/rename/context (same shape), use after close().
\*
\* A model-check pass covers these bounds; it is not a proof of the code.

CONSTANTS
    AddsA,        \* collection names store A adds, e.g. {"a1"}
    AddsB,        \* collection names store B adds, e.g. {"b1"}
    OwnedSource   \* TRUE: target bound per store (current); FALSE: global

VARIABLES
    src,          \* collections.ts `configSource` -- one per process (legacy)
    pcA,          \* store A: "none" -> "live" -> "closed"
    pcB,          \* store B: "none" -> "live" -> "closed"
    fileA,        \* collections recorded in A's YAML (A's declared target)
    inlineB,      \* collections recorded in B's in-memory config object
    defaultFile,  \* collections in the fallback ~/.config/qmd/<index>.yml
    dbA,          \* A's own SQLite store_collections (correctly owned)
    dbB,          \* B's own SQLite store_collections (correctly owned)
    writeTarget   \* where the last write-through actually landed

AllNames == AddsA \cup AddsB

vars == <<src, pcA, pcB, fileA, inlineB, defaultFile, dbA, dbB, writeTarget>>

Init ==
    /\ src = "defaultFile"        \* collections.ts initial value
    /\ pcA = "none"
    /\ pcB = "none"
    /\ fileA = {}
    /\ inlineB = {}
    /\ defaultFile = {}
    /\ dbA = {}
    /\ dbB = {}
    /\ writeTarget = "none"

\* Which target a write-through from `who` reaches: as coded it is whatever the
\* process-global source is; with the owner-bound guard it is the writer's own.
WriteTarget(who) ==
    IF OwnedSource
    THEN IF who = "a" THEN "fileA" ELSE "inlineB"
    ELSE src

\* createStore({configPath: A's yaml}): the source is pointed at A's file,
\* then loadConfig() reads it -- one synchronous prefix.
CreateA ==
    /\ pcA = "none"
    /\ pcA' = "live"
    /\ src' = IF OwnedSource THEN "ownedByStore" ELSE "fileA"
    /\ UNCHANGED <<pcB, fileA, inlineB, defaultFile, dbA, dbB, writeTarget>>

\* createStore({config: B's object}): the same global source is pointed at B's
\* in-memory config, replacing whatever A installed.
CreateB ==
    /\ pcB = "none"
    /\ pcB' = "live"
    /\ src' = IF OwnedSource THEN "ownedByStore" ELSE "inlineB"
    /\ UNCHANGED <<pcA, fileA, inlineB, defaultFile, dbA, dbB, writeTarget>>

\* addCollection: the SQLite upsert goes to this store's db, the config write
\* to WriteTarget, which as coded is not the store's to choose. One step.
AddA(n) ==
    /\ pcA = "live"
    /\ n \in AddsA
    /\ dbA' = dbA \cup {n}
    /\ writeTarget' = WriteTarget("a")
    /\ \/ /\ WriteTarget("a") = "fileA"
          /\ fileA' = fileA \cup {n}
          /\ UNCHANGED <<inlineB, defaultFile>>
       \/ /\ WriteTarget("a") = "inlineB"        \* B's config gets A's collection
          /\ inlineB' = inlineB \cup {n}
          /\ UNCHANGED <<fileA, defaultFile>>
       \/ /\ WriteTarget("a") = "defaultFile"    \* falls back to ~/.config/qmd
          /\ defaultFile' = defaultFile \cup {n}
          /\ UNCHANGED <<fileA, inlineB>>
    /\ UNCHANGED <<src, pcA, pcB, dbB>>

AddB(n) ==
    /\ pcB = "live"
    /\ n \in AddsB
    /\ dbB' = dbB \cup {n}
    /\ writeTarget' = WriteTarget("b")
    /\ \/ /\ WriteTarget("b") = "inlineB"
          /\ inlineB' = inlineB \cup {n}
          /\ UNCHANGED <<fileA, defaultFile>>
       \/ /\ WriteTarget("b") = "fileA"          \* A's YAML gets B's collection
          /\ fileA' = fileA \cup {n}
          /\ UNCHANGED <<inlineB, defaultFile>>
       \/ /\ WriteTarget("b") = "defaultFile"
          /\ defaultFile' = defaultFile \cup {n}
          /\ UNCHANGED <<fileA, inlineB>>
    /\ UNCHANGED <<src, pcA, pcB, dbA>>

\* close(): setConfigSource(undefined) -> the default file target, whether or
\* not the other store is still open and still thinks the target is its own.
CloseA ==
    /\ pcA = "live"
    /\ pcA' = "closed"
    /\ src' = IF OwnedSource THEN "ownedByStore" ELSE "defaultFile"
    /\ UNCHANGED <<pcB, fileA, inlineB, defaultFile, dbA, dbB, writeTarget>>

CloseB ==
    /\ pcB = "live"
    /\ pcB' = "closed"
    /\ src' = IF OwnedSource THEN "ownedByStore" ELSE "defaultFile"
    /\ UNCHANGED <<pcA, fileA, inlineB, defaultFile, dbA, dbB, writeTarget>>

Next ==
    \/ CreateA \/ CreateB \/ CloseA \/ CloseB
    \/ \E n \in AddsA : AddA(n)
    \/ \E n \in AddsB : AddB(n)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ src \in {"defaultFile", "fileA", "inlineB", "ownedByStore"}
    /\ pcA \in {"none", "live", "closed"}
    /\ pcB \in {"none", "live", "closed"}
    /\ fileA \subseteq AllNames
    /\ inlineB \subseteq AllNames
    /\ defaultFile \subseteq AllNames
    \* The SQLite side is correctly owned by construction, so this half is a
    \* typing fact, not the promise.
    /\ dbA \subseteq AddsA
    /\ dbB \subseteq AddsB
    /\ writeTarget \in {"none", "fileA", "inlineB", "defaultFile"}

\* Wiring sanity: the recorded write target names a store that exists. It is
\* the LAST write, so it outlives its writer's close.
SaneConfigWiring ==
    /\ (writeTarget = "fileA") => (pcA # "none")
    /\ (writeTarget = "inlineB") => (pcB # "none")

\* The promise: no store's configuration receives another store's collection,
\* and nothing is written through to the fallback default file.
NoCrossStoreConfig ==
    /\ fileA \subseteq AddsA
    /\ inlineB \subseteq AddsB
    /\ defaultFile = {}

\* While a store is open, its own target holds everything its table holds.
NoLostMutation ==
    /\ (pcA = "live") => (dbA \subseteq fileA)
    /\ (pcB = "live") => (dbB \subseteq inlineB)

=============================================================================
