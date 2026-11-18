---------------------------- MODULE ConfigSync ----------------------------
EXTENDS Naturals
\* Current code (read and mirror under the config lock):
\*   bunx everystate -config test/formal/ConfigSync.cfg \
\*        test/formal/ConfigSync.tla                                  exit 0
\* Pre-fix store open (lock-free snapshot, later sync), kept as regression
\* evidence -- the reported lost update:
\*   bunx everystate -config test/formal/ConfigSyncSnapshot.cfg \
\*        test/formal/ConfigSync.tla                                  exit 12
\* Pre-fix SDK mutation (SQLite first, config second), two hazards:
\*   bunx everystate -config test/formal/ConfigSyncDbFirst.cfg \
\*        test/formal/ConfigSync.tla                                  exit 12
\*   bunx everystate -config test/formal/ConfigSyncStaleHash.cfg \
\*        test/formal/ConfigSync.tla                                  exit 12
\* Mutation control (the same pre-fix sync, no concurrent writer):
\*   bunx everystate -config test/formal/ConfigSyncControl.cfg \
\*        test/formal/ConfigSync.tla                                  exit 0
\*
\* src/index.ts createStore()/addCollection(), src/collections.ts
\* CollectionConfigScope, src/store.ts syncConfigToDb():
\*   - a store opening its index reads the config file and mirrors it into
\*     store_collections as ONE critical section under the config lock
\*     (CollectionConfigScope.withLockedConfig), so a mutation another process
\*     publishes before the read is mirrored, and one published after the reader
\*     released cannot be overwritten by a stale snapshot;
\*   - a configuration mutation publishes the file first and then mirrors that
\*     same document into store_collections, still under the config lock, so a
\*     refused config write leaves no database-only edit and no mirror can land
\*     after a newer publish;
\*   - a direct row write invalidates the stored config_hash, so the hash
\*     certified in store_config is never a document other than the current file.
\*
\* Promise Mirror: a completed store open and a completed mutation leave
\* store_collections equal to the collections in the file. Checked only where
\* both processes are idle: a pre-fix mutation sitting between its SQLite write
\* and its config write is a known transient of that design, and its stable
\* outcome (a failed config write) is what this promise rejects.
\* Promise CertCurrent: the stored config_hash either certifies the current file
\* or is absent; it never certifies a stale document.
\*
\* SyncMode models the store-open designs:
\*   "locked"   current: take the config lock, read the file, mirror SQLite,
\*              release -- one boundary, so no writer can interleave.
\*   "snapshot" pre-fix: read the file lock-free, sync SQLite later, and stamp
\*              the snapshot's hash over the rows it produces.
\* MutationMode models the SDK mutation designs:
\*   "locked"   current: publish the file, mirror SQLite and invalidate the hash
\*              in one lock section.
\*   "dbFirst"  pre-fix: write SQLite first, then the file, and leave the stored
\*              hash untouched.
\* Mutates=FALSE is the control: the same sync path with no concurrent writer, so
\* a violation in the other configs is the interleaving's, not the sync's.
\*
\* Assumptions: one config file and one index database are shared by every
\* process; the config lock is a crash-safe SQLite sidecar a process holds for
\* its whole read/publish/mirror section; inline configuration is process-local
\* and has no concurrency to model.
\* Excluded: a crash between the file publish and the SQLite commit (the
\* surviving config_hash does not certify that state, so the next sync re-derives
\* the rows -- this model claims consistency of completed operations, not a
\* single atomic commit across the file and SQLite); documents, vectors and
\* processor state keyed by collection name (a mirror that fails after a rename
\* publish is repaired by re-running, not modeled); concurrent rewrites of the
\* same collection's fields.
\* Bounds: 2 collection names, 1 file, 1 database, 1 store-open process and 1
\* mutating process, one add per mutation. A model-check pass covers these
\* bounds; it is not a proof of the code.

CONSTANTS
    SyncMode,       \* "locked" (current) | "snapshot" (pre-fix store open)
    MutationMode,   \* "locked" (current) | "dbFirst" (pre-fix mutation)
    Mutates         \* FALSE: mutation control, no concurrent writer

None == {"none"}
AddedName == "b"
Names == {"a", AddedName}

\* A collection set, or the sentinel set that nothing is certified or held yet.
\* Both are sets, so TypeOK compares like with like.
ConfigValues == SUBSET (Names \cup None)

VARIABLES
    file,           \* collection names in the YAML file
    db,             \* collection names in store_collections
    cert,           \* document the stored config_hash certifies, or None
    snap,           \* snapshot the pre-fix store read, or None
    pcA,            \* store open: "idle" | "read" (snapshot frozen)
    pcB             \* mutation: "idle" | "dbOnly" (SQLite written, file not)

vars == <<file, db, cert, snap, pcA, pcB>>

Init ==
    /\ file = {"a"}
    /\ db = {"a"}
    /\ cert = {"a"}
    /\ snap = None
    /\ pcA = "idle"
    /\ pcB = "idle"

\* Current code: the file read and the SQLite mirror are one critical section,
\* so this step observes whatever a completed mutation published.
LockedSync ==
    /\ SyncMode = "locked"
    /\ pcA = "idle"
    /\ db' = file
    /\ cert' = file
    /\ UNCHANGED <<file, snap, pcA, pcB>>

\* Pre-fix: the file is read with no lock; a writer may publish afterwards.
SnapshotRead ==
    /\ SyncMode = "snapshot"
    /\ pcA = "idle"
    /\ snap' = file
    /\ pcA' = "read"
    /\ UNCHANGED <<file, db, cert, pcB>>

\* ... and the frozen snapshot is applied later, certifying itself.
SnapshotApply ==
    /\ SyncMode = "snapshot"
    /\ pcA = "read"
    /\ db' = snap
    /\ cert' = snap
    /\ snap' = None
    /\ pcA' = "idle"
    /\ UNCHANGED <<file, pcB>>

\* Current mutation: publish the file, mirror it into SQLite and invalidate the
\* hash, all inside one lock section.
LockedAdd ==
    /\ MutationMode = "locked"
    /\ Mutates
    /\ pcB = "idle"
    /\ file' = file \cup {AddedName}
    /\ db' = db \cup {AddedName}
    /\ cert' = None
    /\ UNCHANGED <<snap, pcA, pcB>>

\* Pre-fix mutation: SQLite first, with no lock on the file ...
DbFirstAdd ==
    /\ MutationMode = "dbFirst"
    /\ Mutates
    /\ pcB = "idle"
    /\ db' = db \cup {AddedName}
    /\ pcB' = "dbOnly"
    /\ UNCHANGED <<file, cert, snap, pcA>>

\* ... then the file write lands, leaving the stored hash on the old document.
PublishAdd ==
    /\ MutationMode = "dbFirst"
    /\ pcB = "dbOnly"
    /\ file' = file \cup {AddedName}
    /\ pcB' = "idle"
    /\ UNCHANGED <<db, cert, snap, pcA>>

\* ... or the file write is refused: the database-only edit stays behind.
PublishFails ==
    /\ MutationMode = "dbFirst"
    /\ pcB = "dbOnly"
    /\ pcB' = "idle"
    /\ UNCHANGED <<file, db, cert, snap, pcA>>

Next ==
    \/ LockedSync
    \/ SnapshotRead
    \/ SnapshotApply
    \/ LockedAdd
    \/ DbFirstAdd
    \/ PublishAdd
    \/ PublishFails

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ file \in SUBSET Names
    /\ db \in SUBSET Names
    /\ cert \in ConfigValues
    /\ snap \in ConfigValues
    /\ pcA \in {"idle", "read"}
    /\ pcB \in {"idle", "dbOnly"}

Quiescent == pcA = "idle" /\ pcB = "idle"

\* No completed operation leaves the mirror short of, or ahead of, the file.
Mirror == Quiescent => db = file

\* The certified document is the current one or the hash is absent.
CertCurrent == Quiescent => (cert = None \/ cert = file)

============================================================================
