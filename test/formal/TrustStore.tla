---------------------------- MODULE TrustStore ----------------------------
EXTENDS Naturals, FiniteSets
\* Check: bunx everystate test/formal/TrustStore.tla
\* Mutation: bunx everystate -config test/formal/TrustStoreUnlocked.cfg test/formal/TrustStore.tla

\* src/trust.ts: <config dir>/trusted.json maps a resolved config path to
\* {hooks: digest, trustedAt}. That file is the whole record of what has been
\* approved; a gate that cannot read it treats every path as unapproved.
\*
\* Every trust write is mutateTrustStore (src/trust.ts:367-374):
\*   loadTrustStore -> mutate -> saveTrustStore
\* Read and write must form one critical section. The lock is an open
\* BEGIN EXCLUSIVE transaction on trusted.json.lock.sqlite
\* (src/trust.ts:339-356, same primitive as src/cli/embed-lock.ts); ROLLBACK
\* and close end it, and so does process death, so the kernel never leaves it
\* held and there is no pidfile to go stale.
\*
\* Serialized = TRUE models the current code. Serialized = FALSE is the
\* pre-fix shape — load, mutate, writeFileSync, no lock — expected to violate
\* NoLostApproval: two processes each read the same store, each publish their
\* own snapshot, and one approval silently disappears.
\*
\* Modeled: two writer processes, one approval each, atomic publish by rename
\* (src/trust.ts:303-314, so a writer killed before the rename leaves the
\* previous complete store), and a crash that dies holding the transaction.
\* Not modeled: revokeTrust (same read-modify-write, and its lost update is
\* the worse direction — a revoked approval survives), digest comparison and
\* gate policy (orthogonal to write interleaving), and readers, which take no
\* lock because the rename makes every published store complete.

CONSTANTS Writers, Serialized

VARIABLES phase, viewed, disk, holder

vars == <<phase, viewed, disk, holder>>

\* phase[w]  : where writer w is in its own recordTrust call
\* viewed[w] : the store that w's loadTrustStore returned
\* disk      : approvals currently published in trusted.json (one per writer)
\* holder    : who owns the BEGIN EXCLUSIVE transaction, or "free"
Init == /\ phase = [w \in Writers |-> "start"]
        /\ viewed = [w \in Writers |-> {}]
        /\ disk = {}
        /\ holder = "free"

\* BEGIN EXCLUSIVE. Contenders queue inside SQLite; exactly one owns it, and
\* it keeps owning it until that writer publishes or dies.
Acquire(w) == /\ Serialized
              /\ phase[w] = "start"
              /\ holder = "free"
              /\ holder' = w
              /\ UNCHANGED <<phase, viewed, disk>>

\* loadTrustStore: reads the published store, under the lock in the current
\* code and with no guard in the pre-fix control.
Read(w) == /\ phase[w] = "start"
           /\ (~Serialized \/ holder = w)
           /\ viewed' = [viewed EXCEPT ![w] = disk]
           /\ phase' = [phase EXCEPT ![w] = "writing"]
           /\ UNCHANGED <<disk, holder>>

\* saveTrustStore: the in-memory mutation (viewed[w] \cup {w}) is renamed over
\* trusted.json in one step, and mutateTrustStore then releases the lock
\* (ROLLBACK + close). The store never passes through a partial state.
Publish(w) == /\ phase[w] = "writing"
              /\ (~Serialized \/ holder = w)
              /\ disk' = viewed[w] \cup {w}
              /\ phase' = [phase EXCEPT ![w] = "done"]
              /\ holder' = IF holder = w THEN "free" ELSE holder
              /\ UNCHANGED viewed

\* The process dies, the lock timeout expires, or the user declines after the
\* prompt: nothing is published, and a held transaction is rolled back.
Crash(w) == /\ phase[w] \in {"start", "writing"}
            /\ phase' = [phase EXCEPT ![w] = "crashed"]
            /\ holder' = IF holder = w THEN "free" ELSE holder
            /\ UNCHANGED <<viewed, disk>>

Next == \E w \in Writers : Acquire(w) \/ Read(w) \/ Publish(w) \/ Crash(w)
Spec == Init /\ [][Next]_vars

TypeOK == /\ phase \in [Writers -> {"start", "writing", "done", "crashed"}]
          /\ viewed \in [Writers -> SUBSET Writers]
          /\ disk \subseteq Writers
          /\ holder \in Writers \cup {"free"}

\* The reported bug: a completed recordTrust whose approval is not in the
\* file. Under Serialized this holds; under the pre-fix control it does not.
NoLostApproval == \A w \in Writers : phase[w] = "done" => w \in disk

\* A writer can only publish a store it read while nobody else could publish.
PublishHoldsLock == \A w \in Writers :
  (phase[w] = "writing" /\ Serialized) => holder = w
============================================================================
