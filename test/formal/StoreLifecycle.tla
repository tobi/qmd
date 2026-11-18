------------------------ MODULE StoreLifecycle ------------------------
EXTENDS Naturals
\* Check: bunx everystate test/formal/StoreLifecycle.tla
\* Mutation: bunx everystate -config test/formal/StoreLifecycleBroken.cfg test/formal/StoreLifecycle.tla
\* One SDK store, one admitted operation and two close callers. The model
\* covers public SDK operations, not direct use of the exposed internal store.
\* EnforceDrain=FALSE removes the active-operation and shared-close barriers.
\* Timeouts, operation failure and reentrant close are outside this model.
CONSTANT EnforceDrain
VARIABLES state, active, firstReturned, secondReturned
vars == <<state, active, firstReturned, secondReturned>>
Init == /\ state = "open" /\ active = FALSE
        /\ firstReturned = FALSE /\ secondReturned = FALSE

\* SDK operation admission is synchronous before the operation awaits.
StartOperation == /\ state = "open" /\ ~active
                  /\ active' = TRUE
                  /\ UNCHANGED <<state, firstReturned, secondReturned>>
FinishOperation == /\ active
                   /\ active' = FALSE
                   /\ UNCHANGED <<state, firstReturned, secondReturned>>
\* close prevents further admission before waiting for existing operations.
BeginClose == /\ state = "open"
              /\ state' = "closing"
              /\ UNCHANGED <<active, firstReturned, secondReturned>>
FinishClose == /\ state = "closing"
               /\ (~EnforceDrain \/ ~active)
               /\ state' = "closed"
               /\ UNCHANGED <<active, firstReturned, secondReturned>>
ReturnFirst == /\ state = "closed" /\ ~firstReturned
               /\ firstReturned' = TRUE
               /\ UNCHANGED <<state, active, secondReturned>>
ReturnSecond == /\ state \in {"closing", "closed"} /\ ~secondReturned
                /\ (~EnforceDrain \/ state = "closed")
                /\ secondReturned' = TRUE
                /\ UNCHANGED <<state, active, firstReturned>>
Next == StartOperation \/ FinishOperation \/ BeginClose \/ FinishClose \/ ReturnFirst \/ ReturnSecond
Spec == Init /\ [][Next]_vars
TypeOK == /\ state \in {"open", "closing", "closed"}
          /\ active \in BOOLEAN /\ firstReturned \in BOOLEAN /\ secondReturned \in BOOLEAN
NoCloseDuringOperation == active => state # "closed"
NoEarlyCloseReturn == (firstReturned \/ secondReturned) => state = "closed"
=======================================================================
