# Distributed Systems: A Practical Overview

## What Makes a System "Distributed"?

A distributed system is a collection of independent computers that appears to users as a single coherent system. The key challenges arise from:

1. **Partial failure** - Parts of the system can fail independently
2. **Unreliable networks** - Messages can be lost, delayed, or duplicated
3. **No global clock** - Different nodes have different views of time

## The CAP Theorem

Eric Brewer's CAP theorem states that a distributed system can only provide two of three guarantees:

- **Consistency**: All nodes see the same data at the same time
- **Availability**: Every request receives a response
- **Partition tolerance**: System continues operating despite network partitions

In practice, network partitions happen, so you're really choosing between CP and AP systems.

### CP Systems (Consistency + Partition Tolerance)
- Examples: ZooKeeper, etcd, Consul
- Sacrifice availability during partitions
- Good for: coordination, leader election, configuration

### AP Systems (Availability + Partition Tolerance)
- Examples: Cassandra, DynamoDB, CouchDB
- Sacrifice consistency during partitions
- Good for: high-throughput, always-on services

## Consensus Algorithms

When nodes need to agree on something, they use consensus algorithms.

### Paxos
- Original consensus algorithm by Leslie Lamport
- Notoriously difficult to understand and implement
- Foundation for many other algorithms

### Raft
- Designed to be understandable
- Used in etcd, Consul, CockroachDB
- Separates leader election from log replication

### PBFT (Practical Byzantine Fault Tolerance)
- Handles malicious nodes
- Used in blockchain systems
- Higher overhead than crash-fault-tolerant algorithms

## Replication Strategies

### Single-Leader Replication
- One node accepts writes
- Followers replicate from leader
- Simple but leader is bottleneck

### Multi-Leader Replication
- Multiple nodes accept writes
- Must handle write conflicts
- Good for multi-datacenter deployments

### Leaderless Replication
- Any node accepts writes
- Uses quorum reads/writes
- Examples: Dynamo-style databases

## Consistency Models

From strongest to weakest:

1. **Linearizability** - Operations appear instantaneous
2. **Sequential consistency** - Operations appear in some sequential order
3. **Causal consistency** - Causally related operations appear in order
4. **Eventual consistency** - Given enough time, all replicas converge

## Partitioning (Sharding)

Distributing data across nodes:

### Hash Partitioning
- Hash key to determine partition
- Even distribution
- Range queries are inefficient

### Range Partitioning
- Ranges of keys on different nodes
- Good for range queries
- Risk of hot spots

## Conclusion

Building distributed systems requires understanding these fundamental concepts. Start simple, add complexity only when needed, and always plan for failure.
