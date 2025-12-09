# Distributed Frontier Manager: Code Blueprint

**Author**: Manus-F  
**Date**: 2025-12-06  
**Status**: Implementation Blueprint (Ready for Coding)

---

## Overview

This document provides a **complete code blueprint** for the Distributed Frontier Manager, which enables the U2 Planner to scale horizontally across multiple nodes while maintaining complete determinism. The frontier is implemented using Redis as a centralized, atomic priority queue.

---

## 1. Redis Schema

### 1.1. Keyspace Naming Strategy

All Redis keys follow a hierarchical naming convention:

```
frontier:{experiment_id}:{component}
```

Where:
- `{experiment_id}`: Unique identifier for the experiment (e.g., `arithmetic_simple_iter5`)
- `{component}`: Component name (e.g., `queue`, `seen`, `depths`, `items`)

**Example Keys**:
```
frontier:arithmetic_simple_iter5:queue
frontier:arithmetic_simple_iter5:seen
frontier:arithmetic_simple_iter5:depths
frontier:arithmetic_simple_iter5:items
frontier:arithmetic_simple_iter5:candidate_depths
frontier:arithmetic_simple_iter5:assignments
```

### 1.2. Data Structures

#### Priority Queue (Sorted Set)

**Key**: `frontier:{experiment_id}:queue`  
**Type**: Redis Sorted Set (ZSET)  
**Members**: Candidate hashes (SHA-256)  
**Scores**: Adjusted priorities (float)

```redis
ZADD frontier:exp123:queue 0.950001234 "a1b2c3d4..."
ZADD frontier:exp123:queue 0.870002345 "e5f6g7h8..."
```

**Score Calculation**:
```python
hash_int = int(candidate_hash[:8], 16)
tie_breaker = (hash_int % 1000) / 1_000_000.0  # [0, 0.001)
adjusted_priority = priority + tie_breaker
```

This ensures deterministic tie-breaking: same priority → sorted by hash.

#### Deduplication Set

**Key**: `frontier:{experiment_id}:seen`  
**Type**: Redis Set (SET)  
**Members**: Candidate hashes

```redis
SADD frontier:exp123:seen "a1b2c3d4..."
SISMEMBER frontier:exp123:seen "a1b2c3d4..."  # Returns 1 if exists
```

#### Depth Tracking (Hash Map)

**Key**: `frontier:{experiment_id}:depths`  
**Type**: Redis Hash (HASH)  
**Fields**: Depth (int as string)  
**Values**: Count (int as string)

```redis
HINCRBY frontier:exp123:depths "2" 1  # Increment depth 2 count by 1
HGET frontier:exp123:depths "2"  # Get count for depth 2
```

#### Item Storage (Hash Map)

**Key**: `frontier:{experiment_id}:items`  
**Type**: Redis Hash (HASH)  
**Fields**: Candidate hash  
**Values**: Candidate item (JSON string)

```redis
HSET frontier:exp123:items "a1b2c3d4..." '{"statement": {...}, "depth": 2, ...}'
HGET frontier:exp123:items "a1b2c3d4..."
```

#### Candidate Depths (Hash Map)

**Key**: `frontier:{experiment_id}:candidate_depths`  
**Type**: Redis Hash (HASH)  
**Fields**: Candidate hash  
**Values**: Depth (int as string)

```redis
HSET frontier:exp123:candidate_depths "a1b2c3d4..." "2"
HGET frontier:exp123:candidate_depths "a1b2c3d4..."
```

#### Worker Assignments (Hash Map)

**Key**: `frontier:{experiment_id}:assignments`  
**Type**: Redis Hash (HASH)  
**Fields**: Candidate hash  
**Values**: Worker ID (string)

```redis
HSET frontier:exp123:assignments "a1b2c3d4..." "worker_0"
HGET frontier:exp123:assignments "a1b2c3d4..."
```

---

## 2. Atomic Operation List

All frontier operations use Redis atomic commands to prevent race conditions.

### 2.1. Push Operation

**Purpose**: Add a candidate to the frontier.

**Atomic Sequence**:
1. `SADD seen_key candidate_hash` → Check if new (returns 1 if new, 0 if duplicate)
2. If new:
   - `ZADD queue_key adjusted_priority candidate_hash` → Add to priority queue
   - `HINCRBY depths_key depth 1` → Increment depth count
   - `HSET items_key candidate_hash item_json` → Store item
   - `HSET candidate_depths_key candidate_hash depth` → Store depth
   - `HSET assignments_key candidate_hash worker_id` → Track worker

**Pseudocode**:
```python
def push(item, priority, depth, worker_id):
    candidate_hash = sha256(item)
    
    # Atomic deduplication check
    is_new = redis.sadd(seen_key, candidate_hash)
    if not is_new:
        return False  # Duplicate
    
    # Compute adjusted priority (deterministic tie-breaking)
    hash_int = int(candidate_hash[:8], 16)
    tie_breaker = (hash_int % 1000) / 1_000_000.0
    adjusted_priority = priority + tie_breaker
    
    # Atomic push
    redis.zadd(queue_key, {candidate_hash: adjusted_priority})
    redis.hincrby(depths_key, str(depth), 1)
    redis.hset(items_key, candidate_hash, item)
    redis.hset(candidate_depths_key, candidate_hash, str(depth))
    redis.hset(assignments_key, candidate_hash, worker_id)
    
    # Prune if needed
    prune_if_needed()
    
    return True
```

### 2.2. Pop Operation

**Purpose**: Retrieve the highest-priority candidate.

**Atomic Sequence**:
1. `ZPOPMIN queue_key 1` → Pop lowest score (highest priority) atomically
2. `HGET items_key candidate_hash` → Retrieve item
3. `HGET candidate_depths_key candidate_hash` → Retrieve depth
4. `HINCRBY depths_key depth -1` → Decrement depth count
5. `HSET assignments_key candidate_hash worker_id` → Update assignment

**Pseudocode**:
```python
def pop(worker_id):
    # Atomic pop (lowest score = highest priority)
    result = redis.zpopmin(queue_key, count=1)
    if not result:
        return None  # Empty
    
    candidate_hash, priority = result[0]
    
    # Retrieve item and depth
    item = redis.hget(items_key, candidate_hash)
    depth = int(redis.hget(candidate_depths_key, candidate_hash))
    
    # Update depth tracking
    redis.hincrby(depths_key, str(depth), -1)
    
    # Update assignment
    redis.hset(assignments_key, candidate_hash, worker_id)
    
    return item, priority, depth
```

### 2.3. Prune Operation

**Purpose**: Enforce beam width limit.

**Atomic Sequence**:
1. `ZCARD queue_key` → Get current size
2. If size > max_beam_width:
   - `ZPOPMAX queue_key (size - max_beam_width)` → Remove lowest-priority items
   - For each removed item:
     - `HGET candidate_depths_key candidate_hash` → Get depth
     - `HINCRBY depths_key depth -1` → Decrement depth count

**Pseudocode**:
```python
def prune_if_needed():
    current_size = redis.zcard(queue_key)
    if current_size <= max_beam_width:
        return
    
    num_to_remove = current_size - max_beam_width
    
    # Atomic removal (highest scores = lowest priorities)
    removed = redis.zpopmax(queue_key, count=num_to_remove)
    
    # Update depth tracking
    for candidate_hash, _ in removed:
        depth = int(redis.hget(candidate_depths_key, candidate_hash))
        redis.hincrby(depths_key, str(depth), -1)
```

---

## 3. Tie-Break Priority Derivation

### 3.1. Problem

When two candidates have the same logical priority, Redis's `ZPOPMIN` uses lexicographic ordering of the member (candidate hash). Without a tie-breaker, the order would be arbitrary and non-deterministic across different Redis versions or configurations.

### 3.2. Solution

Add a **hash-based tie-breaker** to the priority score:

```python
def compute_adjusted_priority(priority: float, candidate_hash: str) -> float:
    """
    Compute adjusted priority with deterministic tie-breaking.
    
    Args:
        priority: Logical priority (higher = more important)
        candidate_hash: SHA-256 hash of candidate
        
    Returns:
        Adjusted priority for Redis sorted set
    """
    # Extract first 8 hex digits (32 bits)
    hash_int = int(candidate_hash[:8], 16)
    
    # Compute tie-breaker in range [0, 0.001)
    tie_breaker = (hash_int % 1000) / 1_000_000.0
    
    # Add to priority (small enough not to affect logical ordering)
    adjusted_priority = priority + tie_breaker
    
    return adjusted_priority
```

### 3.3. Guarantees

- **Deterministic**: Same hash → same tie-breaker
- **Stable**: Tie-breaker is small enough (< 0.001) not to affect logical priority ordering
- **Unique**: Hash-based tie-breaker ensures no two candidates have exactly the same adjusted priority

---

## 4. Worker Coordination Loop

### 4.1. Worker Lifecycle

Each worker runs a loop:
1. **Pop** candidate from frontier
2. **Execute** candidate with local PRNG
3. **Buffer** result locally
4. **Generate** new candidates from result
5. **Push** new candidates to frontier
6. **Repeat** until cycle budget exhausted

### 4.2. Worker Implementation

```python
class DistributedWorker:
    def __init__(self, worker_id, experiment_id, redis_url, prng, execute_fn):
        self.worker_id = worker_id
        self.experiment_id = experiment_id
        self.prng = prng
        self.execute_fn = execute_fn
        self.frontier = DistributedFrontierManager(experiment_id, redis_url, prng)
        self.results_buffer = []
    
    def run_cycle(self, cycle):
        """Run a single cycle."""
        cycle_prng = self.prng.for_path("cycle", str(cycle))
        cycle_start_time_ms = time.time_ns() // 1_000_000
        
        processed = 0
        generated = 0
        
        while True:
            # Pop candidate
            result = self.frontier.pop(f"worker_{self.worker_id}")
            if not result:
                break  # Frontier empty
            
            item, priority, depth = result
            
            # Get execution PRNG
            candidate_hash = hashlib.sha256(item.encode('utf-8')).hexdigest()
            exec_prng = cycle_prng.for_path("execute", candidate_hash)
            exec_seed = int(exec_prng.seed_canonical[2:], 16)
            
            # Execute
            success, exec_result = self.execute_fn(item, exec_seed, cycle_start_time_ms)
            processed += 1
            
            # Buffer result
            self.results_buffer.append({
                "candidate_hash": candidate_hash,
                "item": item,
                "success": success,
                "result": exec_result.to_canonical_dict(),
                "cycle": cycle,
                "worker_id": self.worker_id,
            })
            
            # Generate new candidates
            if success:
                new_candidates = generate_candidates(exec_result, cycle)
                for new_cand in new_candidates:
                    self.frontier.push(
                        new_cand["item"],
                        new_cand["priority"],
                        new_cand["depth"],
                        f"worker_{self.worker_id}",
                    )
                    generated += 1
            
            # Check budget
            elapsed_ms = (time.time_ns() // 1_000_000) - cycle_start_time_ms
            if elapsed_ms >= CYCLE_BUDGET_MS:
                break
        
        return WorkerCycleResult(
            worker_id=self.worker_id,
            cycle=cycle,
            candidates_processed=processed,
            candidates_generated=generated,
        )
    
    def flush_results(self):
        """Flush results buffer for merging."""
        results = self.results_buffer.copy()
        self.results_buffer.clear()
        return results
```

### 4.3. Coordinator Loop

The coordinator orchestrates workers:
1. **Initialize** workers with partitioned PRNGs
2. **Run** cycle on all workers in parallel
3. **Flush** worker result buffers
4. **Merge** results into canonical global state
5. **Repeat** for next cycle

```python
class DistributedCoordinator:
    def __init__(self, config, num_workers, redis_url):
        self.config = config
        self.num_workers = num_workers
        self.redis_url = redis_url
        self.partitioner = PRNGPartitioner(config.master_seed, config.slice_name, num_workers)
        self.merger = StateMerger(config.experiment_id)
    
    def run(self, execute_fn):
        """Run distributed experiment."""
        # Initialize workers
        workers = [
            DistributedWorker(
                worker_id=i,
                experiment_id=self.config.experiment_id,
                redis_url=self.redis_url,
                prng=self.partitioner.get_worker_prng(i),
                execute_fn=execute_fn,
            )
            for i in range(self.num_workers)
        ]
        
        # Run cycles
        for cycle in range(self.config.total_cycles):
            # Run cycle on all workers (parallel)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(worker.run_cycle, cycle) for worker in workers]
                worker_results = [future.result() for future in as_completed(futures)]
            
            # Flush result buffers
            worker_result_buffers = [worker.flush_results() for worker in workers]
            
            # Merge results
            merge_result = self.merger.merge_cycle_results(cycle, worker_result_buffers)
            
            print(f"Cycle {cycle}: {merge_result.total_processed} processed")
        
        # Merge traces
        self.merger.merge_traces(worker_traces, merged_trace_path)
```

---

## 5. Merge-Sort Reconciliation Logic

### 5.1. Problem

Workers execute asynchronously. Results must be merged into a **canonical global state** that is independent of execution order.

### 5.2. Solution

Use **deterministic sorting** before deduplication:

```python
class StateMerger:
    def merge_cycle_results(self, cycle, worker_results):
        """Merge results from all workers for a cycle."""
        # Flatten all results
        all_results = []
        for worker_id, results in enumerate(worker_results):
            for result in results:
                result["worker_id"] = worker_id
                all_results.append(result)
        
        # CANONICAL SORT: (cycle, worker_id, candidate_hash)
        all_results.sort(key=lambda r: (
            r["cycle"],
            r["worker_id"],
            r["candidate_hash"],
        ))
        
        # Deduplicate (keep first occurrence)
        seen_hashes = set()
        unique_results = []
        for result in all_results:
            h = result["candidate_hash"]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_results.append(result)
        
        return CycleMergeResult(
            cycle=cycle,
            total_processed=len(unique_results),
            results=unique_results,
        )
    
    def merge_traces(self, worker_traces, output_path):
        """Merge worker trace files into canonical global trace."""
        # Read all events
        all_events = []
        for worker_id, trace_path in enumerate(worker_traces):
            with open(trace_path, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    event["worker_id"] = worker_id
                    all_events.append(event)
        
        # CANONICAL SORT: (cycle, timestamp_ms, worker_id, candidate_hash)
        all_events.sort(key=lambda e: (
            e["cycle"],
            e["timestamp_ms"],
            e["worker_id"],
            e.get("data", {}).get("candidate_hash", ""),
        ))
        
        # Write merged trace
        with open(output_path, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event, sort_keys=True) + '\n')
```

### 5.3. Guarantees

- **Deterministic**: Same worker results → same merged state
- **Independent of Execution Order**: Sorting imposes canonical order
- **Deduplication**: First occurrence wins (deterministic)

---

## 6. Implementation Checklist

### Phase 1: Redis Infrastructure
- [ ] Set up Redis server (local or cluster)
- [ ] Implement `DistributedFrontierManager` class
- [ ] Implement `push()` with atomic operations
- [ ] Implement `pop()` with atomic operations
- [ ] Implement `prune_if_needed()` with beam width enforcement
- [ ] Test atomic operations with concurrent clients

### Phase 2: Worker Coordination
- [ ] Implement `DistributedWorker` class
- [ ] Implement `run_cycle()` loop
- [ ] Implement result buffering
- [ ] Test worker with mock execute function

### Phase 3: State Merging
- [ ] Implement `StateMerger` class
- [ ] Implement `merge_cycle_results()` with canonical sorting
- [ ] Implement `merge_traces()` with canonical sorting
- [ ] Test deduplication logic

### Phase 4: Coordinator
- [ ] Implement `DistributedCoordinator` class
- [ ] Implement PRNG partitioning
- [ ] Implement parallel worker execution
- [ ] Test end-to-end distributed experiment

### Phase 5: Determinism Validation
- [ ] Run same experiment with 1 worker and N workers
- [ ] Compare final state hashes
- [ ] Compare trace hashes
- [ ] Verify all four invariants

---

## 7. Redis Configuration

### 7.1. Recommended Settings

```redis
# redis.conf

# Persistence (for fault tolerance)
save 900 1
save 300 10
save 60 10000

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Atomic operations
multi-exec-commands yes

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
```

### 7.2. Connection String

```python
REDIS_URL = "redis://localhost:6379/0"  # Local
REDIS_URL = "redis://redis-cluster:6379/0"  # Cluster
```

---

## 8. Performance Considerations

### 8.1. Latency

- **Target**: <1ms per operation (local Redis)
- **Target**: <10ms per operation (remote Redis)

### 8.2. Throughput

- **Target**: 10,000 push/pop operations per second per worker

### 8.3. Optimization Strategies

1. **Pipeline Commands**: Use Redis pipelining to batch operations
2. **Connection Pooling**: Reuse connections across workers
3. **Lua Scripts**: Use Lua scripts for complex atomic operations
4. **Compression**: Compress item JSON before storing in Redis

---

**Status**: Blueprint complete. Ready for implementation.
