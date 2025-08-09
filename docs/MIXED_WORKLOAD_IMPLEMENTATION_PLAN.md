
## TODO
Instead of having a separate pool of search and insert workers, have a single pool as before. 
However, some fraction of search requests will be converted to inserts. Reuse the test_set for vector inserts, and do not use random vectors. 


# **Mixed Workload Implementation Plan**
*Simple Strategy for Concurrent Search and Insert Operations*

**Date:** August 8, 2025  
**Repository:** redis-scripts/vector-db-benchmark  
**Branch:** mixed_workload  

---

## **Overview**

Add mixed workload capabilities to vector-db-benchmark to measure search performance under concurrent insert load. This enables realistic testing of production scenarios where vector databases handle both read and write traffic simultaneously.

### **Core Approach**
- **Concurrent Operations**: Search existing vectors while inserting new ones
- **Shared Graph**: Both operations work on the same HNSW graph structure  
- **Insert-Only**: Append new vectors (no updates/deletes) to avoid accuracy complications
- **Performance Focus**: Measure search QPS and latency degradation under insert load

---

## **Implementation Strategy**

### **Ultra-Simple Design**
- **Reuse Everything**: Leverage existing search and upload infrastructure
- **Zero New Files**: No new classes, configs, or dependencies
- **Unified Worker Pool**: Extend existing worker pattern to handle both search and insert tasks
- **Existing CLI Patterns**: Just add `--mixed-workload` flag

### **Technical Approach**
```python
# Extend existing BaseSearcher.search_all() to support mixed workloads
# Some workers get search chunks, others get insert tasks
# Same worker_function pattern, different task types
```

---

## **Usage Scenarios**

### **Fresh Setup + Mixed Workload**
```bash
python run.py --engines redis --datasets dataset --mixed-workload
```
Flow: Configure → Upload → Search + Concurrent Inserts

### **Existing Data + Mixed Workload**  
```bash
# Previous run created the index
python run.py --engines redis --datasets dataset

# Mixed workload on existing data using existing --skip-upload
python run.py --engines redis --datasets dataset --skip-upload --mixed-workload
```
Flow: Search + Concurrent Inserts (reuses existing data)

---

## **Implementation Details**

### **Code Changes Required**

**File 1: `engine/base_client/search.py`** (~25 lines)
```python
# Add insert_one method to BaseSearcher for consistency
@classmethod
def insert_one(cls, vector_id: int, vector: List[float], metadata: Optional[dict] = None):
    """Insert a single vector - raw database operation (like search_one)"""
    # Delegate to uploader's upload_batch with single item
    # This will be overridden by engines that have direct insert methods
    raise NotImplementedError("insert_one must be implemented by each engine")

@classmethod
def _insert_one(cls, insert_record):
    """Timed insert operation (like _search_one)"""
    vector_id, vector, metadata = insert_record
    start = time.perf_counter()
    cls.insert_one(vector_id, vector, metadata)
    end = time.perf_counter()
    
    # Return consistent metrics (no precision for inserts)
    return 1.0, end - start  # Always "successful", return latency

# Extend worker_function to support both task types
def worker_function(self, distance, task_func, chunk_or_task, result_queue):
    self.init_client(
        self.host,
        distance, 
        self.connection_params,
        self.search_params,
    )
    self.setup_search()

    start_time = time.perf_counter()
    results = task_func(chunk_or_task)  # process_chunk OR process_insert_chunk
    result_queue.put((start_time, results))

# Mirror process_chunk for inserts
def process_insert_chunk(insert_chunk):
    """Process insert operations (mirrors process_chunk)"""
    # insert_chunk contains insert records instead of queries
    # Same pattern as search: [task_func(item) for item in chunk]
    return [BaseSearcher._insert_one(record) for record in insert_chunk]
```

**File 2: `engine/base_client/client.py`** (~15 lines)
```python
def run_experiment(
    # ... existing parameters ...
    mixed_workload_params: Optional[dict] = None,  # NEW
):
    if mixed_workload_params:
        # Generate insert chunks and modify searcher for mixed workload
        return self._run_mixed_workload(dataset, mixed_workload_params, num_queries)
    # ... existing code unchanged ...

def _run_mixed_workload(self, dataset, mixed_params, num_queries):
    """Generate insert records and configure searcher for unified workers"""
    insert_workers = mixed_params.get("insert_workers", 2)
    
    # Generate insert chunks (same pattern as query chunks)
    insert_chunk_size = 100  # Inserts per chunk
    insert_chunks = []
    for worker_id in range(insert_workers):
        vector_id_start = 1000000 + worker_id * 100000
        chunk = []
        for i in range(insert_chunk_size):
            vector = np.random.random(dataset.config.vector_size).astype(np.float32).tolist()
            chunk.append((vector_id_start + i, vector, {}))
        insert_chunks.append(chunk)
    
    # Configure searcher for mixed workload
    self.searchers[0].search_params["mixed_workload"] = {
        "insert_chunks": insert_chunks,
        "dataset": dataset
    }
    
    # Run normal search_all (now processes both search and insert chunks)
    results = self.searchers[0].search_all(dataset.config.distance, dataset.get_queries(), num_queries)
    
    # Cleanup
    self.searchers[0].search_params.pop("mixed_workload", None)
    
    return {"search": results}
```

**File 3: Extend `BaseSearcher.search_all()`** (~15 lines)
```python
# In BaseSearcher.search_all(), around line 160 where processes are created:

# Check if mixed workload is enabled
mixed_config = self.search_params.get("mixed_workload", None)
if mixed_config:
    insert_chunks = mixed_config["insert_chunks"]
    
    # Create search worker processes (unchanged)
    search_processes = []
    for chunk in query_chunks:
        process = Process(target=worker_function, args=(self, distance, process_chunk, chunk, result_queue))
        search_processes.append(process)
    
    # Create insert worker processes (same pattern!)
    insert_processes = []
    for chunk in insert_chunks:
        process = Process(target=worker_function, args=(self, distance, process_insert_chunk, chunk, result_queue))
        insert_processes.append(process)
    
    processes = search_processes + insert_processes
else:
    # Original search-only processes
    processes = []
    for chunk in query_chunks:
        process = Process(target=worker_function, args=(self, distance, process_chunk, chunk, result_queue))
        processes.append(process)
```

**File 4: Engine-specific `insert_one` implementations** (~5 lines each)
```python
# Example: engine/clients/redis/search.py
@classmethod
def insert_one(cls, vector_id: int, vector: List[float], metadata: Optional[dict] = None):
    """Redis-specific single vector insert"""
    cls.client.hset(
        str(vector_id),
        mapping={"vector": np.array(vector).astype(np.float32).tobytes(), **(metadata or {})}
    )

# Example: engine/clients/qdrant/search.py  
@classmethod
def insert_one(cls, vector_id: int, vector: List[float], metadata: Optional[dict] = None):
    """Qdrant-specific single vector insert"""
    cls.client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[{"id": vector_id, "vector": vector, "payload": metadata or {}}],
        wait=False,
    )
```

### **Redis-Specific Implementation Details**

**File 4a: `engine/clients/redis/search.py`** (~15 lines)
```python
@classmethod
def insert_one(cls, vector_id: int, vector: List[float], metadata: Optional[dict] = None):
    """RediSearch single vector insert using HSET"""
    vector_key = str(vector_id)
    meta = metadata or {}
    payload = {}
    geopoints = {}
    
    # Process metadata (same logic as upload_batch)
    for k, v in meta.items():
        if k == "labels" and isinstance(v, list):
            payload[k] = ";".join(v)  # Special handling for labels
        elif v is not None and not isinstance(v, dict) and not isinstance(v, list):
            payload[k] = v
        elif isinstance(v, dict) and "lon" in v and "lat" in v:
            # Handle geopoints
            geopoints[k] = ",".join(map(str, convert_to_redis_coords(v["lon"], v["lat"])))
    
    # Insert using HSET (same as upload_batch but single operation)
    cls.client.hset(
        vector_key,
        mapping={
            "vector": np.array(vector).astype(cls.np_data_type).tobytes(),
            **payload,
            **geopoints,
        },
    )
```

**File 4b: `engine/clients/vectorsets/search.py`** (~10 lines)  
```python
@classmethod
def insert_one(cls, vector_id: int, vector: List[float], metadata: Optional[dict] = None):
    """Redis Vector Sets single vector insert using VADD"""
    # Use same parameters from upload configuration
    upload_params = cls.search_params.get("upload_params", {})
    hnsw_params = upload_params.get("hnsw_config", {})
    M = hnsw_params.get("M", 16)
    efc = hnsw_params.get("EF_CONSTRUCTION", 200)
    quant = hnsw_params.get("quant", "NOQUANT")
    
    # Convert vector to bytes (same as upload_batch)
    vec_bytes = np.array(vector).astype(np.float32).tobytes()
    
    # Insert using VADD command (same as upload_batch but single operation)
    cls.client.execute_command("VADD", "idx", "FP32", vec_bytes, vector_id, quant, "M", M, "EF", efc, "CAS")
```

**File 5: `run.py`** (~5 lines)
```python
@app.command()
def run(
    # ... all existing parameters unchanged ...
    mixed_workload: bool = typer.Option(False, help="Enable concurrent inserts during search"),
):
    # MINIMAL change:
    if mixed_workload:
        mixed_params = {"insert_workers": 2}  # Default: 2 insert workers
        client.run_experiment(
            dataset, skip_upload, skip_search, skip_if_exists,
            parallels, upload_start_idx, upload_end_idx, queries, ef_runtime,
            mixed_workload_params=mixed_params
        )
    else:
        # UNCHANGED - existing code path
        client.run_experiment(
            dataset, skip_upload, skip_search, skip_if_exists,
            parallels, upload_start_idx, upload_end_idx, queries, ef_runtime
        )
```

### **Configuration**

#### **Mixed Workload Parameters**
```json
{
  "mixed_workload_params": {
    "insert_workers": 2
  }
}
```

#### **Redis Engine Configurations**

**RediSearch Configuration**
```json
{
  "search_params": {
    "algorithm": "hnsw",
    "search_params": {"ef": 200, "data_type": "FLOAT32"},
    "upload_params": {
      "data_type": "FLOAT32",
      "parallel": 1,
      "batch_size": 64
    }
  }
}
```

**Vector Sets Configuration**  
```json
{
  "search_params": {
    "search_params": {"ef": 200},
    "upload_params": {
      "hnsw_config": {"M": 16, "EF_CONSTRUCTION": 200, "quant": "NOQUANT"}
    }
  }
}
```

### **Results Format**
```json
{
  "search": {
    "total_time": 52.8,
    "rps": 1876.3,
    "mean_precisions": 0.945,
    "p95_time": 0.103,
    "mixed_workload": {
      "insert_count": 360, 
      "insert_rate": 6.8,
      "insert_workers": 2,
      "search_workers": 6
    }
  }
}
```

---

## **Redis-Specific Implementation Details**

### **Two Redis Engines Support**

**RediSearch (`engine/clients/redis/`)**
- **Index Type**: HNSW/FLAT using RediSearch module  
- **Insert Method**: `HSET` with vector field and metadata
- **Data Format**: Vector as bytes blob + metadata fields
- **Cluster Support**: Automatic shard distribution via `get_primaries()`

**Vector Sets (`engine/clients/vectorsets/`)**  
- **Index Type**: Native HNSW using Vector Sets module
- **Insert Method**: `VADD` command with HNSW parameters
- **Data Format**: FP32 bytes + element ID + quantization settings
- **Cluster Support**: Automatic shard distribution via `get_primaries()`

### **Implementation Strategy**

**1. Leverage Existing Upload Logic**
```python
# RediSearch: Reuse hset logic from RedisUploader.upload_batch()
# Vector Sets: Reuse VADD logic from RedisVsetUploader.upload_batch()
# Both: Handle metadata processing, geopoints, data types consistently
```

**2. Configuration Inheritance**
```python
# Access upload_params from search_params for consistency
upload_params = cls.search_params.get("upload_params", {})
hnsw_config = upload_params.get("hnsw_config", {})
```

**3. Cluster Awareness**
```python
# Both engines inherit cluster connection handling
# insert_one uses same cls.client (already cluster-aware)
# No additional cluster logic needed
```

### **Data Type Consistency**  

**RediSearch Vector Format**
```python
# Uses np_data_type from init_client (FLOAT32/BFLOAT16/etc)
vector_bytes = np.array(vector).astype(cls.np_data_type).tobytes()
```

**Vector Sets Format**  
```python
# Always FP32 for Vector Sets
vector_bytes = np.array(vector).astype(np.float32).tobytes()
```

### **Metadata Handling**

**RediSearch Metadata Processing**
```python
# Special cases from upload_batch:
# - labels: list → semicolon-separated string
# - geopoints: {lat, lon} → "lat,lon" string  
# - other: direct field mapping
```

**Vector Sets Metadata**
```python  
# Vector Sets: metadata handled via SETATTR (future enhancement)
# Current: focus on vector insertion only
# metadata parameter reserved for future SETATTR integration
```

### **Performance Considerations**

**Single Insert Efficiency**
- **RediSearch**: Direct `HSET` - very fast single operation
- **Vector Sets**: Direct `VADD` - bypasses pipeline overhead
- **Both**: No pipeline creation/execution overhead for single inserts

**Cluster Distribution**
- **Automatic**: Both use existing cluster-aware connections
- **Load Balancing**: Insert workers distribute across shards naturally  
- **Connection Reuse**: Same connection pool as search operations

---

## **Expected Outcomes**

### **Performance Metrics**
- **Search QPS**: Throughput during concurrent inserts
- **Search Latency**: P50, P95, P99 under insert load  
- **Insert Rate**: Actual inserts achieved per second
- **Resource Usage**: CPU, memory during mixed operations

### **Value**
- **Realistic Testing**: Measure production-like concurrent workloads
- **Database Comparison**: Compare mixed workload performance across engines
- **Capacity Planning**: Determine sustainable insert rates for given search SLAs

---

## **Implementation Timeline**

| Task | Duration | Description |
|------|----------|-------------|
| Add insert_one to BaseSearcher | 1 hour | Mirror search_one pattern for inserts |
| Extend worker_function | 1 hour | Support both search and insert chunks |
| Engine-specific insert_one | 2 hours | Implement for all engines |
| CLI integration | 1 hour | Add --mixed-workload flag |
| Testing | 2 hours | Validate with small dataset |

**Total: ~7 hours**

---

## **Summary**

**Goal**: Measure search performance under concurrent insert load  
**Approach**: Perfect consistency using insert_one mirroring search_one pattern  
**Changes**: ~60 lines across 4+ files  
**Benefits**: Architectural consistency with single-operation semantics  
