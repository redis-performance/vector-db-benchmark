

---

## **Overview**

Add mixed workload capabilities to vector-db-benchmark to measure search performance under concurrent insert load. This enables realistic testing of production scenarios where vector databases handle both read and write traffic simultaneously.

### **Technical Approach**
```python
# In BaseSearcher.search_all() or equivalent:
# - Each worker processes a chunk of tasks (queries).
# - For each task, with probability insert_fraction, do an insert (using test_set vector), else do a search.
# - Use a parameter like insert_fraction to control the ratio.
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

**File 1: `engine/base_client/search.py`**
```python
def process_chunk(chunk, search_one, insert_one, insert_fraction=0.1, test_set=None):
    results = []
    for i, query in enumerate(chunk):
        if random.random() < insert_fraction:
            # Insert: use a vector from test_set
            vector_id, vector, metadata = test_set[i % len(test_set)]
            result = insert_one(vector_id, vector, metadata)
        else:
            # Search
            result = search_one(query)
        results.append(result)
    return results
```

**File 2: `worker_function`**
```python
def worker_function(self, distance, search_one, insert_one, chunk, result_queue, insert_fraction=0.1, test_set=None):
    self.init_client(self.host, distance, self.connection_params, self.search_params)
    self.setup_search()
    start_time = time.perf_counter()
    results = process_chunk(chunk, search_one, insert_one, insert_fraction, test_set)
    result_queue.put((start_time, results))
```

**File 3: `BaseSearcher.search_all()`**
- When creating worker processes, pass `search_one`, `insert_one`, `insert_fraction`, and `test_set` as arguments to each worker.

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
        "insert_fraction": 0.1
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
    "insert_fraction": 0.1,
    "search_workers": 8
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

**File 2: `worker_function`**
```python
def worker_function(self, distance, search_one, insert_one, chunk, result_queue, insert_fraction=0.1, test_set=None):
    self.init_client(self.host, distance, self.connection_params, self.search_params)
    self.setup_search()
    start_time = time.perf_counter()
    results = process_chunk(chunk, search_one, insert_one, insert_fraction, test_set)
    result_queue.put((start_time, results))
```
