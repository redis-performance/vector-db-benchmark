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
- **Consistent Multiprocessing**: Use same pattern as search workers for insert workers
- **Existing CLI Patterns**: Just add `--mixed-workload` flag

### **Technical Approach**
```python
# Single method addition to BaseClient
def _run_with_background_inserts(self, dataset, mixed_params, num_queries):
    # Start background processes that insert vectors using existing uploader pattern
    # Run normal search experiment (unchanged)
    # Stop inserts and collect stats
    # Add insert metrics to search results
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

**File 1: `engine/base_client/client.py`** (~30 lines)
```python
def run_experiment(
    # ... existing parameters ...
    mixed_workload_params: Optional[dict] = None,  # NEW
):
    if mixed_workload_params:
        return self._run_with_background_inserts(dataset, mixed_workload_params, num_queries)
    # ... existing code unchanged ...

def _run_with_background_inserts(self, dataset, mixed_params, num_queries):
    """Run search with concurrent inserts using multiprocessing like existing patterns"""
    import multiprocessing as mp
    from multiprocessing import Process, Queue
    import numpy as np
    
    insert_stats = {"count": 0, "lock": mp.Lock()}
    stop_event = mp.Event()
    result_queue = Queue()
    
    def insert_worker_function(uploader, dataset, worker_id, insert_stats, stop_event, result_queue):
        # Create own connection like search workers do
        uploader.init_client(
            uploader.host, 
            dataset.config.distance, 
            uploader.connection_params, 
            uploader.upload_params
        )
        
        vector_id = 1000000 + worker_id * 100000  # Unique range per worker
        local_count = 0
        start_time = time.perf_counter()
        
        while not stop_event.is_set():
            # Generate and insert vector using existing upload_batch pattern
            new_vector = np.random.random(dataset.config.vector_size).astype(np.float32)
            uploader.upload_batch([vector_id], [new_vector], [{}])
            local_count += 1
            vector_id += 1
            time.sleep(0.01)  # Rate limiting
        
        result_queue.put((start_time, local_count))
        uploader.delete_client()
    
    # Start insert workers (same pattern as search multiprocessing)
    insert_workers = mixed_params.get("insert_workers", 4)
    processes = []
    for worker_id in range(insert_workers):
        process = Process(
            target=insert_worker_function, 
            args=(self.uploader, dataset, worker_id, insert_stats, stop_event, result_queue)
        )
        processes.append(process)
        process.start()
    
    # Run normal search experiment (UNCHANGED!)
    results = self.run_experiment(dataset, skip_upload=True, num_queries=num_queries)
    
    # Stop inserts and collect stats
    stop_event.set()
    
    # Collect insert results from all workers
    total_inserts = 0
    for _ in processes:
        _, worker_inserts = result_queue.get()
        total_inserts += worker_inserts
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    # Add insert stats to results
    results["search"]["mixed_workload"] = {
        "insert_count": total_inserts,
        "insert_rate": total_inserts / results["search"]["total_time"],
        "insert_workers": insert_workers
    }
    
    return results
```

**File 2: `run.py`** (~5 lines)
```python
@app.command()
def run(
    # ... all existing parameters unchanged ...
    mixed_workload: bool = typer.Option(False, help="Enable concurrent inserts during search"),
):
    # MINIMAL change:
    if mixed_workload:
        mixed_params = {"insert_workers": 4}  # Simple default
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
```json
{
  "mixed_workload_params": {
    "insert_workers": 4
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
      "insert_count": 720, 
      "insert_rate": 13.6,
      "insert_workers": 4
    }
  }
}
```

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
| Add mixed_workload_params | 1 hour | Extend BaseClient.run_experiment() |
| Background insert workers | 3 hours | Implement multiprocessing approach |
| CLI integration | 1 hour | Add --mixed-workload flag |
| Testing | 2 hours | Validate with small dataset |

**Total: ~7 hours**

---

## **Summary**

**Goal**: Measure search performance under concurrent insert load  
**Approach**: Consistent multiprocessing with existing search/upload patterns  
**Changes**: ~40 lines across 2 files  
**Benefits**: Realistic production testing with architectural consistency  
