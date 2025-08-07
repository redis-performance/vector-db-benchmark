# **Mixed Workload Implementation Plan**
*Comprehensive Strategy for Concurrent Search and Update Operations*

**Date Created:** August 7, 2025  
**Repository:** redis-scripts/vector-db-benchmark  
**Branch:** svs-production  

---

## **Executive Summary**

This document outlines a comprehensive implementation plan for adding mixed workload capabilities to vector-db-benchmark. The mixed workload feature will enable concurrent search and update operations to simulate realistic production scenarios where vector databases handle both read and write traffic simultaneously.

### **Key Design Principles**
1. **Leverage Existing Infrastructure**: Use proven dataset partitioning, multiprocessing, and search architectures
2. **Preserve Performance**: Don't modify existing search process model that's already optimized
3. **Clean Separation**: Mixed workload as additive capability, not architectural change
4. **Production Realistic**: Large search pool + small high-churn update pool  
5. **Comprehensive Analysis**: Detailed performance impact measurement and recommendations

### **Architecture Strategy**
- **Large Search Pool (90%)**: Maintains realistic search performance characteristics
- **Small Update Pool (10%)**: High-frequency updates for system stress testing
- **Timeline-Based Execution**: Warmup → Mixed → Cooldown phases for clear performance impact measurement
- **Existing Infrastructure Reuse**: Leverages current dataset partitioning (`upload_start_idx`/`upload_end_idx`) and multiprocessing architecture

---

## **Phase 1: Foundation - Add `skip_configure` Support**

### **1.1 Extend BaseClient.run_experiment()**
**File:** `engine/base_client/client.py`

**Changes:**
- Add `skip_configure: Optional[bool] = False` parameter
- Modify configuration logic to conditionally skip `self.configurator.configure(dataset)`

**Implementation:**
```python
def run_experiment(
    self,
    dataset: Dataset,
    skip_upload: bool = False,
    skip_search: bool = False,
    skip_if_exists: bool = True,
    skip_configure: Optional[bool] = False,  # NEW
    parallels: [int] = [],
    upload_start_idx: int = 0,
    upload_end_idx: int = -1,
    num_queries: int = -1,
    ef_runtime: List[int] = [],
):
    # ... existing code ...
    
    if not skip_upload:
        if not skip_configure:  # NEW LOGIC
            print("Experiment stage: Configure")
            self.configurator.configure(dataset)
        
        print(f"Experiment stage: Upload. Vector range [{upload_start_idx}:{upload_end_idx}]")
        # ... rest of upload logic
```

### **1.2 Extend CLI Interface**
**File:** `run.py`

**Changes:**
- Add `skip_configure: Optional[bool] = False` parameter
- Pass parameter through to BaseClient

**CLI Usage:**
```bash
python run.py --engines redis --datasets dbpedia-openai-1M-1536-angular --skip-configure
```

### **1.3 Result Directory Structure**
**Create directories:**
```
results/
├── mixed/           # NEW - Mixed workload results
│   ├── search/      # Search performance during mixed workload
│   ├── update/      # Update operation statistics  
│   └── system/      # System metrics and combined analysis
└── (existing dirs)
```

---

## **Phase 2: Mixed Workload Configuration Framework**

### **2.1 Mixed Workload Configuration**
**New File:** `benchmark/mixed_workload.py`

```python
@dataclass
class MixedWorkloadConfig:
    # Dataset partitioning (using existing infrastructure)
    update_pool_size: int = 10000           # Vectors for update operations
    
    # Update operation rates and patterns  
    update_ops_per_second: float = 100.0    # Target update rate
    insert_ratio: float = 0.4               # 40% inserts
    update_ratio: float = 0.4               # 40% updates  
    delete_ratio: float = 0.2               # 20% deletes
    
    # Timeline configuration
    warmup_duration: int = 30               # Baseline measurement (seconds)
    mixed_duration: int = 300               # Mixed workload duration
    cooldown_duration: int = 30             # Recovery measurement
    
    # Performance monitoring
    stats_collection_interval: float = 1.0  # Statistics sampling rate
    
    def __post_init__(self):
        assert abs(self.insert_ratio + self.update_ratio + self.delete_ratio - 1.0) < 0.001
        
    def get_search_range(self, total_vectors: int) -> Tuple[int, int]:
        """Returns (start_idx, end_idx) for search pool using existing infrastructure"""
        return (0, total_vectors - self.update_pool_size)
        
    def get_update_range(self, total_vectors: int) -> Tuple[int, int]:
        """Returns (start_idx, end_idx) for update pool using existing infrastructure"""
        return (total_vectors - self.update_pool_size, total_vectors)
```

### **2.2 Update Operation Framework**
**New File:** `benchmark/update_operations.py`

```python
@dataclass
class UpdateOperation:
    operation_type: str  # 'insert', 'update', 'delete'
    vector_id: int
    vector: Optional[List[float]] = None
    metadata: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)

class UpdatePatternGenerator:
    """Generates update operations based on configuration and existing data"""
    
    def __init__(self, config: MixedWorkloadConfig, update_records: List[Record]):
        self.config = config
        self.update_records = update_records
        self.next_insert_id = len(update_records) + 1000000  # Avoid ID conflicts
        self.deleted_ids = set()
        
    def generate_operation(self) -> UpdateOperation:
        """Generate next update operation based on configured ratios"""
        rand = random.random()
        
        if rand < self.config.insert_ratio:
            return self._generate_insert()
        elif rand < self.config.insert_ratio + self.config.update_ratio:
            return self._generate_update() 
        else:
            return self._generate_delete()
```

---

## **Phase 3: Update Infrastructure**

### **3.1 Base Update Framework**
**New File:** `engine/base_client/update.py`

```python
class BaseUpdater:
    """Abstract base class for database-specific update operations"""
    
    def __init__(self, connection_params: dict):
        self.connection_params = connection_params
        
    def insert_vector(self, vector_id: int, vector: List[float], metadata: dict = None) -> bool:
        raise NotImplementedError()
        
    def update_vector(self, vector_id: int, vector: List[float], metadata: dict = None) -> bool:
        raise NotImplementedError()
        
    def delete_vector(self, vector_id: int) -> bool:
        raise NotImplementedError()
        
    def batch_operations(self, operations: List[UpdateOperation]) -> BatchUpdateResult:
        """Execute batch of update operations efficiently"""
        raise NotImplementedError()

@dataclass        
class UpdateStats:
    """Statistics for update operations"""
    insert_count: int = 0
    update_count: int = 0  
    delete_count: int = 0
    error_count: int = 0
    total_operations: int = 0
    total_time: float = 0.0
    operation_times: List[float] = field(default_factory=list)
    
    def add_operation(self, operation_type: str, duration: float, success: bool):
        if success:
            if operation_type == 'insert':
                self.insert_count += 1
            elif operation_type == 'update':
                self.update_count += 1
            elif operation_type == 'delete':
                self.delete_count += 1
        else:
            self.error_count += 1
            
        self.total_operations += 1
        self.total_time += duration
        self.operation_times.append(duration)
```

### **3.2 Redis Update Implementation**
**New File:** `engine/clients/redis/update.py`

```python
class RedisUpdater(BaseUpdater):
    """Redis-specific update operations for mixed workload"""
    
    def __init__(self, host: str, connection_params: dict):
        super().__init__(connection_params)
        self.client = redis.Redis(host=host, **connection_params)
        
    def insert_vector(self, vector_id: int, vector: List[float], metadata: dict = None) -> bool:
        try:
            # Convert vector to bytes (matching existing upload format)
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            
            # Use same key format as existing uploader
            key = f"doc:{vector_id}"
            
            # HSET with vector and metadata
            data = {"vector": vector_bytes}
            if metadata:
                data.update(metadata)
                
            result = self.client.hset(key, mapping=data)
            return result >= 1
            
        except Exception as e:
            print(f"Insert error for vector {vector_id}: {e}")
            return False
            
    def update_vector(self, vector_id: int, vector: List[float], metadata: dict = None) -> bool:
        # Redis HSET overwrites existing keys, so same as insert
        return self.insert_vector(vector_id, vector, metadata)
        
    def delete_vector(self, vector_id: int) -> bool:
        try:
            key = f"doc:{vector_id}"
            result = self.client.delete(key)
            return result >= 1
        except Exception as e:
            print(f"Delete error for vector {vector_id}: {e}")
            return False
```

---

## **Phase 4: Mixed Workload Process Architecture**

### **4.1 Update Worker Process**
**New File:** `engine/base_client/update_worker.py`

```python
class UpdateWorker:
    """Worker process for executing update operations"""
    
    def __init__(self, 
                 updater: BaseUpdater,
                 config: MixedWorkloadConfig, 
                 update_records: List[Record],
                 result_queue: Queue,
                 control_event: Event):
        self.updater = updater
        self.config = config
        self.pattern_generator = UpdatePatternGenerator(config, update_records)
        self.result_queue = result_queue
        self.control_event = control_event
        self.stats = UpdateStats()
        
    def run_update_workload(self):
        """Main worker loop with rate limiting"""
        interval = 1.0 / self.config.update_ops_per_second
        
        while not self.control_event.is_set():
            start_time = time.time()
            
            # Generate and execute operation
            operation = self.pattern_generator.generate_operation()
            success = self._execute_operation(operation)
            
            duration = time.time() - start_time
            self.stats.add_operation(operation.operation_type, duration, success)
            
            # Send periodic updates
            if self.stats.total_operations % 100 == 0:
                self.result_queue.put(('stats_update', copy.deepcopy(self.stats)))
                
            # Rate limiting
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
```

### **4.2 Mixed Workload Orchestrator**
**New File:** `engine/base_client/mixed_orchestrator.py`

```python
class MixedWorkloadOrchestrator:
    """Coordinates mixed workload execution using existing infrastructure"""
    
    def __init__(self, 
                 base_client: BaseClient,
                 updater: BaseUpdater, 
                 config: MixedWorkloadConfig):
        self.base_client = base_client
        self.updater = updater  
        self.config = config
        
    def execute_mixed_workload(self, dataset: Dataset) -> MixedWorkloadResults:
        """Execute complete mixed workload using existing infrastructure"""
        
        # Phase 1: Initial dataset upload (existing infrastructure)
        print("Phase 1: Initial dataset setup")
        self.base_client.run_experiment(dataset, skip_search=True)
        
        # Phase 2: Partition dataset using existing read_data ranges
        print("Phase 2: Dataset partitioning") 
        total_vectors = dataset.config.vector_count or self._estimate_vector_count(dataset)
        search_start, search_end = self.config.get_search_range(total_vectors)
        update_start, update_end = self.config.get_update_range(total_vectors)
        
        reader = dataset.get_reader(normalize=False)
        update_records = list(reader.read_data(update_start, update_end))
        
        # Phase 3: Execute mixed workload timeline
        return self._execute_timeline(dataset, update_records)
        
    def _execute_timeline(self, dataset: Dataset, update_records: List[Record]) -> MixedWorkloadResults:
        """Execute warmup -> mixed -> cooldown timeline"""
        
        # Timeline coordination using existing search infrastructure
        timeline_results = {}
        
        # Warmup phase: Search only (baseline)
        print(f"Warmup phase: {self.config.warmup_duration}s")
        timeline_results['warmup'] = self._run_search_phase(dataset, self.config.warmup_duration)
        
        # Mixed phase: Search + Updates concurrently
        print(f"Mixed phase: {self.config.mixed_duration}s") 
        timeline_results['mixed'] = self._run_mixed_phase(dataset, update_records, self.config.mixed_duration)
        
        # Cooldown phase: Search only (recovery)
        print(f"Cooldown phase: {self.config.cooldown_duration}s")
        timeline_results['cooldown'] = self._run_search_phase(dataset, self.config.cooldown_duration)
        
        return MixedWorkloadResults(timeline_results, self.config)
```

---

## **Phase 5: Integration with Existing Search Architecture**

### **5.1 Preserve Existing Search Process Model**
**Strategy:** Reuse existing `BaseClient.run_experiment()` search execution without modification

```python
def _run_search_phase(self, dataset: Dataset, duration: int) -> SearchPhaseResults:
    """Run search workload using existing infrastructure"""
    
    # Use existing search infrastructure with time limiting
    original_timeout = 86400.0  # Default timeout
    
    # Temporarily modify dataset queries to limit execution time
    with TimeboxedExecution(duration):
        # This uses existing searcher.search() and worker processes
        self.base_client.run_experiment(
            dataset=dataset,
            skip_upload=True,           # Skip upload - already done
            skip_configure=True,        # Skip configure - already done  
            skip_search=False,          # Execute search
        )
        
    return SearchPhaseResults.from_results_dir()

def _run_mixed_phase(self, dataset: Dataset, update_records: List[Record], duration: int) -> MixedPhaseResults:
    """Run concurrent search + update workload"""
    
    # Start update worker processes
    update_processes = self._start_update_workers(update_records)
    
    # Start search workload (existing infrastructure)
    search_future = self._start_search_workload_async(dataset, duration)
    
    # Coordinate both workloads
    time.sleep(duration)
    
    # Stop update workers
    self._stop_update_workers(update_processes)
    
    # Collect results from both workloads
    search_results = search_future.result()
    update_results = self._collect_update_results(update_processes)
    
    return MixedPhaseResults(search_results, update_results)
```

### **5.2 Leverage Existing Multiprocessing Model**
**Approach:** Add update workers alongside existing search workers without changing proven architecture

```python
def _start_update_workers(self, update_records: List[Record]) -> List[Process]:
    """Start update worker processes using existing multiprocessing patterns"""
    
    update_processes = []
    control_event = Event()
    result_queue = Queue()
    
    # Start update workers (similar to existing search workers)
    for worker_id in range(self.config.update_worker_count):
        worker = UpdateWorker(
            updater=self.updater,
            config=self.config,
            update_records=update_records,
            result_queue=result_queue,
            control_event=control_event
        )
        
        process = Process(target=worker.run_update_workload)
        process.start()
        update_processes.append(process)
        
    return update_processes
```

---

## **Phase 6: Results and Analysis Framework**

### **6.1 Mixed Workload Metrics**
**New File:** `benchmark/mixed_metrics.py`

```python
@dataclass
class MixedWorkloadResults:
    """Comprehensive results from mixed workload execution"""
    
    # Search performance analysis
    warmup_search_qps: float
    mixed_search_qps: float  
    cooldown_search_qps: float
    search_degradation_percent: float
    
    # Update performance analysis  
    update_ops_per_second: float
    update_success_rate: float
    update_latency_p50: float
    update_latency_p95: float
    
    # System impact analysis
    memory_delta_mb: float
    cpu_utilization_delta: float
    index_size_change: int
    
    # Cross-workload correlation
    update_search_correlation: float  # How update rate affects search latency
    recovery_time_seconds: float      # Time to return to baseline after mixed phase
    
    def save_results(self, output_dir: Path):
        """Save results to mixed workload directory structure"""
        mixed_dir = output_dir / "mixed"
        mixed_dir.mkdir(exist_ok=True)
        
        # Save detailed analysis
        with open(mixed_dir / "mixed_workload_analysis.json", "w") as f:
            json.dump(asdict(self), f, indent=2)
```

### **6.2 Performance Analysis Tools**
```python
class MixedWorkloadAnalyzer:
    """Analyze mixed workload performance and generate insights"""
    
    def analyze_performance_impact(self, results: MixedWorkloadResults) -> PerformanceImpactReport:
        """Detailed analysis of search performance under update load"""
        
        degradation = results.search_degradation_percent
        
        if degradation < 5:
            impact_level = "Minimal"
        elif degradation < 15:
            impact_level = "Moderate"  
        else:
            impact_level = "Significant"
            
        return PerformanceImpactReport(
            impact_level=impact_level,
            degradation_percent=degradation,
            recommendations=self._generate_recommendations(results)
        )
        
    def generate_comparison_report(self, baseline_results, mixed_results) -> ComparisonReport:
        """Compare mixed workload vs baseline performance"""
        # Implementation for detailed comparison analysis
```

---

## **Phase 7: CLI Integration and Automation**

### **7.1 Extended CLI Interface**
**Modify:** `run.py`

```python
def run(
    # ... existing parameters ...
    
    # Mixed workload parameters
    mixed_workload: bool = False,
    update_pool_size: int = 10000,
    update_rate: float = 100.0, 
    insert_ratio: float = 0.4,
    update_ratio: float = 0.4,
    delete_ratio: float = 0.2,
    mixed_duration: int = 300,
    warmup_duration: int = 30,
    cooldown_duration: int = 30,
):
    if mixed_workload:
        # Execute mixed workload
        config = MixedWorkloadConfig(
            update_pool_size=update_pool_size,
            update_ops_per_second=update_rate,
            insert_ratio=insert_ratio,
            update_ratio=update_ratio, 
            delete_ratio=delete_ratio,
            mixed_duration=mixed_duration,
            warmup_duration=warmup_duration,
            cooldown_duration=cooldown_duration,
        )
        
        orchestrator = MixedWorkloadOrchestrator(client, updater, config)
        results = orchestrator.execute_mixed_workload(dataset)
        
        analyzer = MixedWorkloadAnalyzer()
        report = analyzer.analyze_performance_impact(results)
        print(f"Mixed workload completed. Performance impact: {report.impact_level}")
    else:
        # Standard single workload (existing)
        client.run_experiment(dataset, skip_upload, skip_search, skip_if_exists)
```

### **7.2 Script Integration**
**New File:** `tools/run_mixed_benchmark.sh`

```bash
#!/bin/bash
# Mixed workload benchmark automation

DATASET=${1:-"dbpedia-openai-1M-1536-angular"}
ENGINE=${2:-"redis"}
UPDATE_RATE=${3:-100}

echo "Running mixed workload benchmark: $ENGINE on $DATASET at $UPDATE_RATE ops/sec"

python run.py \
    --engines $ENGINE \
    --datasets $DATASET \
    --mixed-workload \
    --update-rate $UPDATE_RATE \
    --mixed-duration 300 \
    --warmup-duration 30 \
    --cooldown-duration 30
```

---

## **Phase 8: Testing and Validation Strategy**

### **8.1 Unit Testing**
```python
# tests/test_mixed_workload.py
def test_mixed_workload_config():
    """Test mixed workload configuration validation"""
    
def test_update_pattern_generator():
    """Test update operation generation patterns"""
    
def test_dataset_partitioning():
    """Test dataset partitioning using existing infrastructure"""
    
def test_update_operations():
    """Test Redis update operations"""
```

### **8.2 Integration Testing**  
```python
def test_mixed_workload_execution():
    """End-to-end mixed workload test with small dataset"""
    
def test_timeline_coordination():
    """Test warmup -> mixed -> cooldown timeline"""
    
def test_process_coordination():
    """Test update worker + search worker coordination"""
```

### **8.3 Performance Validation**
- Mixed workload vs baseline comparison
- Update rate scaling analysis  
- Memory usage profiling
- Long-running stability tests

---

## **Implementation Timeline**

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| **Phase 1** | 2 days | None | `skip_configure` support |
| **Phase 2** | 2 days | Phase 1 | Mixed workload config framework |
| **Phase 3** | 3 days | Phase 2 | Update infrastructure + Redis implementation |
| **Phase 4** | 3 days | Phase 3 | Update workers + orchestration |
| **Phase 5** | 2 days | Phase 4 | Search integration |
| **Phase 6** | 2 days | Phase 5 | Results + analysis |
| **Phase 7** | 1 day | Phase 6 | CLI + automation |
| **Phase 8** | 2 days | All phases | Testing + validation |

**Total: ~17 days**

---

## **Key Technical Decisions**

### **Dataset Partitioning Strategy**
- **Decision**: Leverage existing `upload_start_idx`/`upload_end_idx` infrastructure instead of creating new `DatasetPartitioner`
- **Rationale**: Existing system already handles complex multi-file datasets, ranges, and chunking efficiently
- **Implementation**: Use `reader.read_data(start_idx, end_idx)` to partition search and update pools

### **Search Pool vs Update Pool Size**
- **Decision**: 90% search pool, 10% update pool
- **Rationale**: Maintains realistic search performance while creating sufficient update stress
- **Benefits**: Large search space for realistic latency characteristics, small high-churn update pool for system stress

### **Process Architecture**
- **Decision**: Preserve existing multiprocessing search architecture, add parallel update workers
- **Rationale**: Existing search process model is proven and optimized
- **Implementation**: Update workers run alongside search workers without modifying search processes

### **Timeline Execution**
- **Decision**: Warmup → Mixed → Cooldown phases
- **Rationale**: Clear baseline measurement, performance impact analysis, and recovery assessment
- **Benefits**: Enables precise measurement of update impact on search performance

---

## **Expected Outcomes**

### **Performance Insights**
1. **Search Degradation Analysis**: Quantify how concurrent updates affect search QPS and latency
2. **Update Throughput Measurement**: Determine sustainable update rates under search load
3. **Resource Utilization**: Memory, CPU, and I/O impact of mixed workloads
4. **Recovery Characteristics**: How quickly system returns to baseline after update load

### **Production Applicability**
1. **Realistic Load Testing**: Simulate production scenarios with concurrent read/write operations
2. **Capacity Planning**: Determine optimal update rates for given search performance requirements
3. **System Tuning**: Identify bottlenecks and optimization opportunities
4. **SLA Validation**: Verify search performance SLAs under realistic mixed workloads

### **Benchmarking Value**
1. **Cross-Database Comparison**: Compare mixed workload performance across Redis, Qdrant, etc.
2. **Configuration Optimization**: Find optimal parameters for mixed workload scenarios
3. **Scalability Analysis**: Understand how mixed workload performance scales with data size
4. **Use Case Validation**: Validate database suitability for specific mixed workload patterns

---

## **Risk Mitigation**

### **Performance Risks**
- **Risk**: Update operations significantly degrade search performance
- **Mitigation**: Start with low update rates, gradually increase, implement circuit breakers

### **Stability Risks**
- **Risk**: Concurrent operations cause deadlocks or race conditions
- **Mitigation**: Comprehensive testing, Redis-specific locking analysis, monitoring

### **Complexity Risks**
- **Risk**: Mixed workload implementation becomes overly complex
- **Mitigation**: Leverage existing infrastructure, modular design, thorough documentation

### **Resource Risks**
- **Risk**: Mixed workload consumes excessive system resources
- **Mitigation**: Resource monitoring, configurable worker counts, graceful degradation

---

## **Future Enhancements**

### **Advanced Update Patterns**
- Geographic distribution of updates
- Time-based update patterns (burst vs steady)
- Vector similarity-based update targeting

### **Extended Database Support**
- Qdrant mixed workload implementation
- Milvus mixed workload implementation
- Elasticsearch mixed workload implementation

### **Advanced Analytics**
- Real-time performance dashboards
- Machine learning-based anomaly detection
- Automated performance optimization recommendations

### **Production Integration**
- Kubernetes deployment support
- Cloud provider integration
- CI/CD pipeline integration

---

## **Conclusion**

This implementation plan provides a comprehensive roadmap for adding sophisticated mixed workload capabilities to vector-db-benchmark while preserving the existing proven architecture. The phased approach ensures incremental progress with clear milestones and deliverables.

The design leverages existing infrastructure for dataset partitioning and multiprocessing, ensuring compatibility and performance. The focus on realistic production scenarios (large search pool + small update pool) provides valuable insights for capacity planning and system optimization.

**Next Steps**: Begin Phase 1 implementation with `skip_configure` parameter support as the foundation for mixed workload functionality.

---

**Document Version**: 1.0  
**Last Updated**: August 7, 2025  
**Authors**: GitHub Copilot  
**Review Status**: Ready for Implementation  
