import json

threads = [16]
ws_constructs = [100]
ws_search = [32, 40, 48, 64]
#ws_search = [48]
graph_degree = [32]
#quantization = ["NO_COMPRESSION", "LVQ4x4", "LVQ4x8", "LVQ4", "LVQ8", "LeanVec4x8", "LeanVec8x8"]
quantization = ["LVQ4x8"]
topKs = [10]
data_types = ["FLOAT32"]

for algo in ["svs-vamana"]:
    for data_type in data_types:
        for ws_construct in ws_constructs:
            for graph_d in graph_degree:
                for quant in quantization:
                    configs = []
                    for thread in threads:
                        config = {
                            "name": f"svs-test-algo-{algo}-graph-{graph_d}-ws-con-{ws_construct}-quant-{quant}-threads-{thread}-dt-{data_type}",
                            "engine": "redis",
                            "connection_params": {},
                            "collection_params": {
                                "algorithm": algo,
                                "data_type": data_type,
                                f"{algo}_config": {"NUM_THREADS": thread, "GRAPH_MAX_DEGREE": graph_d, "CONSTRUCTION_WINDOW_SIZE": ws_construct, "COMPRESSION": quant},
                            },
                            "search_params": [],
                            "upload_params": {
                                "parallel": 128,
                                "data_type": data_type,
                                "algorithm": algo,
                            },
                        }
                        for client in [1, 8, 16, 32, 64, 128]:
                            for ws_s in ws_search:
                                for top in topKs:
                                    test_config = {
                                        "algorithm": algo,
                                        "parallel": client,
                                        "top": top,
                                        "search_params": {"SEARCH_WINDOW_SIZE": ws_s, "data_type": data_type},
                                    }
                                    config["search_params"].append(test_config)
                        configs.append(config)

                    fname = f"svs-test-algo-{algo}-graph-{graph_d}-ws-con-{ws_construct}-quant-{quant}-threads-{thread}-dt-{data_type}.json"
                    with open(fname, "w") as json_fd:
                        json.dump(configs, json_fd, indent=2)
                        print(f"Created {len(configs)} configs for {fname}.")