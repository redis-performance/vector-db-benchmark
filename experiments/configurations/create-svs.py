import json

threads = [16]
ws_constructs = [200]
ws_search = [177]
#ws_search = [48]
graph_degree = [32]
#quantization = ["LVQ4X4", "LVQ4x8", "LVQ8", "LVQ4"]
quantization = ["LVQ4X4"]
topKs = [100]
data_types = ["FLOAT16", "FLOAT32"]

for algo in ["svs-vamana"]:
    for data_type in data_types:
        for ws_construct in ws_constructs:
            for graph_d in graph_degree:
                for quant in quantization:
                    configs = []
                    for thread in threads:
                        config = {
                            "name": f"svs-{algo}-quant-{quant}-dt-{data_type}",
                            "engine": "redis",
                            "connection_params": {},
                            "collection_params": {
                                "algorithm": algo,
                                "data_type": data_type,
                                f"{algo}_config": {"NUM_THREADS": thread, "GRAPH_MAX_DEGREE": graph_d, "CONSTRUCTION_WINDOW_SIZE": ws_construct, "compression": quant},
                            },
                            "search_params": [],
                            "upload_params": {
                                "parallel": 100,
                                "data_type": data_type,
                                "algorithm": algo,
                            },
                        }
                        for client in [100]:
                            for ws_s in ws_search:
                                for top in topKs:
                                    test_config = {
                                        "algorithm": algo,
                                        "parallel": client,
                                        "top": top,
                                        "search_params": {"WS_SEARCH": ws_s, "data_type": data_type},
                                    }
                                    config["search_params"].append(test_config)
                        configs.append(config)

                    fname = f"svs-{algo}-quant-{quant}-dt-{data_type}.json"
                    with open(fname, "w") as json_fd:
                        json.dump(configs, json_fd, indent=2)
                        print(f"Created {len(configs)} configs for {fname}.")
