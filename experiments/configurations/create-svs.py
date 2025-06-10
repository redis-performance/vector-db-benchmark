import json

threads = [16]
ws_constructs = [100]
ws_search = [32, 40, 48, 64]
#ws_search = [48]
graph_degree = [32]
#quantization = ["0", "4x4", "4x8", "8", "4"]
quantization = ["8"]
topKs = [10]
data_types = ["FLOAT32"]

for algo in ["svs_tiered"]:
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
                                f"{algo}_config": {"NUM_THREADS": thread, "GRAPH_DEGREE": graph_d, "WS_CONSTRUCTION": ws_construct, "QUANTIZATION": quant},
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
                                        "search_params": {"WS_SEARCH": ws_s, "data_type": data_type},
                                    }
                                    config["search_params"].append(test_config)
                        configs.append(config)

                    fname = f"svs-test-algo-{algo}-graph-{graph_d}-ws-con-{ws_construct}-quant-{quant}-threads-{thread}-dt-{data_type}.json"
                    with open(fname, "w") as json_fd:
                        json.dump(configs, json_fd, indent=2)
                        print(f"Created {len(configs)} configs for {fname}.")