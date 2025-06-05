import json

ms = [16]
ef_constructs = [100]
ef_runtimes = [20, 40, 80]
# qants = ["NOQUANT", "Q8", "BIN"]
qants = ["NOQUANT"]
configs = []
for m in ms:
    for ef_construct in ef_constructs:
        for quant in qants:
            config = {
                "name": f"redis-intel-vectorsets-m-{m}-ef-{ef_construct}-quant-{quant}",
                "engine": "vectorsets",
                "connection_params": {},
                "collection_params": {},
                "search_params": [],
                "upload_params": {
                    "parallel": 128,
                    "hnsw_config": {
                        "M": m,
                        "EF_CONSTRUCTION": ef_construct,
                        "quant": quant,
                    },
                },
            }
            for client in [1, 8]:
                for ef_runtime in ef_runtimes:
                    test_config = {
                        "parallel": client,
                        "search_params": {"ef": ef_runtime},
                    }
                    config["search_params"].append(test_config)
            configs.append(config)
    fname = f"redis-intel-vectorsets.json"
    with open(fname, "w") as json_fd:
        json.dump(configs, json_fd, indent=2)
        print(f"created {len(configs)} configs for {fname}.")
