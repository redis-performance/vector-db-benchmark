import json

ms = [4, 8, 16, 32]
ef_constructs = [4, 8, 16, 32]
ef_runtimes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
data_type = "FLOAT16"
for algo in ["hnsw"]:
    configs = []
    for m in ms:
        for ef_construct in ef_constructs:
            config = {
                "name": f"redis-intel-{data_type.lower()}-{algo}-m-{m}-ef-{ef_construct}",
                "engine": "redis",
                "connection_params": {},
                "collection_params": {
                    "algorithm": algo,
                    "data_type": data_type,
                    f"{algo}_config": {"M": m, "EF_CONSTRUCTION": ef_construct},
                },
                "search_params": [],
                "upload_params": {
                    "parallel": 128,
                    "algorithm": algo,
                    "data_type": data_type,
                },
            }
            for client in [1, 50, 100, 200]:
                for ef_runtime in ef_runtimes:
                    test_config = {
                        "parallel": client,
                        "search_params": {"ef": ef_runtime, "data_type": data_type},
                    }
                    config["search_params"].append(test_config)
            configs.append(config)
    fname = f"redis-intel-{algo}-single-node.json"
    with open(fname, "w") as json_fd:
        json.dump(configs, json_fd, indent=2)
        print(f"created {len(configs)} configs for {fname}.")
