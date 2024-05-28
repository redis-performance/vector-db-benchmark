import json

ms = [16, 64, 256]
ef_constructs = [16, 512, 1024]
ef_runtimes = [16, 32, 64, 128, 256, 512, 1024, 2048]

for algo in ["hnsw"]:
    configs = []
    for m in ms:
        for ef_construct in ef_constructs:
            config = {
                "name": f"redis-intel-{algo}-m-{m}-ef-{ef_construct}",
                "engine": "redis",
                "connection_params": {},
                "collection_params": {
                    "algorithm": algo,
                    f"{algo}_config": {"M": m, "EF_CONSTRUCTION": ef_construct},
                },
                "search_params": [],
                "upload_params": {"parallel": 128, "algorithm": algo},
            }
            for client in [100, 200, 400, 800, 1600, 3200]:
                for ef_runtime in ef_runtimes:
                    test_config = {
                        "parallel": client,
                        "search_params": {"ef": ef_runtime},
                    }
                    config["search_params"].append(test_config)
            configs.append(config)
    fname = f"redis-intel-{algo}-single-node.json"
    with open(fname, "w") as json_fd:
        json.dump(configs, json_fd, indent=2)
        print(f"created {len(configs)} configs for {fname}.")
