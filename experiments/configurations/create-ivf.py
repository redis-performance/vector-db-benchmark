import json

n_lists = [256, 512, 1024, 1536, 2048]
n_probes = [16, 20, 32, 64, 128, 256]


for algo in ["raft_ivf_pq", "raft_ivf_flat"]:
    configs = []
    for lists in n_lists:
        for probes in n_probes:
            config = {
                "name": f"redis-{algo}-n_lists-{lists}-n_probes-{probes}",
                "engine": "redis",
                "connection_params": {},
                "collection_params": {
                    "algorithm": algo,
                    f"{algo}_config": {"N_LISTS": lists, "N_PROBES": probes},
                },
                "search_params": [
                    {
                        "parallel": 1,
                        "search_params": {
                            "algorithm": algo,
                        },
                    },
                    {
                        "parallel": 100,
                        "search_params": {
                            "algorithm": algo,
                        },
                    },
                ],
                "upload_params": {"parallel": 16,  "algorithm": algo},
            }
            configs.append(config)
    fname = f"redis-{algo}-single-node.json"
    with open(fname, "w") as json_fd:
        json.dump(configs, json_fd, indent=2)
        print(f"created {len(configs)} configs for {fname}.")
