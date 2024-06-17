import json

experiments = []

for m in [4, 10]:
    for efConstruction in [100, 500, 1000]:
        config = (
            {
                "name": "redis-m-{m}-ef-{efConstruction}",
                "engine": "redis",
                "connection_params": {},
                "collection_params": {
                    "hnsw_config": {"M": m, "EF_CONSTRUCTION": efConstruction}
                },
                "search_params": [],
                "upload_params": {"parallel": 16},
            },
        )

        for efSearch in [100, 500, 1000]:
            config["search_params"].append({"parallel": 1, "config": {"EF": efSearch}})
            config["search_params"].append({"parallel": 50, "config": {"EF": efSearch}})

        experiments.append(config)

with open("redis-vs-azure-ai-search.json", "w") as fd:
    json.dump(experiments, fd)
