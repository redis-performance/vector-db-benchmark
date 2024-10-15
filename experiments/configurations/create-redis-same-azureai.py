import json

experiments = []

for m in [4, 10]:
    for efConstruction in [100, 500, 1000]:
        search_params = []
        config = {
            "name": f"redis-m-{m}-ef-{efConstruction}",
            "engine": "redis",
            "connection_params": {},
            "collection_params": {
                "hnsw_config": {"M": m, "EF_CONSTRUCTION": efConstruction}
            },
            "search_params": [],
            "upload_params": {"parallel": 16},
        }

        for efSearch in [100, 500, 1000]:
            single_client_config = {"parallel": 1, "search_params": {"ef": efSearch}}
            multi_client_config = {"parallel": 50, "search_params": {"ef": efSearch}}
            search_params.append(single_client_config)
            search_params.append(multi_client_config)
        config["search_params"] = search_params

        experiments.append(config)

with open("redis-vs-azure-ai-search.json", "w") as fd:
    json.dump(experiments, fd)
