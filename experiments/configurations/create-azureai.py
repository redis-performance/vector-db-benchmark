import json

experiments = []

for m in [4, 8, 10]:
    for efConstruction in [128, 256, 512]:
        for efSearch in [128, 256, 512]:
            name = f"azureai-m-{m}-efConstruction-{efConstruction}-efSearch-{efSearch}"

            config = {
                "name": name,
                "engine": "azureai",
                "connection_params": {},
                "collection_params": {
                    "hnsw_config": {
                        "m": m,
                        "efConstruction": efConstruction,
                        "efSearch": efSearch,
                    }
                },
                "search_params": [
                    {"parallel": 1, "config": {}},
                    {"parallel": 100, "config": {}},
                ],
                "upload_params": {"parallel": 16, "batch_size": 1024},
            }

            experiments.append(config)

with open("azure-ai-search.json", "w") as fd:
    json.dump(experiments,fd)
