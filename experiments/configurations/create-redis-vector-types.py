import json

experiments = []

for data_type in ["FLOAT16", "BFLOAT16", "FLOAT32", "FLOAT64"]:
    for m in [8, 16, 32, 64]:
        # for efConstruction in [32, 64]:
        for efConstruction in [16, 32, 64, 128, 256, 512]:
            search_params = []
            config = {
                "name": f"redis-{data_type.lower()}-m-{m}-ef-{efConstruction}",
                "engine": "redis",
                "connection_params": {},
                "collection_params": {
                    "data_type": data_type,
                    "hnsw_config": {"M": m, "EF_CONSTRUCTION": efConstruction},
                },
                "search_params": [],
                "upload_params": {"parallel": 16, "data_type": data_type},
            }
            #            for efSearch in [16, 32, 1024]:
            for efSearch in [8, 16, 32, 64, 128, 256, 512, 1024]:
                single_client_config = {
                    "parallel": 1,
                    "search_params": {"ef": efSearch, "data_type": data_type},
                }
                multi_client_config = {
                    "parallel": 100,
                    "search_params": {"ef": efSearch, "data_type": data_type},
                }
                search_params.append(single_client_config)
                search_params.append(multi_client_config)
            config["search_params"] = search_params

            experiments.append(config)

with open("redis-vector-types.json", "w") as fd:
    json.dump(experiments, fd)
