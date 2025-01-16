import json

experiments = []
batch_size = 64

for data_type in ["INT8","FLOAT16", "BFLOAT16", "FLOAT32", "FLOAT64"]:
    # flat
    search_params = []
    config = {
        "name": f"redis-{data_type.lower()}-flat",
        "engine": "redis",
        "connection_params": {},
        "collection_params": {
            "data_type": data_type,
            "flat_config": {},
        },
        "search_params": [],
        "upload_params": {"parallel": 16, "data_type": data_type, "batch_size": batch_size},
    }
    single_client_config = {
        "parallel": 1,
        "search_params": {"data_type": data_type},
    }
    multi_client_config = {
        "parallel": 100,
        "search_params": {"data_type": data_type},
    }
    search_params.append(single_client_config)
    search_params.append(multi_client_config)
    config["search_params"] = search_params
    experiments.append(config)

    for m in [8, 16, 32, 64, 128, 256]:
        # for efConstruction in [32, 64]:
        for efConstruction in [16, 32, 64, 128, 256, 512, 1024, 2048]:
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
                "upload_params": {"parallel": 16, "data_type": data_type, "batch_size": batch_size},
            }
            #            for efSearch in [16, 32, 1024]:
            for efSearch in [8, 16, 32, 64, 128, 256, 512, 1024, 1024, 2048, 4096]:
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
