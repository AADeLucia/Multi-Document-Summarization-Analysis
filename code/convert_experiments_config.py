import json

INPUT_FILE = "experiments.json"
OUTPUT_FILE = "experiment_array.jsonl"

with open(INPUT_FILE) as f:
    data = json.load(f)

models = data['models']
datasets = data['datasets']
splits = data['splits']
metrics = data['metrics']

with open(OUTPUT_FILE, 'w') as out_f:
    for model in models:
        for dataset in datasets:
            obj = {
                'MODEL': model,
                'DATASET': dataset,
                'SPLITS': splits,
                'METRICS': metrics
            }
            out_f.write(json.dumps(obj) + '\n')
