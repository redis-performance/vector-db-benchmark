# Dataset Validation

This document describes the dataset validation system for the vector-db-benchmark project.

## Overview

The validation system ensures that:
- All datasets in `datasets/datasets.json` have the required fields
- Field types and values are correct and reasonable
- Dataset names are unique
- The `--describe` functionality works correctly

## Validation Components

### 1. Local Validation Script

**File**: `validate_datasets.py`

Run locally to validate datasets:

```bash
# Basic validation
python validate_datasets.py

# Strict mode (treat warnings as errors)
python validate_datasets.py --strict
```

**What it checks:**
- JSON structure and syntax
- Required fields: `name`, `vector_size`, `distance`, `type`, `path`, `vector_count`, `description`
- Field types (handles special cases like `h5-multi` datasets with dict paths)
- Data consistency (positive vector sizes/counts, valid distance metrics)
- Unique dataset names
- `--describe datasets` and `--describe engines` functionality

### 2. GitHub Action

**File**: `.github/workflows/validate-datasets.yml`

Automatically runs on:
- Push to files: `datasets/datasets.json`, `run.py`, `benchmark/dataset.py`
- Pull requests affecting the same files

The action simply runs the validation script:
```yaml
- name: Validate datasets.json
  run: python validate_datasets.py
```

## Required Dataset Fields

Each dataset in `datasets/datasets.json` must have:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | string | Unique dataset identifier | `"glove-100-angular"` |
| `vector_size` | integer | Vector dimensions | `100` |
| `distance` | string | Distance metric | `"cosine"`, `"l2"`, `"dot"`, `"euclidean"` |
| `type` | string | Dataset format | `"h5"`, `"tar"`, `"jsonl"`, `"h5-multi"` |
| `path` | string/dict | File path or multi-file structure | `"glove-100/file.hdf5"` |
| `vector_count` | integer/null | Number of vectors | `1183514` or `null` |
| `description` | string/null | Human-readable description | `"Word vectors"` or `null` |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `link` | string | Download URL |
| `schema` | dict | Additional metadata fields |

## Special Cases

### Multi-file Datasets (h5-multi)

For large datasets split across multiple files, the `path` field can be a dictionary:

```json
{
  "name": "laion-img-emb-768d-1Billion-cosine",
  "type": "h5-multi",
  "path": {
    "data": [
      {
        "file_number": "1",
        "path": "laion-1b/part1.hdf5",
        "link": "http://example.com/part1.hdf5",
        "start_idx": 0,
        "end_idx": 10000000
      }
    ],
    "queries": [
      {
        "path": "laion-1b/queries.hdf5",
        "link": "http://example.com/queries.hdf5"
      }
    ]
  }
}
```

## Validation Warnings

The validator may show warnings for:
- **Round vector counts**: Numbers like 1,000,000 that look like estimates
- **Missing descriptions**: Datasets with `null` descriptions
- **Missing download links**: Non-local datasets without download URLs
- **Large vector sizes**: Dimensions > 4096 (flagged for verification)

Warnings don't fail validation but should be reviewed.

## Adding New Datasets

1. Add your dataset to `datasets/datasets.json`
2. Run `python validate_datasets.py` locally
3. Fix any errors or warnings
4. Commit and push (GitHub Action will validate automatically)

## Testing Describe Functionality

The validation includes testing the `--describe` commands:

```bash
python run.py --describe datasets
python run.py --describe engines
```

This ensures the new dataset display functionality works correctly.
