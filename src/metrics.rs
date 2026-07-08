//! Retrieval quality metrics: recall@K, precision@K, MRR, NDCG@K.

use std::collections::HashSet;

/// Per-query retrieval quality metrics.
#[derive(Debug, Clone, Default)]
pub struct QueryMetrics {
    /// recall@K: |retrieved ∩ truth_top_K| / K
    pub recall: f64,
    /// precision@K: |retrieved ∩ truth_top_K| / |retrieved|
    pub precision: f64,
    /// Mean Reciprocal Rank: 1/rank of the first relevant result
    pub mrr: f64,
    /// Normalized Discounted Cumulative Gain @ K
    pub ndcg: f64,
}

/// Compute all retrieval quality metrics for a single query.
///
/// - `result_ids_ordered`: engine results in ranked order (position 0 = best)
/// - `ground_truth`: true top-K neighbor IDs from the dataset
/// - `k`: the K value (top)
pub fn compute_metrics(result_ids_ordered: &[i64], ground_truth: &[i64], k: usize) -> QueryMetrics {
    if k == 0 {
        return QueryMetrics {
            recall: 1.0,
            precision: 1.0,
            mrr: 1.0,
            ndcg: 1.0,
        };
    }

    // Ground truth: drop sentinel/invalid ids (e.g. the `-1` padding HDF5
    // `neighbors` rows use when a query has fewer than K true neighbors), then
    // cap at K. The recall denominator is the number of *valid* truth ids, so a
    // query with fewer than K real neighbors can still reach recall 1.0 — this
    // matches the NDCG ideal-DCG convention below.
    let truth_set: HashSet<i64> = ground_truth
        .iter()
        .copied()
        .filter(|&id| id >= 0)
        .take(k)
        .collect();
    let truth_count = truth_set.len();

    // No valid ground truth (e.g. a filtered query with no matching points):
    // nothing to retrieve, so this query is not penalized.
    if truth_count == 0 {
        return QueryMetrics {
            recall: 1.0,
            precision: 1.0,
            mrr: 1.0,
            ndcg: 1.0,
        };
    }

    // Engine results: dedup (preserving rank order) and keep only the top K, so
    // hits can't be double-counted and recall can't exceed 1.0.
    let mut seen = HashSet::new();
    let results_topk: Vec<i64> = result_ids_ordered
        .iter()
        .copied()
        .filter(|id| seen.insert(*id))
        .take(k)
        .collect();

    let hits = results_topk
        .iter()
        .filter(|id| truth_set.contains(id))
        .count();

    let recall = hits as f64 / truth_count as f64;
    let precision = if results_topk.is_empty() {
        0.0
    } else {
        hits as f64 / results_topk.len() as f64
    };

    // MRR: 1/rank of the first relevant result within the top K.
    let mrr = results_topk
        .iter()
        .enumerate()
        .find(|(_, id)| truth_set.contains(id))
        .map(|(rank, _)| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0);

    // NDCG@K over the deduped top-K results.
    let ndcg = {
        let dcg: f64 = results_topk
            .iter()
            .enumerate()
            .filter(|(_, id)| truth_set.contains(id))
            .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        let idcg: f64 = (0..truth_count)
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    };

    QueryMetrics {
        recall,
        precision,
        mrr,
        ndcg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_retrieval() {
        let results = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9);
        assert!((m.precision - 1.0).abs() < 1e-9);
        assert!((m.mrr - 1.0).abs() < 1e-9);
        assert!((m.ndcg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_overlap() {
        let results = vec![6, 7, 8, 9, 10];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall).abs() < 1e-9);
        assert!((m.precision).abs() < 1e-9);
        assert!((m.mrr).abs() < 1e-9);
        assert!((m.ndcg).abs() < 1e-9);
    }

    #[test]
    fn test_first_relevant_at_position_3() {
        let results = vec![10, 20, 3, 1, 5];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 0.6).abs() < 1e-9);
        assert!((m.precision - 0.6).abs() < 1e-9);
        assert!((m.mrr - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_fewer_results_than_k() {
        let results = vec![1, 2, 3];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 0.6).abs() < 1e-9);
        assert!((m.precision - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_k_zero() {
        let m = compute_metrics(&[], &[], 0);
        assert!((m.recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fewer_ground_truth_than_k_reaches_full_recall() {
        // Only 2 real neighbors but k=5: a perfect engine must score recall 1.0
        // (denominator = valid gt, not k). Previously this capped at 2/5 = 0.4.
        let results = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
        assert!((m.ndcg - 1.0).abs() < 1e-9, "ndcg={}", m.ndcg);
    }

    #[test]
    fn test_sentinel_padding_ignored() {
        // HDF5-style -1 padding must not count as truth ids.
        let results = vec![1, 2, 9, 8, 7];
        let truth = vec![1, 2, -1, -1, -1];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
    }

    #[test]
    fn test_duplicate_results_not_double_counted() {
        // Engine returns duplicates; each relevant id counts once.
        let results = vec![1, 1, 2, 2, 3];
        let truth = vec![1, 2, 3, 4, 5];
        let m = compute_metrics(&results, &truth, 5);
        assert!((m.recall - 0.6).abs() < 1e-9, "recall={}", m.recall);
    }

    #[test]
    fn test_excess_results_truncated_to_k() {
        // More than k results: only the top-k count; recall cannot exceed 1.0.
        let results = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let truth = vec![1, 2, 3];
        let m = compute_metrics(&results, &truth, 3);
        assert!((m.recall - 1.0).abs() < 1e-9, "recall={}", m.recall);
        assert!(m.recall <= 1.0 + 1e-9);
    }

    #[test]
    fn test_empty_ground_truth_not_penalized() {
        let m = compute_metrics(&[9, 8, 7], &[], 5);
        assert!((m.recall - 1.0).abs() < 1e-9);
    }
}
