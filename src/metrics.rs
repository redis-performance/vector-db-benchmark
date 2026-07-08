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

    let truth_set: HashSet<i64> = ground_truth.iter().take(k).copied().collect();
    let hits = result_ids_ordered
        .iter()
        .filter(|id| truth_set.contains(id))
        .count();

    let recall = hits as f64 / k as f64;
    let precision = if result_ids_ordered.is_empty() {
        0.0
    } else {
        hits as f64 / result_ids_ordered.len() as f64
    };

    // MRR: 1/rank of first relevant result
    let mrr = result_ids_ordered
        .iter()
        .enumerate()
        .find(|(_, id)| truth_set.contains(id))
        .map(|(rank, _)| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0);

    // NDCG@K
    let ndcg = {
        let dcg: f64 = result_ids_ordered
            .iter()
            .take(k)
            .enumerate()
            .filter(|(_, id)| truth_set.contains(id))
            .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        let ideal_hits = k.min(truth_set.len());
        let idcg: f64 = (0..ideal_hits).map(|i| 1.0 / (i as f64 + 2.0).log2()).sum();

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
}
