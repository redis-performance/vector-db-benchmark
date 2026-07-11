//! Client-side CPU / concurrency-saturation coverage (Linux).
//!
//! The benchmark reports database throughput and latency, but those numbers are
//! only trustworthy while the *client* has spare CPU. Once the client saturates
//! (more worker threads than cores, or CPU-bound on serialization/TLS), QPS
//! plateaus and the measured per-query latency captures run-queue wait rather
//! than the database's service time — yet nothing in the output flagged it.
//!
//! This module samples CPU around the timed search window and derives whether
//! the client was saturated, which is attached to `SearchResults` and the result
//! JSON so a downstream reader knows when a data point is client-bound.
//!
//! ## CLK_TCK-free measurement
//!
//! `/proc/self/stat` reports our process CPU as `utime + stime` in clock ticks;
//! `/proc/stat`'s aggregate `cpu` line reports busy/idle jiffies summed across
//! **all** cores in the same tick unit. Taking the ratio of process ticks to
//! per-core wall jiffies makes the (unknown) USER_HZ cancel, so we need neither
//! `libc`/`sysconf` nor a hardcoded 100:
//!
//! ```text
//! cores_used = Δproc_ticks * num_cores / Δall_core_jiffies
//! system_cpu = (Δall_core_jiffies - Δidle_jiffies) / Δall_core_jiffies
//! ```
//!
//! On non-Linux or if `/proc` is unavailable, sampling returns `None` and the
//! saturation fields degrade to "unknown" (never a false positive).

use std::fs;

/// A CPU snapshot: our process's consumed ticks and the system-wide aggregate.
#[derive(Debug, Clone, Copy)]
pub struct CpuSample {
    /// utime + stime for this process, in clock ticks.
    proc_ticks: u64,
    /// Sum of all fields on the aggregate `cpu` line of /proc/stat, in jiffies
    /// (summed across every core).
    total_jiffies: u64,
    /// idle + iowait jiffies from the aggregate `cpu` line.
    idle_jiffies: u64,
}

/// Take a CPU sample. Returns `None` when `/proc` is unreadable (non-Linux).
pub fn sample() -> Option<CpuSample> {
    let proc_ticks = read_proc_self_ticks()?;
    let (total_jiffies, idle_jiffies) = read_proc_stat_totals()?;
    Some(CpuSample {
        proc_ticks,
        total_jiffies,
        idle_jiffies,
    })
}

/// Parse `utime` (field 14) + `stime` (field 15) from /proc/self/stat.
///
/// Field 2 (`comm`) is wrapped in parentheses and may itself contain spaces or
/// `)`, so we split *after the last* `)` to reach the space-delimited numeric
/// fields, where index 0 is field 3 (`state`).
fn read_proc_self_ticks() -> Option<u64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    parse_proc_self_ticks(&stat)
}

/// Pure parser for a `/proc/self/stat` line (extracted for testability).
fn parse_proc_self_ticks(stat: &str) -> Option<u64> {
    let after = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = after.split_whitespace().collect();
    // After the last ')', fields[0] = state (field 3). utime = field 14, stime =
    // field 15 → indices 11 and 12 here.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// Parse the aggregate `cpu` line of /proc/stat → (total_jiffies, idle_jiffies).
fn read_proc_stat_totals() -> Option<(u64, u64)> {
    let stat = fs::read_to_string("/proc/stat").ok()?;
    parse_proc_stat_totals(&stat)
}

/// Pure parser for `/proc/stat` content (extracted for testability). Reads the
/// first line, which must be the aggregate `cpu` line.
fn parse_proc_stat_totals(stat: &str) -> Option<(u64, u64)> {
    let line = stat.lines().next()?; // first line is the "cpu " aggregate
    let mut it = line.split_whitespace();
    if it.next()? != "cpu" {
        return None;
    }
    // user nice system idle iowait irq softirq steal guest guest_nice
    let vals: Vec<u64> = it.filter_map(|v| v.parse().ok()).collect();
    if vals.len() < 5 {
        return None;
    }
    let total: u64 = vals.iter().sum();
    let idle = vals[3] + vals[4]; // idle + iowait
    Some((total, idle))
}

/// Derived client-CPU / saturation metrics for a search run.
#[derive(Debug, Clone, Default)]
pub struct Saturation {
    /// Average client cores used across the window (e.g. 6.5). `None` if CPU
    /// sampling was unavailable.
    pub client_cpu_cores_used: Option<f64>,
    /// System-wide CPU utilization over the window, 0.0–1.0. `None` if
    /// unavailable.
    pub system_cpu_pct: Option<f64>,
    /// Logical cores available to the client.
    pub available_cores: usize,
    /// Client requested more worker threads than it has cores.
    pub oversubscribed: bool,
    /// The run is likely client-bound; the DB numbers should not be trusted as
    /// server-side measurements.
    pub client_saturated: bool,
    /// Human-readable reason(s); empty when not saturated.
    pub saturation_reason: String,
}

/// Fraction of a core above which the client is considered CPU-bound.
const CLIENT_CORE_SATURATION: f64 = 0.85;
/// System CPU fraction above which the whole box is considered saturated.
const SYSTEM_SATURATION: f64 = 0.90;

/// Compute saturation for a run from a before/after CPU sample pair, the client
/// concurrency (`parallel`), and the number of logical cores. `before`/`after`
/// are `None` on non-Linux, in which case only the concurrency-vs-cores signal
/// (`oversubscribed`) is available.
pub fn compute(
    before: Option<CpuSample>,
    after: Option<CpuSample>,
    parallel: usize,
    available_cores: usize,
) -> Saturation {
    let oversubscribed = available_cores > 0 && parallel > available_cores;

    let (cores_used, system_pct) = match (before, after) {
        (Some(b), Some(a)) => {
            let d_proc = a.proc_ticks.saturating_sub(b.proc_ticks);
            let d_total = a.total_jiffies.saturating_sub(b.total_jiffies);
            let d_idle = a.idle_jiffies.saturating_sub(b.idle_jiffies);
            if d_total == 0 || available_cores == 0 {
                (None, None)
            } else {
                // Tick unit cancels: proc ticks vs per-core wall jiffies.
                let cores = d_proc as f64 * available_cores as f64 / d_total as f64;
                let sys = (d_total.saturating_sub(d_idle)) as f64 / d_total as f64;
                (Some(cores), Some(sys))
            }
        }
        _ => (None, None),
    };

    let mut reasons: Vec<String> = Vec::new();
    if oversubscribed {
        reasons.push(format!("parallel={} > {} cores", parallel, available_cores));
    }
    if let Some(c) = cores_used {
        if available_cores > 0 && c / available_cores as f64 > CLIENT_CORE_SATURATION {
            reasons.push(format!(
                "client CPU {:.0}% of {} cores",
                100.0 * c / available_cores as f64,
                available_cores
            ));
        }
    }
    if let Some(s) = system_pct {
        if s > SYSTEM_SATURATION {
            reasons.push(format!("system CPU {:.0}%", 100.0 * s));
        }
    }

    Saturation {
        client_cpu_cores_used: cores_used,
        system_cpu_pct: system_pct,
        available_cores,
        oversubscribed,
        client_saturated: !reasons.is_empty(),
        saturation_reason: reasons.join("; "),
    }
}

/// Logical cores available to the process (falls back to 1).
pub fn available_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(proc_ticks: u64, total: u64, idle: u64) -> CpuSample {
        CpuSample {
            proc_ticks,
            total_jiffies: total,
            idle_jiffies: idle,
        }
    }

    #[test]
    fn one_full_core_reads_as_one() {
        // Over the window each of 8 cores advanced 100 jiffies (total 800); the
        // process burned 100 ticks → exactly one core, and only our process was
        // busy so the box is otherwise idle (700 idle jiffies).
        let sat = compute(Some(s(0, 0, 0)), Some(s(100, 800, 700)), 1, 8);
        let cores = sat.client_cpu_cores_used.unwrap();
        assert!((cores - 1.0).abs() < 1e-9, "cores={}", cores);
        assert!((sat.system_cpu_pct.unwrap() - 0.125).abs() < 1e-9);
        assert!(!sat.client_saturated); // 1/8 cores, idle box, not oversubscribed
    }

    #[test]
    fn all_cores_busy_flags_saturated() {
        // Process consumed all 8 cores fully.
        let sat = compute(Some(s(0, 0, 0)), Some(s(800, 800, 0)), 8, 8);
        let cores = sat.client_cpu_cores_used.unwrap();
        assert!((cores - 8.0).abs() < 1e-9, "cores={}", cores);
        assert!(sat.client_saturated);
        assert!(sat.saturation_reason.contains("client CPU"));
    }

    #[test]
    fn oversubscription_flags_without_cpu_samples() {
        let sat = compute(None, None, 64, 8);
        assert!(sat.oversubscribed);
        assert!(sat.client_saturated);
        assert!(sat.client_cpu_cores_used.is_none());
        assert!(sat.saturation_reason.contains("> 8 cores"));
    }

    #[test]
    fn idle_box_not_saturated() {
        // 8 cores, only 80 of 800 jiffies busy (10%), process used 0.5 core.
        let sat = compute(Some(s(0, 0, 0)), Some(s(50, 800, 720)), 4, 8);
        assert!(!sat.client_saturated);
        assert!((sat.system_cpu_pct.unwrap() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn live_sample_parses() {
        // On Linux CI this should read real /proc; elsewhere it's None. Either
        // way it must not panic.
        let _ = sample();
    }

    #[test]
    fn parse_self_ticks_handles_comm_with_spaces_and_parens() {
        // comm = "(evil ) name)" contains both a space and an interior ')'. The
        // parser splits after the LAST ')', so the numeric fields start at the
        // real `state` field. After that ')': index 0=state(R), then 10 more
        // fields, index 11=utime(100), index 12=stime(50) → 150.
        let line = "1234 (evil ) name) R 1 1 1 0 -1 0 0 0 0 0 100 50 3 4 20 0 1 0 999";
        assert_eq!(parse_proc_self_ticks(line), Some(150));
    }

    #[test]
    fn parse_self_ticks_simple_comm() {
        // Plain comm with no tricks: utime=7, stime=8 → 15.
        let line = "42 (cat) R 1 1 1 0 -1 0 0 0 0 0 7 8 0 0";
        assert_eq!(parse_proc_self_ticks(line), Some(15));
    }

    #[test]
    fn parse_self_ticks_none_when_truncated() {
        // Too few numeric fields after ')': utime/stime indices missing → None.
        let line = "42 (cat) R 1 1 1";
        assert_eq!(parse_proc_self_ticks(line), None);
        // No ')' at all → rsplit_once fails → None.
        assert_eq!(parse_proc_self_ticks("no parens here"), None);
    }

    #[test]
    fn parse_stat_totals_basic() {
        // cpu user nice system idle iowait ... → total = sum(all) = 870,
        // idle = idle(700) + iowait(20) = 720.
        assert_eq!(
            parse_proc_stat_totals("cpu 100 0 50 700 20 0 0 0\ncpu0 1 2 3 4 5\n"),
            Some((870, 720))
        );
    }

    #[test]
    fn parse_stat_totals_rejects_non_cpu_first_line() {
        // First token isn't exactly "cpu" (it's "cpu0") → None.
        assert_eq!(parse_proc_stat_totals("cpu0 1 2 3 4 5 6"), None);
    }

    #[test]
    fn parse_stat_totals_rejects_truncated_line() {
        // Fewer than 5 numeric fields → None (can't compute idle+iowait).
        assert_eq!(parse_proc_stat_totals("cpu 1 2 3"), None);
        // Empty content → no first line → None.
        assert_eq!(parse_proc_stat_totals(""), None);
    }

    #[test]
    fn compute_before_equals_after_is_none() {
        // Identical snapshots → d_total == 0 → cpu-derived fields degrade to None.
        let same = s(500, 4000, 3000);
        let sat = compute(Some(same), Some(same), 4, 8);
        assert!(sat.client_cpu_cores_used.is_none());
        assert!(sat.system_cpu_pct.is_none());
        assert!(!sat.client_saturated); // not oversubscribed either (4 <= 8)
    }

    #[test]
    fn compute_flags_system_cpu_from_other_processes() {
        // The box is busy from OTHER processes: over the window the aggregate
        // advanced 1000 jiffies with only 50 idle → system CPU 95% (> 90%), but
        // our process burned only 10 ticks → 10*8/1000 = 0.08 cores, well under
        // the 0.85 client threshold. Saturation must be flagged, and the reason
        // must cite SYSTEM CPU (not client CPU).
        let sat = compute(Some(s(0, 0, 0)), Some(s(10, 1000, 50)), 4, 8);
        let cores = sat.client_cpu_cores_used.unwrap();
        assert!((cores - 0.08).abs() < 1e-9, "cores={}", cores);
        assert!((sat.system_cpu_pct.unwrap() - 0.95).abs() < 1e-9);
        assert!(sat.client_saturated);
        assert!(
            sat.saturation_reason.contains("system CPU"),
            "reason={}",
            sat.saturation_reason
        );
        assert!(
            !sat.saturation_reason.contains("client CPU"),
            "reason={}",
            sat.saturation_reason
        );
        assert_eq!(sat.saturation_reason, "system CPU 95%");
    }
}
