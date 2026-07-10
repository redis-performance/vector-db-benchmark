//! Shared utilities for Redis-protocol engines (Redis, Valkey, VectorSets).
//!
//! Provides `INFO COMMANDSTATS` validation to detect silent command failures.

use redis::Connection;
use std::collections::HashMap;

/// Baseline snapshot of failed_calls per command, used when CONFIG RESETSTAT is unavailable.
pub type CommandStatsBaseline = HashMap<String, u64>;

/// Parse `INFO COMMANDSTATS` output into a map of command → failed_calls.
fn parse_failed_calls(info: &str) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for line in info.lines() {
        let Some((cmd_part, stats_part)) = line.split_once(':') else {
            continue;
        };
        let cmd_name = cmd_part
            .strip_prefix("cmdstat_")
            .unwrap_or(cmd_part)
            .to_ascii_uppercase();

        for field in stats_part.split(',') {
            if let Some(val) = field.strip_prefix("failed_calls=") {
                if let Ok(n) = val.parse::<u64>() {
                    map.insert(cmd_name.clone(), n);
                }
            }
        }
    }
    map
}

/// Reset server command statistics so subsequent checks start from zero.
/// If CONFIG RESETSTAT is not permitted (e.g. Redis Cloud ACLs), captures
/// a baseline snapshot of current failed_calls so check_commandstats can
/// compare against it.
pub fn reset_commandstats(conn: &mut Connection) -> Result<Option<CommandStatsBaseline>, String> {
    match redis::cmd("CONFIG").arg("RESETSTAT").query::<()>(conn) {
        Ok(()) => Ok(None),
        Err(e) => {
            eprintln!(
                "Warning: CONFIG RESETSTAT not available ({}), will use baseline diff for commandstats",
                e
            );
            // Snapshot current stats as baseline
            let info: String = redis::cmd("INFO")
                .arg("commandstats")
                .query(conn)
                .unwrap_or_default();
            Ok(Some(parse_failed_calls(&info)))
        }
    }
}

/// Check `INFO COMMANDSTATS` for failed_calls on the given commands.
///
/// If a baseline is provided (from a failed RESETSTAT), only reports failures
/// that are NEW since the baseline was captured.
///
/// Returns `Err` if any of the specified commands have new `failed_calls`,
/// listing the offending commands and their failure counts.
pub fn check_commandstats(
    conn: &mut Connection,
    commands: &[&str],
    context: &str,
    baseline: Option<&CommandStatsBaseline>,
) -> Result<(), String> {
    let info: String = match redis::cmd("INFO").arg("commandstats").query(conn) {
        Ok(v) => v,
        Err(_) => return Ok(()), // not available, skip validation
    };

    let current = parse_failed_calls(&info);
    let mut failures = Vec::new();

    for cmd in commands {
        let cmd_upper = cmd.to_ascii_uppercase();
        let current_fails = current.get(&cmd_upper).copied().unwrap_or(0);
        let baseline_fails = baseline
            .and_then(|b| b.get(&cmd_upper))
            .copied()
            .unwrap_or(0);
        let new_fails = current_fails.saturating_sub(baseline_fails);
        if new_fails > 0 {
            failures.push(format!("{}: {} failed_calls", cmd, new_fails));
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Command failures detected after {}: {}",
            context,
            failures.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::parse_failed_calls;

    #[test]
    fn parses_failed_calls_and_uppercases_command() {
        let info = "cmdstat_FT.SEARCH:calls=10,usec=1234,failed_calls=3\r\n\
                    cmdstat_hset:calls=100,usec=50,rejected_calls=0,failed_calls=0\r\n";
        let m = parse_failed_calls(info);
        assert_eq!(m.get("FT.SEARCH"), Some(&3));
        // lowercase `cmdstat_hset` → command name uppercased to HSET
        assert_eq!(m.get("HSET"), Some(&0));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn skips_lines_without_colon_or_failed_calls() {
        // Header lines, blank lines, and stat lines with no failed_calls field
        // must be ignored rather than producing spurious/zero entries.
        let info = "# Commandstats\r\n\r\ncmdstat_ping:calls=1,usec=1\r\n";
        let m = parse_failed_calls(info);
        assert!(m.is_empty(), "got {:?}", m);
    }

    #[test]
    fn handles_missing_cmdstat_prefix_and_bad_number() {
        // A line without the `cmdstat_` prefix keeps its name as-is (uppercased);
        // an unparseable failed_calls value is skipped.
        let info = "FT.CREATE:failed_calls=2\r\ncmdstat_bad:failed_calls=notanumber";
        let m = parse_failed_calls(info);
        assert_eq!(m.get("FT.CREATE"), Some(&2));
        assert!(!m.contains_key("BAD"));
    }
}
