//! Shared utilities for Redis-protocol engines (Redis, Valkey, VectorSets).
//!
//! Provides `INFO COMMANDSTATS` validation to detect silent command failures.

use redis::Connection;

/// Reset server command statistics so subsequent checks start from zero.
/// Non-fatal: warns if the command is not permitted (e.g. Redis Cloud ACLs).
pub fn reset_commandstats(conn: &mut Connection) -> Result<(), String> {
    if let Err(e) = redis::cmd("CONFIG").arg("RESETSTAT").query::<()>(conn) {
        eprintln!(
            "Warning: CONFIG RESETSTAT not available ({}), skipping commandstats validation",
            e
        );
    }
    Ok(())
}

/// Check `INFO COMMANDSTATS` for failed_calls on the given commands.
///
/// Returns `Err` if any of the specified commands have `failed_calls > 0`,
/// listing the offending commands and their failure counts.
pub fn check_commandstats(
    conn: &mut Connection,
    commands: &[&str],
    context: &str,
) -> Result<(), String> {
    let info: String = match redis::cmd("INFO").arg("commandstats").query(conn) {
        Ok(v) => v,
        Err(_) => return Ok(()), // not available, skip validation
    };

    let mut failures = Vec::new();

    for line in info.lines() {
        // Lines look like:
        // cmdstat_hset:calls=317804,usec=8510177522,usec_per_call=26778.07,rejected_calls=0,failed_calls=0
        // cmdstat_FT.SEARCH:calls=160371,...,failed_calls=160186
        let Some((cmd_part, stats_part)) = line.split_once(':') else {
            continue;
        };
        let cmd_name = cmd_part.strip_prefix("cmdstat_").unwrap_or(cmd_part);

        // Case-insensitive match against the commands we care about
        let matches = commands.iter().any(|c| c.eq_ignore_ascii_case(cmd_name));
        if !matches {
            continue;
        }

        // Parse failed_calls from the stats
        for field in stats_part.split(',') {
            if let Some(val) = field.strip_prefix("failed_calls=") {
                if let Ok(n) = val.parse::<u64>() {
                    if n > 0 {
                        failures.push(format!("{}: {} failed_calls", cmd_name, n));
                    }
                }
            }
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
