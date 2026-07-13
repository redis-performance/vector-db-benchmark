//! CLI smoke tests — the exact commands the docker-build validation job runs
//! (`--help`, `--describe datasets`, `--describe engines`), promoted into the
//! blocking test suite so a regression fails CI without waiting on the
//! non-blocking docker-build job. No engine/container required.

use std::process::Command;

mod common;

/// Run the built binary with `args`, returning (success, stdout, stderr).
fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(common::binary_path())
        .args(args)
        .output()
        .expect("failed to spawn benchmark binary");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn help_exits_zero_and_lists_key_flags() {
    let (ok, stdout, _stderr) = run(&["--help"]);
    assert!(ok, "--help should exit 0");
    // clap renders usage + the flags the docker smoke and users rely on.
    assert!(stdout.contains("Usage"), "help missing Usage: {stdout}");
    for flag in ["--engines", "--datasets", "--describe"] {
        assert!(stdout.contains(flag), "help missing {flag}");
    }
}

#[test]
fn describe_datasets_exits_zero_and_lists_a_known_dataset() {
    let (ok, stdout, stderr) = run(&["--describe", "datasets"]);
    assert!(ok, "--describe datasets should exit 0; stderr={stderr}");
    // random-100 ships in the image and is what the docker smoke benchmarks.
    assert!(
        stdout.contains("random-100"),
        "expected random-100 in --describe datasets output:\n{stdout}"
    );
}

#[test]
fn describe_engines_exits_zero_and_lists_known_engines() {
    let (ok, stdout, stderr) = run(&["--describe", "engines"]);
    assert!(ok, "--describe engines should exit 0; stderr={stderr}");
    assert!(
        stdout.contains("Available engines"),
        "missing header:\n{stdout}"
    );
    // pin a couple of Redis-family engines incl. the newest (dragonfly).
    for engine in ["redis", "dragonfly"] {
        assert!(
            stdout.contains(engine),
            "expected engine '{engine}' in --describe engines output:\n{stdout}"
        );
    }
}

#[test]
fn describe_unknown_option_errors_out() {
    let (ok, _stdout, stderr) = run(&["--describe", "bogus"]);
    assert!(!ok, "--describe bogus should exit non-zero");
    assert!(
        stderr.contains("Unknown describe option"),
        "expected a helpful error on stderr:\n{stderr}"
    );
}
