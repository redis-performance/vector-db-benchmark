//! Dataset download and extraction.
//!
//! Downloads datasets from remote URLs when not available locally.
//! Supports .tgz/.tar.gz archives and plain files (.hdf5, .h5).

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};

/// Download a dataset from a URL and extract/place it so that `target_path` exists.
///
/// For archives (.tgz/.tar.gz): creates `target_path` as a directory and extracts
/// the archive contents into it (matching Python's `tar.extractall(target_path)`).
///
/// For plain files (.hdf5): copies directly to `target_path`.
pub fn download_dataset(link: &str, target_path: &Path) -> Result<(), String> {
    println!("Downloading from {} to {}...", link, target_path.display());

    let client = reqwest::blocking::Client::builder()
        .user_agent("vector-db-benchmark/0.1")
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(link)
        .send()
        .map_err(|e| format!("Failed to download: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status {}: {}",
            response.status(),
            link
        ));
    }

    let total_size = response.content_length().unwrap_or(0);

    // Set up progress bar
    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_message("Downloading");
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_message("Downloading");
        pb
    };

    // Download to temp file
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("vdb_download_{}", std::process::id()));
    let mut tmp_file =
        fs::File::create(&tmp_path).map_err(|e| format!("Failed to create temp file: {}", e))?;

    let mut reader = response;
    let mut buffer = [0u8; 8192];
    let mut downloaded: u64 = 0;

    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| format!("Download read error: {}", e))?;
        if bytes_read == 0 {
            break;
        }
        tmp_file
            .write_all(&buffer[..bytes_read])
            .map_err(|e| format!("Failed to write temp file: {}", e))?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("Downloaded");

    // Extract or move
    extract_or_move(&tmp_path, target_path, link)?;

    // Clean up temp file
    let _ = fs::remove_file(&tmp_path);

    Ok(())
}

/// Extract an archive or copy a plain file to the target path.
fn extract_or_move(tmp_path: &Path, target_path: &Path, link: &str) -> Result<(), String> {
    let link_lower = link.to_lowercase();

    if link_lower.ends_with(".tgz") || link_lower.ends_with(".tar.gz") {
        // Extract into target_path directory (create it first).
        // Matches Python: tar.extractall(target_path)
        println!("Extracting to {:?}...", target_path);
        extract_tgz(tmp_path, target_path)
    } else if link_lower.ends_with(".hdf5") || link_lower.ends_with(".h5") {
        // Plain file — copy directly to target_path
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {}", e))?;
        }
        fs::copy(tmp_path, target_path).map_err(|e| format!("Failed to copy file: {}", e))?;
        println!("Saved to {:?}", target_path);
        Ok(())
    } else {
        // Try tgz extraction as default for unknown extensions
        println!("Extracting to {:?}...", target_path);
        extract_tgz(tmp_path, target_path)
    }
}

/// Extract a .tgz/.tar.gz archive into the target directory.
fn extract_tgz(archive_path: &Path, target_dir: &Path) -> Result<(), String> {
    let file =
        fs::File::open(archive_path).map_err(|e| format!("Failed to open archive: {}", e))?;

    let decoder = GzDecoder::new(io::BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);

    fs::create_dir_all(target_dir).map_err(|e| format!("Failed to create target dir: {}", e))?;

    archive
        .unpack(target_dir)
        .map_err(|e| format!("Failed to extract archive: {}", e))?;

    println!("Extraction complete.");
    Ok(())
}
