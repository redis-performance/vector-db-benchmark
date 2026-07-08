//! Dataset download and extraction.
//!
//! Downloads datasets from remote URLs when not available locally.
//! Supports .tgz/.tar.gz archives, .bz2 compressed files, and plain files (.hdf5, .h5).
//! S3 URLs (s3://bucket/key) are converted to HTTPS for public bucket access.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};

/// Convert an S3 URL to an HTTPS URL for public bucket access.
///
/// Handles both formats:
/// - `s3://bucket/key` → `https://bucket.s3.amazonaws.com/key`
/// - Already HTTPS S3 URLs are returned as-is
fn normalize_s3_url(link: &str) -> String {
    if let Some(rest) = link.strip_prefix("s3://") {
        if let Some(slash_pos) = rest.find('/') {
            let bucket = &rest[..slash_pos];
            let key = &rest[slash_pos + 1..];
            return format!("https://{}.s3.amazonaws.com/{}", bucket, key);
        }
    }
    link.to_string()
}

/// Download a dataset from a URL and extract/place it so that `target_path` exists.
///
/// For archives (.tgz/.tar.gz): creates `target_path` as a directory and extracts
/// the archive contents into it (matching Python's `tar.extractall(target_path)`).
///
/// For .bz2 files: decompresses to target_path (stripping .bz2 extension).
///
/// For plain files (.hdf5): copies directly to `target_path`.
///
/// S3 URLs (s3://bucket/key) are automatically converted to HTTPS.
pub fn download_dataset(link: &str, target_path: &Path) -> Result<(), String> {
    let url = normalize_s3_url(link);
    println!("Downloading from {} to {}...", url, target_path.display());

    let client = reqwest::blocking::Client::builder()
        .user_agent("vector-db-benchmark/0.1")
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(&url)
        .send()
        .map_err(|e| format!("Failed to download: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status {}: {}",
            response.status(),
            url
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
    extract_or_move(&tmp_path, target_path, &url)?;

    // Clean up temp file
    let _ = fs::remove_file(&tmp_path);

    Ok(())
}

/// Extract an archive or copy a plain file to the target path.
fn extract_or_move(tmp_path: &Path, target_path: &Path, link: &str) -> Result<(), String> {
    let link_lower = link.to_lowercase();

    if link_lower.ends_with(".tgz") || link_lower.ends_with(".tar.gz") {
        println!("Extracting tgz to {:?}...", target_path);
        extract_tgz(tmp_path, target_path)
    } else if link_lower.ends_with(".bz2") {
        // Decompress bz2 — strip .bz2 from target if present
        let final_target = if target_path.extension().is_some_and(|e| e == "bz2") {
            target_path.with_extension("")
        } else {
            target_path.to_path_buf()
        };
        println!("Extracting bz2 to {:?}...", final_target);
        extract_bz2(tmp_path, &final_target)
    } else if link_lower.ends_with(".hdf5") || link_lower.ends_with(".h5") {
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

/// Decompress a .bz2 file to the target path.
fn extract_bz2(bz2_path: &Path, target_path: &Path) -> Result<(), String> {
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {}", e))?;
    }

    let file = fs::File::open(bz2_path).map_err(|e| format!("Failed to open bz2 file: {}", e))?;
    let mut decoder = bzip2::read::BzDecoder::new(io::BufReader::new(file));

    let mut out_file = fs::File::create(target_path)
        .map_err(|e| format!("Failed to create output file: {}", e))?;
    io::copy(&mut decoder, &mut out_file)
        .map_err(|e| format!("Failed to decompress bz2: {}", e))?;

    println!("Decompression complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_s3_url_s3_scheme() {
        assert_eq!(
            normalize_s3_url("s3://my-bucket/path/to/file.hdf5"),
            "https://my-bucket.s3.amazonaws.com/path/to/file.hdf5"
        );
    }

    #[test]
    fn test_normalize_s3_url_https_passthrough() {
        let url = "https://example.com/file.hdf5";
        assert_eq!(normalize_s3_url(url), url);
    }

    #[test]
    fn test_normalize_s3_url_http_passthrough() {
        let url = "http://example.com/file.tar.gz";
        assert_eq!(normalize_s3_url(url), url);
    }
}
