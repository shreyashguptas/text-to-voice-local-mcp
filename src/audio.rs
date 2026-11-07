//! Audio out: WAV files, other formats through macOS `afconvert`, and
//! playback through CoreAudio. Only the shared background copy (the daemon)
//! ever plays sound, so there is never more than one voice at a time.

use anyhow::{Context, Result, bail};
use std::num::NonZero;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// `/tmp/text-to-voice-local-mcp-<uid>`: the daemon's socket, logs and
/// default audio files. The same for every app this user runs, whatever
/// environment the app passes along. Only this user can open it.
pub fn scratch_dir() -> Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let uid = std::env::var_os("HOME")
        .and_then(|h| std::fs::metadata(h).ok())
        .map(|m| m.uid())
        .unwrap_or(0);
    let dir = PathBuf::from(format!("/tmp/text-to-voice-local-mcp-{uid}"));
    std::fs::create_dir_all(&dir).with_context(|| format!("could not create {}", dir.display()))?;
    let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    Ok(dir)
}

pub fn unique_stamp() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let ms = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_millis();
    format!("{ms}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed))
}

// ---- files -------------------------------------------------------------------

/// Output formats. Everything but WAV is made by `afconvert`, which ships
/// with macOS, so no encoder is linked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// 16-bit PCM WAV. Plays everywhere. About 2.9 MB per minute at 24 kHz.
    Wav,
    /// AAC in an .m4a. About 10x smaller than WAV.
    M4a,
    /// 16-bit PCM AIFF.
    Aiff,
    /// Lossless FLAC. About half the size of WAV.
    Flac,
    /// Core Audio Format, 16-bit PCM.
    Caf,
}

impl Format {
    pub fn extension(self) -> &'static str {
        match self {
            Format::Wav => "wav",
            Format::M4a => "m4a",
            Format::Aiff => "aiff",
            Format::Flac => "flac",
            Format::Caf => "caf",
        }
    }
    pub fn mime(self) -> &'static str {
        match self {
            Format::Wav => "audio/wav",
            Format::M4a => "audio/mp4",
            Format::Aiff => "audio/aiff",
            Format::Flac => "audio/flac",
            Format::Caf => "audio/x-caf",
        }
    }
    pub fn from_path(p: &Path) -> Option<Format> {
        match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "wav" | "wave" => Some(Format::Wav),
            "m4a" | "aac" | "mp4" => Some(Format::M4a),
            "aif" | "aiff" => Some(Format::Aiff),
            "flac" => Some(Format::Flac),
            "caf" => Some(Format::Caf),
            _ => None,
        }
    }
}

pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    b
}

/// Write `samples` to `path` in `format`. Returns the file size in bytes.
pub fn save(samples: &[f32], sample_rate: u32, format: Format, path: &Path) -> Result<u64> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() && !dir.exists() {
            bail!("the folder {} does not exist", dir.display());
        }
    }
    let wav = wav_bytes(samples, sample_rate);
    if format == Format::Wav {
        std::fs::write(path, &wav).with_context(|| format!("could not write {}", path.display()))?;
    } else {
        let tmp = scratch_dir()?.join(format!("convert-{}.wav", unique_stamp()));
        std::fs::write(&tmp, &wav)?;
        let (file_type, data) = match format {
            Format::M4a => ("m4af", "aac"),
            Format::Aiff => ("AIFF", "BEI16"),
            Format::Flac => ("flac", "flac"),
            Format::Caf => ("caff", "LEI16"),
            Format::Wav => unreachable!(),
        };
        let out = std::process::Command::new("/usr/bin/afconvert")
            .args(["-f", file_type, "-d", data])
            .arg(&tmp)
            .arg(path)
            .output()
            .context("could not run /usr/bin/afconvert");
        let _ = std::fs::remove_file(&tmp);
        let out = out?;
        if !out.status.success() {
            bail!("afconvert failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
    }
    Ok(std::fs::metadata(path)?.len())
}

/// Read a 16-bit PCM WAV (as written by `say`) into mono samples.
pub fn read_wav_pcm16(b: &[u8]) -> Result<(Vec<f32>, u32)> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        bail!("not a WAV file");
    }
    let mut i = 12;
    let (mut channels, mut rate, mut bits) = (1u16, 0u32, 16u16);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if id == b"fmt " && body.len() >= 16 {
            channels = u16::from_le_bytes([body[2], body[3]]).max(1);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            if bits != 16 {
                bail!("expected 16-bit audio, got {bits}-bit");
            }
            let frame = channels as usize;
            let samples: Vec<f32> = body
                .chunks_exact(2 * frame)
                .map(|f| i16::from_le_bytes([f[0], f[1]]) as f32 / 32768.0)
                .collect();
            return Ok((samples, rate));
        }
        i += 8 + len + (len & 1);
    }
    bail!("WAV file has no audio data")
}

// ---- playback ------------------------------------------------------------------

/// What happened to a playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayOutcome {
    Finished,
    Stopped,
}

/// Play chunks of mono audio as they arrive on `chunks`, gaplessly. Blocks
/// until everything is played or `cancel` is set.
pub fn play_stream(chunks: Receiver<Vec<f32>>, sample_rate: u32, volume: f32, cancel: Arc<AtomicBool>) -> Result<PlayOutcome> {
    let rate = NonZero::new(sample_rate).context("sample rate is zero")?;
    let mono = NonZero::new(1u16).unwrap();
    // Wait for the first piece before opening the audio device, so a request
    // that fails never touches it.
    let first = loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(PlayOutcome::Stopped);
        }
        match chunks.recv_timeout(Duration::from_millis(40)) {
            Ok(samples) => break samples,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(PlayOutcome::Finished),
        }
    };
    let mut sink = rodio::DeviceSinkBuilder::open_default_sink().context("no audio output device")?;
    sink.log_on_drop(false);
    let player = rodio::Player::connect_new(sink.mixer());
    player.set_volume(volume.clamp(0.0, 2.0));
    player.append(rodio::buffer::SamplesBuffer::new(mono, rate, first));

    let mut producer_done = false;
    let outcome = loop {
        if cancel.load(Ordering::Relaxed) {
            player.stop();
            break PlayOutcome::Stopped;
        }
        if !producer_done {
            match chunks.recv_timeout(Duration::from_millis(40)) {
                Ok(samples) if !samples.is_empty() => {
                    player.append(rodio::buffer::SamplesBuffer::new(mono, rate, samples))
                }
                Ok(_) | Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => producer_done = true,
            }
        } else if player.empty() {
            break PlayOutcome::Finished;
        } else {
            std::thread::sleep(Duration::from_millis(40));
        }
    };
    drop(player);
    drop(sink);
    Ok(outcome)
}
