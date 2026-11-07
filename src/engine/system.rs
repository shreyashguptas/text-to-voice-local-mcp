//! The voices built into macOS, through `/usr/bin/say`.
//!
//! These are the lowest-quality option and are never picked by default while
//! a neural model is installed. They are listed because they cover many
//! languages, and the downloadable "Premium" and "Enhanced" ones are decent.

use super::{Chunk, LoadedModel, Model, ModelInfo, SynthRequest, VoiceInfo, bcp47, language_name};
use crate::{audio, phonemize};
use anyhow::{Context, Result, bail};
use std::process::Command;

pub const ID: &str = "macos-system";
const SAY: &str = "/usr/bin/say";
const SAMPLE_RATE: u32 = 22_050;
/// Words per minute at speed 1.0.
const BASE_RATE: f32 = 185.0;

const NOVELTY: &[&str] = &[
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Fred", "Good News", "Jester",
    "Junior", "Kathy", "Organ", "Ralph", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox",
];

pub struct SystemModel {
    info: ModelInfo,
    voices: Vec<VoiceInfo>,
}

pub fn detect() -> Option<SystemModel> {
    if !std::path::Path::new(SAY).exists() {
        return None;
    }
    let out = Command::new(SAY).args(["-v", "?"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut voices: Vec<VoiceInfo> = text.lines().filter_map(parse_voice_line).collect();
    voices.sort_by(|a, b| a.language.cmp(&b.language).then(a.id.cmp(&b.id)));
    voices.dedup_by(|a, b| a.id == b.id);
    if voices.is_empty() {
        return None;
    }
    let mut languages: Vec<String> = voices.iter().map(|v| v.language.clone()).collect();
    languages.sort();
    languages.dedup();
    let default_voice = voices
        .iter()
        .filter(|v| v.language == "en-US")
        .min_by_key(|v| {
            let score = if v.id.contains("Premium") {
                0
            } else if v.id.contains("Enhanced") {
                1
            } else if v.notes == "Novelty voice." {
                3
            } else {
                2
            };
            (score, !v.id.starts_with("Samantha"), v.id.clone())
        })
        .map(|v| v.id.clone())
        .unwrap_or_else(|| "Samantha".into());
    let info = ModelInfo {
        id: ID.into(),
        engine: "macos".into(),
        name: "macOS system voices".into(),
        description: format!(
            "The voices built into macOS (the `say` command): {} voices in {} languages. Lower quality \
             than the neural models, but always available. Download Premium or Enhanced voices in \
             System Settings > Accessibility > Spoken Content > System Voice > Manage Voices.",
            voices.len(),
            languages.len()
        ),
        quality: "system".into(),
        path: "/System/Library/Speech/Voices".into(),
        size_mb: 0.0,
        approx_memory_mb: 0.0,
        sample_rate: SAMPLE_RATE,
        languages,
        voice_count: voices.len(),
        default_voice,
        speed_range: [0.5, 2.0],
        license: "Apple (built into macOS)".into(),
        loaded: false,
        is_default: false,
    };
    Some(SystemModel { info, voices })
}

/// `Ava (Premium)        en_US    # Hello! My name is Ava.`
fn parse_voice_line(line: &str) -> Option<VoiceInfo> {
    let left = line.split('#').next()?.trim_end();
    let (name, locale) = left.rsplit_once(char::is_whitespace)?;
    let name = name.trim();
    if name.is_empty() || !locale.contains('_') {
        return None;
    }
    let lang = bcp47(locale);
    let base = name.split(" (").next().unwrap_or(name);
    let notes = if name.contains("Premium") {
        "Premium: the best macOS quality."
    } else if name.contains("Enhanced") {
        "Enhanced quality."
    } else if NOVELTY.contains(&base) {
        "Novelty voice."
    } else {
        "Compact (basic) quality."
    };
    Some(VoiceInfo {
        id: name.to_string(),
        model: ID.into(),
        name: name.to_string(),
        language_name: language_name(&lang),
        language: lang,
        gender: "unknown".into(),
        notes: notes.into(),
    })
}

impl Model for SystemModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn voices(&self) -> &[VoiceInfo] {
        &self.voices
    }
    fn load(&self) -> Result<Box<dyn LoadedModel>> {
        Ok(Box::new(LoadedSystem))
    }
}

struct LoadedSystem;

impl LoadedModel for LoadedSystem {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
    fn plan(&mut self, req: &SynthRequest) -> Result<Vec<Chunk>> {
        let pieces = phonemize::split(&req.text.split_whitespace().collect::<Vec<_>>().join(" "), 2000, true);
        Ok(pieces.into_iter().map(|p| Chunk { payload: p, pause_after: 0.0 }).collect())
    }
    fn render(&mut self, req: &SynthRequest, chunk: &Chunk) -> Result<Vec<f32>> {
        let dir = audio::scratch_dir()?;
        let stamp = audio::unique_stamp();
        let input = dir.join(format!("say-{stamp}.txt"));
        let output = dir.join(format!("say-{stamp}.wav"));
        std::fs::write(&input, &chunk.payload)?;
        let rate = (BASE_RATE * req.speed).round().to_string();
        let status = Command::new(SAY)
            .args(["-v", &req.voice, "-r", &rate, "--file-format=WAVE"])
            .arg(format!("--data-format=LEI16@{SAMPLE_RATE}"))
            .arg("-o")
            .arg(&output)
            .arg("-f")
            .arg(&input)
            .status()
            .context("could not run /usr/bin/say")?;
        let _ = std::fs::remove_file(&input);
        if !status.success() {
            let _ = std::fs::remove_file(&output);
            bail!("`say` failed for voice `{}`", req.voice);
        }
        let bytes = std::fs::read(&output)?;
        let _ = std::fs::remove_file(&output);
        let (samples, _) = audio::read_wav_pcm16(&bytes)?;
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_say_lines() {
        let v = parse_voice_line("Ava (Premium)       en_US    # Hello! My name is Ava.").unwrap();
        assert_eq!((v.id.as_str(), v.language.as_str()), ("Ava (Premium)", "en-US"));
        let v = parse_voice_line("Bad News            en_US    # The light you see").unwrap();
        assert_eq!(v.notes, "Novelty voice.");
        assert!(parse_voice_line("").is_none());
    }
}
