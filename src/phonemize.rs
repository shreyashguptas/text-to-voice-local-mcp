//! Text to phonemes.
//!
//! Neural voices do not read letters, they read phonemes (the sounds of a
//! word, written in IPA). Two converters are linked into the binary:
//!
//! * **misaki** for English, the converter Kokoro was trained with. Its
//!   dictionaries are embedded, and unknown words fall back to espeak-ng.
//! * **espeak-ng** for every other language, and for Piper voices, which were
//!   trained on espeak-ng output.
//!
//! espeak-ng keeps global state and is not thread safe, so every call goes
//! through one lock.

use anyhow::{Result, anyhow};
use std::sync::{Mutex, OnceLock};

static ESPEAK_LOCK: Mutex<()> = Mutex::new(());

/// espeak-ng's dictionaries and voice rules (8.7 MB), packed into the binary
/// so the tool is one file. Unpacked to /tmp on first use.
const ESPEAK_DATA_ZIP: &[u8] = include_bytes!("../assets/espeak-ng-data.zip");

/// The folder that will hold `espeak-ng-data`. Call [`point_espeak_at_data`]
/// before any thread starts, so espeak-ng finds it.
fn espeak_data_parent() -> std::path::PathBuf {
    let hash = ESPEAK_DATA_ZIP.len() ^ ESPEAK_DATA_ZIP[ESPEAK_DATA_ZIP.len() / 2] as usize;
    crate::audio::scratch_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join(format!("espeak-{}-{hash:x}", env!("CARGO_PKG_VERSION")))
}

/// Tell espeak-ng where its data will be. Must run at start-up, before any
/// other thread exists (it sets an environment variable).
pub fn point_espeak_at_data() {
    if std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY").is_none() {
        // SAFETY: called first thing in main, while the process has one thread.
        unsafe { std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", espeak_data_parent()) };
    }
}

/// Unpack the data the first time espeak-ng is needed. Cheap afterwards.
fn ensure_espeak_data() -> Result<()> {
    static DONE: OnceLock<Result<(), String>> = OnceLock::new();
    DONE.get_or_init(|| {
        let parent = std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(espeak_data_parent);
        let marker = parent.join("espeak-ng-data").join(".complete");
        if marker.exists() {
            return Ok(());
        }
        // Unpack next to the target, then move into place, so two copies of
        // the tool starting together never see a half-written folder.
        if parent.exists() {
            let _ = std::fs::remove_dir_all(&parent); // left half-done by a crash
        }
        let tmp = parent.with_extension(format!("tmp-{}", crate::audio::unique_stamp()));
        let unpack = || -> Result<()> {
            let mut zip = zip::ZipArchive::new(std::io::Cursor::new(ESPEAK_DATA_ZIP))?;
            zip.extract(&tmp)?;
            std::fs::write(tmp.join("espeak-ng-data").join(".complete"), b"")?;
            match std::fs::rename(&tmp, &parent) {
                Ok(()) => Ok(()),
                Err(_) if marker.exists() => Ok(()), // another copy won the race
                Err(e) => Err(e.into()),
            }
        };
        let r = unpack().map_err(|e| format!("could not unpack espeak-ng data to {}: {e}", parent.display()));
        let _ = std::fs::remove_dir_all(&tmp);
        r
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

/// Marks that end a sentence, and marks that end a clause. Both keep the text
/// before them, which is how the models were trained.
pub const SENTENCE_MARKS: &[char] = &['.', '!', '?', '…'];
pub const CLAUSE_MARKS: &[char] = &[',', ';', ':', '—', '–'];

/// English through misaki, US or British.
pub fn english(text: &str, british: bool) -> Result<String> {
    static US: OnceLock<Mutex<misaki_rs::G2P>> = OnceLock::new();
    static GB: OnceLock<Mutex<misaki_rs::G2P>> = OnceLock::new();
    let cell = if british { &GB } else { &US };
    let g2p = cell.get_or_init(|| {
        let lang = if british {
            misaki_rs::Language::EnglishGB
        } else {
            misaki_rs::Language::EnglishUS
        };
        Mutex::new(misaki_rs::G2P::new(lang))
    });
    // misaki's fallback calls espeak-ng, so it shares the espeak lock.
    ensure_espeak_data()?;
    let _espeak = ESPEAK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let g2p = g2p.lock().unwrap_or_else(|e| e.into_inner());
    let (phonemes, _) = g2p
        .g2p(text)
        .map_err(|e| anyhow!("could not read the text as English: {e}"))?;
    // misaki marks words it could not resolve with ❓; drop them rather than
    // feed the model a symbol it does not know.
    Ok(phonemes.replace('❓', ""))
}

/// Any language espeak-ng knows, e.g. `fr-fr`, `es`, `hi`, `pt-br`, `cmn`.
///
/// Punctuation is kept the way phonemizer keeps it (which is what Kokoro's
/// non-English voices and Piper were trained on): the text is cut at each
/// mark, each piece is converted, and the marks are put back.
pub fn espeak(text: &str, voice: &str) -> Result<String> {
    ensure_espeak_data()?;
    let _guard = ESPEAK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = String::with_capacity(text.len() * 2);
    let mut piece = String::new();
    let flush = |piece: &mut String, out: &mut String| -> Result<()> {
        let words = piece.trim();
        if !words.is_empty() {
            // `fr-fr` is not an espeak-ng voice name but `fr` is: retry with
            // the bare language when the full code is refused.
            let base = voice.split(['-', '_']).next().unwrap_or(voice);
            let sentences = espeak_rs::text_to_phonemes(words, voice, None, true, false)
                .or_else(|e| if base != voice { espeak_rs::text_to_phonemes(words, base, None, true, false) } else { Err(e) })
                .map_err(|e| anyhow!("espeak-ng could not read the text as `{voice}`: {e}"))?;
            let joined = sentences.join(" ");
            let ph = joined.trim().trim_end_matches(['.', ',', '?', '!', ';', ':']);
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(ph.trim());
        }
        piece.clear();
        Ok(())
    };
    for c in text.chars() {
        if SENTENCE_MARKS.contains(&c) || CLAUSE_MARKS.contains(&c) {
            flush(&mut piece, &mut out)?;
            let mark = match c {
                '—' | '–' => ',',
                other => other,
            };
            // Collapse runs like "?!" or "..." into one mark.
            if !out.ends_with(SENTENCE_MARKS) && !out.ends_with(CLAUSE_MARKS) {
                out.push(mark);
            }
        } else if c == '\n' {
            piece.push(' ');
        } else {
            piece.push(c);
        }
    }
    flush(&mut piece, &mut out)?;
    Ok(out.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Split a phoneme string into pieces of at most `max` characters, cutting at
/// the least disruptive place: after a sentence, then after a clause, then
/// between words, and only as a last resort inside a word.
///
/// With `first_short`, the first piece is kept to one sentence so playback
/// can start as soon as possible.
pub fn split(phonemes: &str, max: usize, first_short: bool) -> Vec<String> {
    let atoms = atoms(phonemes.trim(), max, 0);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for atom in atoms {
        let cur_len = cur.chars().count();
        let atom_len = atom.chars().count();
        // Close the first piece at its first sentence end, so it is short.
        let close_first = first_short && out.is_empty() && cur.ends_with(SENTENCE_MARKS);
        if !cur.is_empty() && (close_first || cur_len + 1 + atom_len > max) {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&atom);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn atoms(s: &str, max: usize, level: usize) -> Vec<String> {
    if s.is_empty() {
        return vec![];
    }
    if s.chars().count() <= max && level > 0 {
        return vec![s.to_string()];
    }
    let cuts: [&dyn Fn(char) -> bool; 3] = [
        &|c| SENTENCE_MARKS.contains(&c),
        &|c| CLAUSE_MARKS.contains(&c),
        &|c| c == ' ',
    ];
    for (i, is_cut) in cuts.iter().enumerate().skip(level) {
        let mut pieces = Vec::new();
        let mut cur = String::new();
        for c in s.chars() {
            cur.push(c);
            if is_cut(c) {
                pieces.push(std::mem::take(&mut cur));
            }
        }
        pieces.push(cur);
        let pieces: Vec<String> = pieces
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if pieces.len() > 1 || i == 2 {
            return pieces
                .into_iter()
                .flat_map(|p| {
                    if p.chars().count() <= max { vec![p] } else { atoms(&p, max, i + 1) }
                })
                .collect();
        }
    }
    // One unbroken run longer than the limit: slice it.
    let chars: Vec<char> = s.chars().collect();
    chars.chunks(max).map(|c| c.iter().collect()).collect()
}

/// Seconds of silence to put after a piece that ends with this text.
pub fn pause_after(piece: &str) -> f32 {
    match piece.trim_end().chars().last() {
        Some(c) if SENTENCE_MARKS.contains(&c) => 0.25,
        Some(c) if CLAUSE_MARKS.contains(&c) => 0.1,
        _ => 0.0,
    }
}

/// Trim leading and trailing near-silence (below -60 dB of the loudest part),
/// so pieces join without long gaps.
pub fn trim_silence(audio: &[f32]) -> &[f32] {
    const FRAME: usize = 512;
    if audio.len() < FRAME * 2 {
        return audio;
    }
    let rms: Vec<f32> = audio
        .chunks(FRAME)
        .map(|f| (f.iter().map(|x| x * x).sum::<f32>() / f.len() as f32).sqrt())
        .collect();
    let peak = rms.iter().cloned().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return audio;
    }
    let threshold = peak * 0.001; // -60 dB
    let first = rms.iter().position(|&r| r > threshold).unwrap_or(0);
    let last = rms.iter().rposition(|&r| r > threshold).unwrap_or(rms.len() - 1);
    // Keep one frame of margin on each side so consonants are not clipped.
    let start = first.saturating_sub(1) * FRAME;
    let end = ((last + 2) * FRAME).min(audio.len());
    &audio[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_respects_limit_and_sentences() {
        let s = "aaa bbb. ccc ddd, eee. fff";
        let parts = split(s, 12, false);
        assert!(parts.iter().all(|p| p.chars().count() <= 12), "{parts:?}");
        assert_eq!(parts.join(" "), s);
    }

    #[test]
    fn split_first_short() {
        let parts = split("one. two. three.", 510, true);
        assert_eq!(parts, vec!["one.", "two. three."]);
    }

    #[test]
    fn split_long_word() {
        let parts = split(&"x".repeat(25), 10, false);
        assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), vec![10, 10, 5]);
    }

    #[test]
    fn pauses() {
        assert_eq!(pause_after("hi."), 0.25);
        assert_eq!(pause_after("hi,"), 0.1);
        assert_eq!(pause_after("hi"), 0.0);
    }
}

#[cfg(test)]
mod espeak_tests {
    #[test]
    fn languages() {
        super::point_espeak_at_data();
        for (v, text) in [("fr-fr", "Bonjour le monde."), ("es", "Hola."), ("hi", "नमस्ते दुनिया"), ("cmn", "你好世界"), ("ja", "こんにちは")] {
            let p = super::espeak(text, v).unwrap();
            assert!(!p.trim().is_empty(), "{v}: empty");
            println!("{v}: {p}");
        }
    }
}
