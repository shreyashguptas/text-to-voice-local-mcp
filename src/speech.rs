//! The voice is used by one request at a time. A second request while it is
//! busy is turned away at once with who is using it, instead of waiting.
//!
//! Speaking runs synthesis and playback on two threads, so the first
//! sentence plays while the rest is still being made.

use crate::audio::{self, PlayOutcome};
use crate::registry::{Choice, Registry, SynthStats};
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Who holds the voice right now.
#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
pub struct BusyInfo {
    /// The app using it, as it named itself to MCP (e.g. `claude-code`).
    pub client: String,
    /// `speaking` or `making a file`.
    pub doing: String,
    /// Start of the text being read.
    pub text_preview: String,
    /// Seconds since it started.
    pub seconds: f64,
}

struct Holder {
    client: String,
    doing: String,
    text_preview: String,
    since: Instant,
    cancel: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct Voice {
    holder: Mutex<Option<Holder>>,
}

/// Holds the voice until dropped.
pub struct Guard {
    voice: Arc<Voice>,
    pub cancel: Arc<AtomicBool>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let mut h = self.voice.holder.lock().unwrap_or_else(|e| e.into_inner());
        // Only clear our own hold.
        if h.as_ref().is_some_and(|h| Arc::ptr_eq(&h.cancel, &self.cancel)) {
            *h = None;
        }
    }
}

impl Voice {
    /// Take the voice, or say who has it.
    pub fn try_take(self: &Arc<Self>, client: &str, doing: &str, text: &str) -> Result<Guard, BusyInfo> {
        let mut h = self.holder.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cur) = h.as_ref() {
            return Err(cur.info());
        }
        let cancel = Arc::new(AtomicBool::new(false));
        *h = Some(Holder {
            client: client.to_string(),
            doing: doing.to_string(),
            text_preview: preview(text),
            since: Instant::now(),
            cancel: cancel.clone(),
        });
        Ok(Guard { voice: self.clone(), cancel })
    }

    pub fn busy(&self) -> Option<BusyInfo> {
        self.holder.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(Holder::info)
    }

    /// Stop whatever is using the voice. Returns what was stopped.
    pub fn stop(&self) -> Option<BusyInfo> {
        let h = self.holder.lock().unwrap_or_else(|e| e.into_inner());
        h.as_ref().map(|h| {
            h.cancel.store(true, Ordering::Relaxed);
            h.info()
        })
    }
}

impl Holder {
    fn info(&self) -> BusyInfo {
        BusyInfo {
            client: self.client.clone(),
            doing: self.doing.clone(),
            text_preview: self.text_preview.clone(),
            seconds: (self.since.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        }
    }
}

impl BusyInfo {
    pub fn message(&self) -> String {
        format!(
            "Busy: another chat ({}) is using the voice right now — {} \"{}\" for {:.0} s. \
             Try again when it is done, or call `stop` to cut it off.",
            self.client, self.doing, self.text_preview, self.seconds
        )
    }
}

fn preview(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut s = words.iter().take(8).cloned().collect::<Vec<_>>().join(" ");
    if words.len() > 8 {
        s.push('…');
    }
    s
}

pub struct Speaking {
    first: mpsc::Receiver<Result<(), String>>,
    synth: JoinHandle<Result<SynthStats>>,
    play: JoinHandle<Result<PlayOutcome>>,
}

pub struct Spoken {
    pub stats: SynthStats,
    pub outcome: PlayOutcome,
}

/// Start speaking. The voice is held (via `guard`) until playback ends.
pub fn start(registry: Arc<Registry>, choice: Choice, text: String, speed: f32, volume: f32, guard: Guard) -> Speaking {
    // Synthesis stays at most a few pieces ahead of playback.
    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(3);
    let (first_tx, first) = mpsc::channel();
    let sample_rate = choice.model.info().sample_rate;
    let cancel = guard.cancel.clone();

    let synth = std::thread::spawn({
        let cancel = cancel.clone();
        move || {
            let mut sent_first = false;
            let result = registry.synthesize(&choice, &text, speed, |piece| {
                if !sent_first {
                    sent_first = true;
                    let _ = first_tx.send(Ok(()));
                }
                !cancel.load(Ordering::Relaxed) && tx.send(piece).is_ok()
            });
            if let Err(e) = &result {
                cancel.store(true, Ordering::Relaxed);
                if !sent_first {
                    let _ = first_tx.send(Err(format!("{e:#}")));
                }
            }
            result
        }
    });
    let play = std::thread::spawn(move || {
        let r = audio::play_stream(rx, sample_rate, volume, cancel);
        drop(guard);
        r
    });
    Speaking { first, synth, play }
}

impl Speaking {
    /// Wait until the first sentence is ready, which is when bad requests fail.
    pub fn started(&self) -> Result<()> {
        match self.first.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(anyhow!(e)),
            Err(_) => Err(anyhow!("speech stopped before it started")),
        }
    }

    /// Wait until speech has finished or was stopped.
    pub fn finish(self) -> Result<Spoken> {
        let stats = self.synth.join().map_err(|_| anyhow!("synthesis thread crashed"))??;
        let outcome = self.play.join().map_err(|_| anyhow!("playback thread crashed"))??;
        Ok(Spoken { stats, outcome })
    }
}
