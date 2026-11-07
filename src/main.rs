//! text-to-voice-local-mcp: local text-to-speech for every app on the Mac.
//!
//! Run with no arguments to serve MCP over stdio. That process is a small
//! relay to one shared background copy (the daemon) that holds the model; the
//! subcommands go through the same daemon. See `daemon.rs`.

mod audio;
mod daemon;
mod discovery;
mod engine;
mod mcp;
mod phonemize;
mod registry;
mod speech;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::io::Read;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(
    name = "text-to-voice-local-mcp",
    version,
    about = "Local text-to-speech for any app on your Mac, over MCP.",
    long_about = "Finds the text-to-speech models already on this Mac (Kokoro, Piper, macOS voices) and \
                  lets any app use them. With no command it runs as an MCP server on stdio. \
                  All apps share one background copy of the model, which starts when first \
                  needed and exits when no app has used it for a while."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the MCP server on stdio (the default). Relays to the shared background copy.
    Serve,
    /// Run the shared background copy. Started automatically; you do not need to run this.
    #[command(hide = true)]
    Daemon,
    /// Show whether the voice is in use, which model is loaded, and memory use.
    Status,
    /// List installed models and the folders searched.
    Models {
        #[arg(long)]
        json: bool,
    },
    /// List voices.
    Voices {
        /// Only this model.
        #[arg(long)]
        model: Option<String>,
        /// Only this language: `en`, `en-GB`, `fr`, `French`, ...
        #[arg(long, short)]
        language: Option<String>,
        /// Only `female` or `male`.
        #[arg(long)]
        gender: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Say text out loud. Reads standard input when no text is given.
    Speak {
        text: Vec<String>,
        #[command(flatten)]
        voice: VoiceArgs,
        /// Loudness, 0.0 to 2.0.
        #[arg(long, default_value_t = 1.0)]
        volume: f32,
    },
    /// Save text as an audio file. Reads standard input when no text is given.
    Save {
        text: Vec<String>,
        /// Output file (.wav, .m4a, .aiff, .flac, .caf).
        #[arg(long, short)]
        out: String,
        #[command(flatten)]
        voice: VoiceArgs,
    },
    /// Stop speaking, whichever app started it.
    Stop,
}

#[derive(clap::Args)]
struct VoiceArgs {
    /// Voice id or name, e.g. af_heart.
    #[arg(long, short)]
    voice: Option<String>,
    /// Model id, e.g. kokoro-v1.0.
    #[arg(long, short)]
    model: Option<String>,
    /// Language when no voice is given: en-US, en-GB, fr, es, ja, ...
    #[arg(long, short)]
    language: Option<String>,
    /// Speed, 0.5 to 2.0.
    #[arg(long, short, default_value_t = 1.0)]
    speed: f32,
}

fn main() -> Result<()> {
    phonemize::point_espeak_at_data();
    let cli = Cli::parse();
    let runtime = || tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build();
    match cli.command.unwrap_or(Cmd::Serve) {
        Cmd::Serve => daemon::relay(),
        Cmd::Daemon => runtime()?.block_on(daemon::run_daemon()),
        Cmd::Status => {
            println!("{}", serde_json::to_string_pretty(&daemon::call_tool("status", json!({}))?)?);
            Ok(())
        }
        Cmd::Models { json } => models(json),
        Cmd::Voices { model, language, gender, json } => voices(model, language, gender, json),
        Cmd::Speak { text, voice, volume } => speak(read_text(text)?, voice, volume),
        Cmd::Save { text, out, voice } => save(read_text(text)?, out, voice),
        Cmd::Stop => {
            let r = daemon::call_tool("stop", json!({}))?;
            match r["stopped"].as_object() {
                Some(b) => eprintln!("stopped {} ({})", b["client"].as_str().unwrap_or("?"), b["text_preview"].as_str().unwrap_or("")),
                None => eprintln!("nothing was playing"),
            }
            Ok(())
        }
    }
}

fn read_text(words: Vec<String>) -> Result<String> {
    if !words.is_empty() && words != ["-"] {
        return Ok(words.join(" "));
    }
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    if s.trim().is_empty() {
        bail!("no text given");
    }
    Ok(s)
}

fn s(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn models(as_json: bool) -> Result<()> {
    let r = daemon::call_tool("list_models", json!({ "refresh": true }))?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }
    let models = r["models"].as_array().cloned().unwrap_or_default();
    if models.is_empty() {
        println!("No voice models found.");
    }
    for m in &models {
        println!(
            "{}{}  [{}]  {} voices, {} languages, {} MB on disk, ~{} MB in memory{}",
            s(&m["id"]),
            if m["is_default"] == true { " (default)" } else { "" },
            s(&m["engine"]),
            s(&m["voice_count"]),
            m["languages"].as_array().map(|a| a.len()).unwrap_or(0),
            s(&m["size_mb"]),
            s(&m["approx_memory_mb"]),
            if m["loaded"] == true { "  LOADED" } else { "" }
        );
        println!("    {}", s(&m["description"]));
        let langs: Vec<String> = m["languages"].as_array().into_iter().flatten().map(s).collect();
        println!("    languages: {}", langs.join(", "));
        println!(
            "    default voice: {}   speed: {}-{}   license: {}",
            s(&m["default_voice"]),
            s(&m["speed_range"][0]),
            s(&m["speed_range"][1]),
            s(&m["license"])
        );
        println!("    {}\n", s(&m["path"]));
    }
    println!("Searched:");
    for r in r["searched"].as_array().into_iter().flatten() {
        println!("    {} (depth {})", s(&r["path"]), s(&r["depth"]));
    }
    Ok(())
}

fn voices(model: Option<String>, language: Option<String>, gender: Option<String>, as_json: bool) -> Result<()> {
    let r = daemon::call_tool("list_voices", json!({ "model": model, "language": language, "gender": gender }))?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&r["voices"])?);
        return Ok(());
    }
    let v = r["voices"].as_array().cloned().unwrap_or_default();
    let w = v.iter().map(|x| s(&x["id"]).len()).max().unwrap_or(5).max(5);
    println!("{:w$}  {:30}  {:7}  {:14}  NOTES", "VOICE", "LANGUAGE", "GENDER", "MODEL");
    for x in &v {
        println!(
            "{:w$}  {:30}  {:7}  {:14}  {}",
            s(&x["id"]),
            format!("{} {}", s(&x["language"]), s(&x["language_name"])),
            s(&x["gender"]),
            s(&x["model"]),
            s(&x["notes"])
        );
    }
    println!("\n{} voices", v.len());
    Ok(())
}

fn speak(text: String, v: VoiceArgs, volume: f32) -> Result<()> {
    let r = daemon::call_tool(
        "speak",
        json!({ "text": text, "voice": v.voice, "model": v.model, "language": v.language,
                "speed": v.speed, "volume": volume, "wait": true }),
    )?;
    eprintln!("{} / {}: {}, {} s", s(&r["model"]), s(&r["voice"]), s(&r["status"]), s(&r["duration_seconds"]));
    Ok(())
}

fn save(text: String, out: String, v: VoiceArgs) -> Result<()> {
    // The daemon runs in another folder, so send an absolute path.
    let p = std::path::PathBuf::from(&out);
    let abs = if p.is_absolute() || out.starts_with('~') {
        out
    } else {
        std::env::current_dir()?.join(p).display().to_string()
    };
    let r = daemon::call_tool(
        "synthesize",
        json!({ "text": text, "voice": v.voice, "model": v.model, "language": v.language,
                "speed": v.speed, "output_path": abs }),
    )?;
    println!("{}", s(&r["path"]));
    eprintln!(
        "{} s of speech, {} / {}, made in {} s (+{} s loading)",
        s(&r["duration_seconds"]),
        s(&r["model"]),
        s(&r["voice"]),
        s(&r["synth_seconds"]),
        s(&r["load_seconds"])
    );
    Ok(())
}

/// Resident memory of this process in megabytes.
pub fn memory_mb() -> f64 {
    std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<f64>().ok())
        .map(|kb| (kb / 1024.0 * 10.0).round() / 10.0)
        .unwrap_or(0.0)
}
