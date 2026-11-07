//! One shared copy of the voice for the whole Mac.
//!
//! Every app starts this binary as its MCP server, but that process is only a
//! relay (a few MB, no model): it passes MCP messages between the app and one
//! background copy, the *daemon*, over a Unix socket. The daemon holds the
//! model and the speaker, so there is only ever one model in memory and one
//! voice playing.
//!
//! * The first relay to need it starts the daemon; a lock file makes sure
//!   only one ever runs.
//! * The daemon exits once its model has gone `TTV_IDLE_UNLOAD_SECS` (300)
//!   without use (freed memory is not returned to macOS while the process
//!   lives), or after `TTV_DAEMON_IDLE_SECS` (600) with no app connected.
//!   Relays reconnect by themselves: the next request restarts the daemon and
//!   the relay replays the app's MCP handshake, so the app never notices.
//! * A relay from a newer build replaces a daemon from an older one.

use crate::audio;
use crate::mcp::{Shared, TtsServer};
use anyhow::{Context, Result, anyhow, bail};
use rmcp::ServiceExt;
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn paths() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let d = audio::scratch_dir()?;
    Ok((d.join("daemon.sock"), d.join("daemon.lock"), d.join("daemon.json"), d.join("daemon.log")))
}

/// Identifies this exact binary, so a rebuilt one can replace an old daemon.
fn build_id() -> String {
    let exe = std::env::current_exe().ok();
    let meta = exe.as_ref().and_then(|e| std::fs::metadata(e).ok());
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "{}:{}:{}:{}",
        env!("CARGO_PKG_VERSION"),
        exe.map(|e| e.display().to_string()).unwrap_or_default(),
        meta.map(|m| m.len()).unwrap_or(0),
        mtime
    )
}

fn secs(var: &str, default: u64) -> Duration {
    Duration::from_secs(std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(default))
}

// ---- the daemon --------------------------------------------------------------------

pub async fn run_daemon() -> Result<()> {
    let (sock, lock_path, info_path, _) = paths()?;
    let lock = File::options().create(true).truncate(false).write(true).open(&lock_path)?;
    if lock.try_lock().is_err() {
        // Another daemon is already running; nothing to do.
        return Ok(());
    }
    let _ = std::fs::remove_file(&sock);
    let listener = tokio::net::UnixListener::bind(&sock).with_context(|| format!("could not listen on {}", sock.display()))?;
    std::fs::write(
        &info_path,
        serde_json::to_vec(&json!({ "pid": std::process::id(), "build": build_id() }))?,
    )?;
    eprintln!("daemon {} listening on {}", std::process::id(), sock.display());

    let shared = Shared::new();
    let idle_exit = secs("TTV_DAEMON_IDLE_SECS", 600);
    let mut last_activity = Instant::now();
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let shared = shared.clone();
                shared.clients.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let (r, w) = stream.into_split();
                    match TtsServer::new(shared.clone()).serve((r, w)).await {
                        Ok(service) => { let _ = service.waiting().await; }
                        Err(e) => eprintln!("client failed to start: {e}"),
                    }
                    shared.clients.fetch_sub(1, Ordering::SeqCst);
                });
            }
            _ = tick.tick() => {
                if shared.voice.busy().is_none() && shared.registry.unload_if_idle() {
                    eprintln!("model idle; exiting to return its memory");
                    break;
                }
                if shared.clients.load(Ordering::SeqCst) > 0 || shared.voice.busy().is_some() {
                    last_activity = Instant::now();
                } else if last_activity.elapsed() >= idle_exit {
                    eprintln!("no apps connected for {} s; exiting", idle_exit.as_secs());
                    break;
                }
                // Another daemon took over (newer build): step aside.
                if std::fs::read(&info_path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                    .and_then(|v| v["pid"].as_u64()) != Some(std::process::id() as u64) {
                    break;
                }
            }
        }
    }
    let _ = std::fs::remove_file(&sock);
    let _ = std::fs::remove_file(&info_path);
    Ok(())
}

// ---- reaching the daemon ---------------------------------------------------------

/// Wait (up to 3 s) until no daemon holds the lock.
fn wait_for_lock_free(lock_path: &std::path::Path) {
    let Ok(f) = File::options().create(true).truncate(false).write(true).open(lock_path) else { return };
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if f.try_lock().is_ok() {
            let _ = f.unlock();
            return;
        }
        std::thread::sleep(Duration::from_millis(30));
    }
}

/// Connect to the daemon, starting it (or replacing an outdated one) first.
pub fn connect() -> Result<UnixStream> {
    let (sock, lock_path, info_path, log_path) = paths()?;
    let running: Option<Value> = std::fs::read(&info_path).ok().and_then(|b| serde_json::from_slice(&b).ok());
    if let Some(info) = &running {
        if info["build"].as_str() == Some(build_id().as_str()) {
            if let Ok(s) = UnixStream::connect(&sock) {
                return Ok(s);
            }
        } else if let Some(pid) = info["pid"].as_u64() {
            // An older build is running. Ask it to quit so this one takes over.
            let _ = Command::new("/bin/kill").arg(pid.to_string()).status();
            wait_for_lock_free(&lock_path);
        }
    } else if let Ok(s) = UnixStream::connect(&sock) {
        return Ok(s);
    }

    let log = File::options().create(true).append(true).open(&log_path)?;
    let exe = std::env::current_exe()?;
    {
        use std::os::unix::process::CommandExt;
        Command::new(&exe)
            .arg("daemon")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .process_group(0) // survives the app that started it
            .spawn()
            .with_context(|| format!("could not start {}", exe.display()))?;
    }
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if let Ok(s) = UnixStream::connect(&sock) {
            return Ok(s);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    bail!("the background voice server did not start; see {}", log_path.display())
}

/// Relay stdio to the daemon. This is what apps run as their MCP server.
///
/// MCP over stdio is one JSON message per line. The relay remembers the
/// app's `initialize` request and `initialized` notification, so when the
/// daemon has exited (it does when idle) it can start a new one and replay
/// the handshake before passing on the next message.
pub fn relay() -> Result<()> {
    use std::sync::{Arc, Mutex};
    let stdout = Arc::new(Mutex::new(std::io::stdout()));
    let mut conn: Option<UnixStream> = None;
    let mut init: Option<(String, Value)> = None; // (line, id)
    let mut initialized: Option<String> = None;
    let mut reader_threads: Vec<std::thread::JoinHandle<()>> = Vec::new();

    let open = |init: &Option<(String, Value)>, initialized: &Option<String>| -> Result<UnixStream> {
        let mut stream = connect()?;
        if let Some((line, id)) = init {
            // Replay the handshake and swallow the reply the app already has.
            writeln!(stream, "{line}")?;
            let mut r = BufReader::new(stream.try_clone()?);
            let mut reply = String::new();
            loop {
                reply.clear();
                if r.read_line(&mut reply)? == 0 {
                    bail!("the voice server closed the connection during start-up");
                }
                if serde_json::from_str::<Value>(reply.trim()).ok().is_some_and(|v| &v["id"] == id) {
                    break;
                }
            }
            if let Some(n) = initialized {
                writeln!(stream, "{n}")?;
            }
        }
        Ok(stream)
    };

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        let is_init = msg["method"] == "initialize";
        if msg["method"] == "notifications/initialized" {
            initialized = Some(line.clone());
        }
        // A daemon that exited while we waited: reconnect.
        if reader_threads.last().is_some_and(|t| t.is_finished()) {
            conn = None;
        }
        for attempt in 0..2 {
            if conn.is_none() {
                // A fresh connection: the handshake is replayed unless this
                // line is the handshake itself.
                let replay = if is_init { None } else { init.clone() };
                let stream = open(&replay, &initialized)?;
                let out = stdout.clone();
                let mut r = BufReader::new(stream.try_clone()?);
                reader_threads.push(std::thread::spawn(move || {
                    let mut buf = String::new();
                    while matches!(r.read_line(&mut buf), Ok(n) if n > 0) {
                        let mut o = out.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = o.write_all(buf.as_bytes());
                        let _ = o.flush();
                        buf.clear();
                    }
                }));
                conn = Some(stream);
            }
            let stream = conn.as_mut().expect("connected");
            if writeln!(stream, "{line}").and_then(|_| stream.flush()).is_ok() {
                break;
            }
            conn = None; // the daemon went away; reconnect once
            if attempt == 1 {
                bail!("could not reach the voice server");
            }
        }
        if is_init {
            init = Some((line.clone(), msg["id"].clone()));
        }
    }
    // The app hung up. Let the daemon finish any reply in flight.
    if let Some(s) = &conn {
        let _ = s.shutdown(std::net::Shutdown::Write);
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    for t in reader_threads {
        while !t.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    Ok(())
}

// ---- a minimal MCP client for the command line ---------------------------------------

/// Call one tool on the daemon and return its structured result.
pub fn call_tool(name: &str, args: Value) -> Result<Value> {
    let stream = connect()?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut send = |v: Value| -> Result<()> {
        writeln!(writer, "{v}")?;
        Ok(())
    };
    let read_reply = |reader: &mut BufReader<UnixStream>, id: u64| -> Result<Value> {
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                bail!("the voice server closed the connection");
            }
            let v: Value = serde_json::from_str(line.trim()).unwrap_or(Value::Null);
            if v["id"].as_u64() == Some(id) {
                if let Some(e) = v.get("error") {
                    bail!("{}", e["message"].as_str().unwrap_or("error"));
                }
                return Ok(v["result"].clone());
            }
        }
    };
    send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "terminal (text-to-voice-local-mcp CLI)", "version": env!("CARGO_PKG_VERSION") }
        }
    }))?;
    read_reply(&mut reader, 1)?;
    send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))?;
    send(json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": { "name": name, "arguments": args } }))?;
    let result = read_reply(&mut reader, 2)?;
    let text = result["content"][0]["text"].as_str().unwrap_or("").to_string();
    if result["isError"].as_bool() == Some(true) {
        return Err(anyhow!(text));
    }
    Ok(match result.get("structuredContent") {
        Some(v) if !v.is_null() => v.clone(),
        _ => serde_json::from_str(&text).unwrap_or(Value::String(text)),
    })
}
