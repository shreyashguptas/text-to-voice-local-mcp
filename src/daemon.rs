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

/// This binary's identity: version, size and modification time of the real
/// file. The path is left out (symlinks and `/tmp` vs `/private/tmp` spell
/// one file many ways), so only a rebuilt binary counts as different.
fn build() -> (String, u64) {
    let exe = std::env::current_exe().ok().and_then(|e| std::fs::canonicalize(e).ok());
    let meta = exe.as_ref().and_then(|e| std::fs::metadata(e).ok());
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (format!("{}:{}:{mtime}", env!("CARGO_PKG_VERSION"), meta.map(|m| m.len()).unwrap_or(0)), mtime)
}

/// Is `pid` a running daemon of this tool (and not some process that reused
/// a stale pid)?
fn is_our_daemon(pid: u64) -> bool {
    Command::new("/bin/ps")
        .args(["-o", "command=", "-p", &pid.to_string()])
        .output()
        .map(|o| {
            let cmd = String::from_utf8_lossy(&o.stdout);
            cmd.contains("text-to-voice-local-mcp") && cmd.trim_end().ends_with(" daemon")
        })
        .unwrap_or(false)
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
        serde_json::to_vec(&json!({ "pid": std::process::id(), "build": build().0, "mtime": build().1 }))?,
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
    let (my_build, my_mtime) = build();
    if let Some(info) = &running {
        let pid = info["pid"].as_u64().unwrap_or(0);
        let newer_than_running = my_mtime > info["mtime"].as_u64().unwrap_or(0);
        if info["build"].as_str() != Some(my_build.as_str()) && newer_than_running && is_our_daemon(pid) {
            // An older build is running: ask it to quit so this one takes over.
            // (An older binary never replaces a newer daemon; it just uses it.)
            let _ = Command::new("/bin/kill").arg(pid.to_string()).status();
            wait_for_lock_free(&lock_path);
        } else if let Ok(s) = UnixStream::connect(&sock) {
            return Ok(s);
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
///
/// It also remembers which requests are still waiting for an answer. If the
/// daemon goes away before answering (a restart for a new build, a crash),
/// each of them gets an error reply instead of hanging, and the relay keeps
/// running: the next request reconnects.
pub fn relay() -> Result<()> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    type Out = Arc<Mutex<std::io::Stdout>>;
    /// Request id -> the connection (generation) it was sent on.
    type Pending = Arc<Mutex<HashMap<String, u64>>>;

    let stdout: Out = Arc::new(Mutex::new(std::io::stdout()));
    let pending: Pending = Arc::default();
    let mut conn: Option<UnixStream> = None;
    let mut reader: Option<std::thread::JoinHandle<()>> = None;
    let mut generation: u64 = 0;
    let mut init: Option<(String, Value)> = None; // (line, id)
    let mut initialized: Option<String> = None;

    fn write_line(out: &Out, line: &str) {
        let mut o = out.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(o, "{}", line.trim_end());
        let _ = o.flush();
    }
    fn error_reply(out: &Out, id: &Value, why: &str) {
        let msg = json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32603, "message": why } });
        write_line(out, &msg.to_string());
    }

    // Connect, replay the handshake (swallowing the reply the app already
    // has) and start a thread that copies the daemon's replies to stdout.
    let open = |generation: u64,
                replay_init: Option<&(String, Value)>,
                replay_initialized: Option<&String>|
     -> Result<(UnixStream, std::thread::JoinHandle<()>)> {
        let mut stream = connect()?;
        let mut r = BufReader::new(stream.try_clone()?);
        if let Some((line, id)) = replay_init {
            stream.set_read_timeout(Some(Duration::from_secs(15)))?;
            writeln!(stream, "{line}")?;
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
            stream.set_read_timeout(None)?;
            if let Some(n) = replay_initialized {
                writeln!(stream, "{n}")?;
            }
        }
        let (out, pending) = (stdout.clone(), pending.clone());
        let thread = std::thread::spawn(move || {
            let mut buf = String::new();
            while matches!(r.read_line(&mut buf), Ok(n) if n > 0) {
                if let Ok(v) = serde_json::from_str::<Value>(buf.trim()) {
                    if v.get("result").is_some() || v.get("error").is_some() {
                        pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&v["id"].to_string());
                    }
                }
                write_line(&out, &buf);
                buf.clear();
            }
            // The daemon went away. Answer whatever it left unanswered on
            // this connection.
            let left: Vec<String> = {
                let mut p = pending.lock().unwrap_or_else(|e| e.into_inner());
                let ids: Vec<String> = p.iter().filter(|(_, g)| **g == generation).map(|(id, _)| id.clone()).collect();
                ids.iter().for_each(|id| {
                    p.remove(id);
                });
                ids
            };
            for id in left {
                let id: Value = serde_json::from_str(&id).unwrap_or(Value::Null);
                error_reply(&out, &id, "The voice server restarted before answering. Please try again.");
            }
        });
        Ok((stream, thread))
    };

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        let is_init = msg["method"] == "initialize";
        let is_initialized = msg["method"] == "notifications/initialized";
        let is_request = msg.get("method").is_some() && msg.get("id").is_some();
        if is_initialized {
            initialized = Some(line.clone());
        }
        if reader.as_ref().is_some_and(|t| t.is_finished()) {
            conn = None; // the daemon exited while we waited
        }
        let id_key = msg["id"].to_string();

        let mut sent = false;
        for _attempt in 0..2 {
            if conn.is_none() {
                // Replay the handshake unless this line is part of it.
                let replay_init = if is_init { None } else { init.as_ref() };
                let replay_initialized = if is_initialized { None } else { initialized.as_ref() };
                generation += 1;
                match open(generation, replay_init, replay_initialized) {
                    Ok((s, t)) => {
                        conn = Some(s);
                        reader = Some(t);
                    }
                    Err(e) => {
                        eprintln!("text-to-voice relay: {e:#}");
                        continue;
                    }
                }
            }
            if is_request {
                pending.lock().unwrap_or_else(|e| e.into_inner()).insert(id_key.clone(), generation);
            }
            let stream = conn.as_mut().expect("connected");
            if writeln!(stream, "{line}").and_then(|_| stream.flush()).is_ok() {
                sent = true;
                break;
            }
            // The daemon went away. Take the request back before its reader
            // answers it with an error; if that already happened, stop here.
            conn = None;
            if is_request && pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id_key).is_none() {
                sent = true;
                break;
            }
        }
        if !sent && is_request {
            error_reply(&stdout, &msg["id"], "Could not reach the local voice server. Please try again.");
        }
        if is_init {
            init = Some((line.clone(), msg["id"].clone()));
        }
    }
    // The app hung up. Let the daemon finish any reply in flight.
    if let Some(s) = &conn {
        let _ = s.shutdown(std::net::Shutdown::Write);
    }
    if let Some(t) = reader {
        let deadline = Instant::now() + Duration::from_secs(10);
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
