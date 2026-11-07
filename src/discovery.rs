//! Finding voice models on disk.
//!
//! Nothing is downloaded. The tool looks in the places models usually end up
//! and reports what it finds. Add more places with `TTV_MODEL_PATHS`
//! (colon-separated folders).

use crate::engine::{self, Model};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A folder to search and how deep to go.
#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
pub struct Root {
    pub path: String,
    pub depth: usize,
}

const SKIP: &[&str] = &[
    "node_modules", ".git", "target", "__pycache__", ".Trash", "Caches", "site-packages", "dist-packages",
    "build", ".npm", ".cargo", ".rustup", "Photos Library.photoslibrary", "Music", "Movies", "Pictures",
];

pub fn roots() -> Vec<Root> {
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/"));
    let mut roots: Vec<(PathBuf, usize)> = Vec::new();
    if let Ok(extra) = std::env::var("TTV_MODEL_PATHS") {
        roots.extend(extra.split(':').filter(|s| !s.is_empty()).map(|s| (PathBuf::from(s), 8)));
    }
    roots.push((home.join("Library/Application Support/text-to-voice-local-mcp/models"), 6));
    let hf = std::env::var("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|_| home.join(".cache/huggingface/hub"));
    roots.push((hf, 7));
    roots.push((home.join(".cache"), 5));
    roots.push((home.join(".local/share"), 5));
    roots.push((home.join("Library/Application Support"), 4));
    roots.push((home.join("models"), 5));
    roots.push((home.join("Models"), 5));
    roots.push((home.join("Developer"), 5));
    roots.push((home.join("Documents"), 5));
    roots.push((home.join("Downloads"), 3));
    roots
        .into_iter()
        .map(|(p, depth)| Root { path: p.display().to_string(), depth })
        .collect()
}

/// Every model found, best first.
pub fn scan() -> Vec<Arc<dyn Model>> {
    let mut onnx_files = Vec::new();
    let mut seen_dirs = HashSet::new();
    for root in roots() {
        walk(Path::new(&root.path), root.depth, &mut onnx_files, &mut seen_dirs);
    }

    let mut models: Vec<Arc<dyn Model>> = Vec::new();
    let mut seen_files = HashSet::new();
    for path in onnx_files {
        // The same file can be reached twice (Hugging Face snapshots are
        // symlinks; one root can sit inside another). Keep the first.
        let real = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let size = std::fs::metadata(&real).map(|m| m.len()).unwrap_or(0);
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if !seen_files.insert(real.clone()) || !seen_files.insert(PathBuf::from(format!("{name}#{size}"))) {
            continue;
        }
        let lower = path.to_string_lossy().to_lowercase();
        if lower.contains("kokoro") {
            if let Some(m) = engine::kokoro::detect(&path) {
                models.push(Arc::new(m));
                continue;
            }
        }
        if let Some(m) = engine::piper::detect(&path) {
            models.push(Arc::new(m));
        }
    }
    if let Some(m) = engine::system::detect() {
        models.push(Arc::new(m));
    }

    // Unique ids: a second copy of `kokoro-v1.0` becomes `kokoro-v1.0-2`.
    let mut ids = HashSet::new();
    let mut out: Vec<Arc<dyn Model>> = Vec::new();
    for m in models {
        let id = m.info().id.clone();
        if ids.insert(id.clone()) {
            out.push(m);
        } else {
            let mut n = 2;
            while ids.contains(&format!("{id}-{n}")) {
                n += 1;
            }
            let new_id = format!("{id}-{n}");
            ids.insert(new_id.clone());
            out.push(Arc::new(Renamed::new(m, new_id)));
        }
    }
    out.sort_by_key(|m| rank(m.as_ref()));
    out
}

/// Lower is better: neural before system, full precision before quantized,
/// Kokoro before Piper.
fn rank(m: &dyn Model) -> (u8, u8, String) {
    let i = m.info();
    let quality = match i.quality.as_str() {
        "neural-high" => 0,
        "neural" => 1,
        _ => 2,
    };
    let engine = match i.engine.as_str() {
        "kokoro" => 0,
        "piper" => 1,
        _ => 2,
    };
    (quality, engine, i.id.clone())
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>) {
    let Ok(real) = std::fs::canonicalize(dir) else { return };
    if !seen.insert(real) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name();
        let name = name.to_string_lossy();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            if depth > 0 && !SKIP.contains(&name.as_ref()) && !name.ends_with(".app") && !name.starts_with('.') {
                walk(&path, depth - 1, out, seen);
            }
        } else if name.ends_with(".onnx") && (ft.is_file() || ft.is_symlink() && path.is_file()) {
            out.push(path);
        }
    }
}

/// A model with its id changed, for duplicates.
struct Renamed {
    inner: Arc<dyn Model>,
    info: engine::ModelInfo,
    voices: Vec<engine::VoiceInfo>,
}

impl Renamed {
    fn new(inner: Arc<dyn Model>, id: String) -> Self {
        let mut info = inner.info().clone();
        info.id = id.clone();
        let voices = inner
            .voices()
            .iter()
            .cloned()
            .map(|mut v| {
                v.model = id.clone();
                v
            })
            .collect();
        Self { inner, info, voices }
    }
}

impl Model for Renamed {
    fn info(&self) -> &engine::ModelInfo {
        &self.info
    }
    fn voices(&self) -> &[engine::VoiceInfo] {
        &self.voices
    }
    fn load(&self) -> anyhow::Result<Box<dyn engine::LoadedModel>> {
        self.inner.load()
    }
}
