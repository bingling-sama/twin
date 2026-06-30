import { useEffect, useState, useRef } from "react";
import {
  getHealth,
  getSyncStatus,
  getBatchStatus,
  getConfig,
  updateConfig,
  setGpuEnabled,
  indexImage,
  indexBatch,
  clearIndex,
  trainIndex,
  type HealthData,
  type SyncStatus,
  type BatchStatus,
  type RebuildStatus,
  type RuntimeConfig,
  type IndexStatus,
  type BatchIndexResult,
} from "../api/twin";

export default function Manage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [result, setResult] = useState<IndexStatus | BatchIndexResult | { status: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [rebuildStatus, setRebuildStatus] = useState<RebuildStatus | null>(null);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [directory, setDirectory] = useState("");
  const [saving, setSaving] = useState<string | null>(null); // label of field being saved
  const [batchStatus, setBatchStatus] = useState<BatchStatus | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refreshHealth() {
    try { setHealth(await getHealth()); } catch { /* server off */ }
  }

  async function refreshSync() {
    try { setSyncStatus(await getSyncStatus()); } catch { /* server off */ }
  }

  async function refreshConfig() {
    try { setConfig(await getConfig()); } catch { /* server off */ }
  }

  async function saveConfig(updates: Record<string, unknown>, label: string) {
    setSaving(label);
    try { setConfig(await updateConfig(updates)); refreshHealth(); }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setSaving(null); }
  }

  useEffect(() => { refreshHealth(); refreshSync(); refreshConfig(); }, []);

  // SSE: sync status stream (connects when sync might be running)
  useEffect(() => {
    if (!syncStatus?.running && !rebuilding) return; // only connect when something is happening
    const es = new EventSource("/api/v1/sync/status/stream");
    es.onmessage = (e) => {
      try { setSyncStatus(JSON.parse(e.data)); } catch { /* ignore */ }
    };
    es.addEventListener("done", () => es.close());
    es.onerror = () => es.close();
    return () => es.close();
  }, [syncStatus?.running]);

  // SSE: batch status stream
  useEffect(() => {
    if (!batchStatus?.running) return;
    const es = new EventSource("/api/v1/index/batch/status/stream");
    es.onmessage = (e) => {
      try { setBatchStatus(JSON.parse(e.data)); } catch { /* ignore */ }
    };
    es.addEventListener("done", () => es.close());
    es.onerror = () => es.close();
    return () => es.close();
  }, [batchStatus?.running]);

  // SSE: rebuild status stream
  useEffect(() => {
    if (!rebuilding) return;
    const es = new EventSource("/api/v1/index/rebuild/status/stream");
    es.onmessage = (e) => {
      try { setRebuildStatus(JSON.parse(e.data)); } catch { /* ignore */ }
    };
    es.addEventListener("done", () => es.close());
    es.onerror = () => es.close();
    return () => es.close();
  }, [rebuilding]);

  async function handleSingle() {
    const file = fileRef.current?.files?.[0];
    if (!file) { setError("Select an image file first."); return; }
    setLoading(true); setError(null); setResult(null);
    try { setResult(await indexImage(file)); refreshHealth(); }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setLoading(false); }
  }

  async function handleBatch() {
    if (!directory.trim()) { setError("Enter a directory path."); return; }
    setLoading(true); setError(null); setResult(null);
    // Seed batch status so polling starts immediately
    setBatchStatus({ running: true, total: 0, indexed: 0, failed: 0, current_batch: 0, total_batches: 0, progress_pct: 0, elapsed_ms: 0, eta_ms: 0, rate_img_per_s: 0 });
    try {
      const batchPromise = indexBatch(directory.trim());
      // Give the backend a tick to initialise _batch_state
      await new Promise(r => setTimeout(r, 100));
      const initial = await getBatchStatus();
      setBatchStatus(initial);
      setResult(await batchPromise);
      refreshHealth();
    }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setLoading(false); setBatchStatus(null); }
  }

  async function handleClear() {
    if (!confirm("Clear the entire index? This cannot be undone.")) return;
    setLoading(true); setError(null); setResult(null);
    try { setResult(await clearIndex()); refreshHealth(); }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setLoading(false); }
  }

  async function handleTrain() {
    setTraining(true); setError(null); setResult(null);
    try { setResult(await trainIndex()); refreshHealth(); }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setTraining(false); }
  }

  async function handleToggleGpu() {
    const currentlyOn = !!health?.faiss_gpu_enabled;
    setSaving("gpu");
    try {
      const r = await setGpuEnabled(!currentlyOn);
      setResult(r);
      refreshHealth();
    }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setSaving(null); }
  }

  async function handleSwitchType(newType: string) {
    if (newType === config?.faiss_index_type) return;
    if (!confirm(`Switch index type to "${newType}"?\n\nVectors are preserved — old index is cached to disk for instant switch-back.`)) return;
    setRebuilding(true); setError(null); setResult(null);
    const startTime = Date.now();
    try {
      const updated = await updateConfig({ faiss_index_type: newType });
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      setConfig(updated);
      setResult({ status: "switched", index_type: updated.faiss_index_type, elapsed_s: elapsed });
      refreshHealth();
    }
    catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setRebuilding(false); setRebuildStatus(null); }
  }

  const count = health ? health.indexed_count.toLocaleString() : "—";

  // Shared stat-label style
  const labelCls = "text-[11px] font-medium text-stone uppercase tracking-wider";
  const valueCls = "text-sm font-medium text-ink";
  const monoCls = "text-sm font-medium text-ink font-mono text-xs";
  const cardCls = "bg-white border border-hairline rounded-lg p-4";
  const dot = (on: boolean) => (
    <span className={`w-2 h-2 rounded-full inline-block ${on ? "bg-brand-green" : "bg-danger"}`} />
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-[22px] font-medium text-ink">Manage</h2>
        <button className="text-ink px-3 py-1 rounded-lg text-xs font-semibold hover:bg-surface cursor-pointer border border-hairline" onClick={() => { refreshHealth(); refreshSync(); }}>
          ↻ Refresh
        </button>
      </div>

      {/* ── Sync status banner ── */}
      {syncStatus && syncStatus.running && (
        <section className="mb-5 px-4 py-3 bg-accent-orange/10 border border-accent-orange/30 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full bg-accent-orange animate-pulse inline-block" />
            <h3 className="text-sm font-medium text-ink">Background Sync in Progress</h3>
          </div>
          <div className="flex items-center gap-4 text-[13px] text-slate flex-wrap">
            <span>{syncStatus.indexed_files} / {syncStatus.total_files} indexed</span>
            <span>{syncStatus.skipped_files} skipped</span>
            {syncStatus.failed_files > 0 && <span className="text-danger">{syncStatus.failed_files} failed</span>}
            <span className="text-stone">{syncStatus.progress_pct.toFixed(1)}%</span>
            {syncStatus.rate_img_per_s > 0 && <span className="text-stone">{syncStatus.rate_img_per_s.toFixed(0)} img/s</span>}
            {syncStatus.eta_ms > 0 && <span className="text-stone">ETA: {(syncStatus.eta_ms / 1000).toFixed(0)}s</span>}
          </div>
          <div className="mt-2 w-full bg-surface border border-hairline rounded-full h-2 overflow-hidden">
            <div className="bg-brand-green h-full rounded-full transition-all duration-500" style={{ width: `${syncStatus.progress_pct}%` }} />
          </div>
        </section>
      )}

      {/* ── Rebuilding banner ── */}
      {rebuilding && (
        <section className="mb-5 px-5 py-4 bg-brand-teal-deep text-white rounded-xl">
          <div className="flex items-center gap-3 mb-2">
            <span className="inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <h3 className="text-base font-semibold">Rebuilding Index</h3>
            {rebuildStatus?.elapsed_ms != null && (
              <span className="text-xs text-white/50 font-mono">{(rebuildStatus.elapsed_ms / 1000).toFixed(1)}s</span>
            )}
          </div>
          <p className="text-sm text-white/80">
            <span className="inline-block w-2 h-2 rounded-full bg-brand-green mr-2 align-middle" />
            {rebuildStatus?.phase === "saving_current" && "Caching current index to disk…"}
            {rebuildStatus?.phase === "loading_cached" && "Loading pre-built index from disk…"}
            {rebuildStatus?.phase === "extracting" && `Extracting ${(rebuildStatus?.n_vectors ?? 0).toLocaleString()} vectors…`}
            {rebuildStatus?.phase === "building" && "Building new index structure + adding vectors…"}
            {rebuildStatus?.phase === "training" && "Training IVF clusters (this is the slow part)…"}
            {rebuildStatus?.phase === "saving" && "Saving new index to disk…"}
            {!rebuildStatus?.phase && "Preparing…"}
          </p>
          {rebuildStatus?.n_vectors > 0 && (
            <p className="text-xs text-white/50 mt-2">{rebuildStatus.n_vectors.toLocaleString()} vectors — {rebuildStatus.phase === "training" ? "1–2 minutes" : "5–30 seconds"}</p>
          )}
        </section>
      )}

      {/* ── System Dashboard ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* CLIP Model */}
        <div className={cardCls}>
          <h3 className="text-xs font-semibold text-slate uppercase tracking-wider mb-3">CLIP Model</h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className={labelCls}>Status</span>
              <span className="flex items-center gap-1.5">
                {dot(!!health?.model_loaded)}
                <span className={valueCls}>{health?.model_loaded ? "Loaded" : "Offline"}</span>
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Model</span>
              <span className={monoCls}>{health?.model_name || "—"}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Device</span>
              <span className="flex items-center gap-1.5">
                {dot(health?.device === "cuda")}
                <span className={valueCls}>{health?.device?.toUpperCase() || "—"}</span>
              </span>
            </div>
            {health?.gpu_name && (
              <div className="flex justify-between items-center">
                <span className={labelCls}>GPU</span>
                <span className="text-sm text-slate text-right max-w-[180px] truncate" title={health.gpu_name}>{health.gpu_name}</span>
              </div>
            )}
          </div>
        </div>

        {/* Faiss Index */}
        <div className={cardCls}>
          <h3 className="text-xs font-semibold text-slate uppercase tracking-wider mb-3">Faiss Index</h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className={labelCls}>Runtime</span>
              <span className={monoCls}>{health?.index_type || "—"}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Type</span>
              <select
                className="border border-hairline-strong rounded text-xs font-mono px-2 py-1 bg-white text-ink cursor-pointer disabled:opacity-50"
                value={config?.faiss_index_type || ""}
                onChange={(e) => handleSwitchType(e.target.value)}
                disabled={rebuilding || training || !!saving}
              >
                <option value="flat">flat</option>
                <option value="ivf_flat">ivf_flat</option>
                <option value="ivf_pq">ivf_pq</option>
                <option value="hnsw">hnsw</option>
              </select>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>GPU Accel</span>
              <button
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors cursor-pointer disabled:opacity-50 ${health?.faiss_gpu_enabled ? "bg-brand-green" : "bg-gray-300"}`}
                onClick={handleToggleGpu}
                disabled={!!saving || health?.index_type?.includes("HNSW")}
                title={health?.index_type?.includes("HNSW") ? "HNSW is CPU-only" : (health?.faiss_gpu_enabled ? "Disable GPU" : "Enable GPU")}
              >
                <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${health?.faiss_gpu_enabled ? "translate-x-4.5" : "translate-x-1"}`} />
                {saving === "gpu" && <span className="absolute -right-5 text-xs text-muted">…</span>}
              </button>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Vectors</span>
              <span className={valueCls}>{count}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>nprobe</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={1} max={256}
                  className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[60px] text-right bg-white text-ink"
                  value={config?.nprobe ?? ""}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    if (isNaN(v) || v < 1) return;
                    setConfig(c => c ? { ...c, nprobe: v } : null);
                  }}
                  onBlur={() => {
                    if (config && config.nprobe !== (health as any)?.nprobe) {
                      saveConfig({ nprobe: config.nprobe }, "nprobe");
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                />
                {saving === "nprobe" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
          </div>
        </div>

        {/* Configuration */}
        <div className={cardCls}>
          <h3 className="text-xs font-semibold text-slate uppercase tracking-wider mb-3">Configuration</h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className={labelCls}>Batch Size</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={1} max={512}
                  className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[56px] text-right bg-white text-ink"
                  value={config?.batch_size ?? ""}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    if (isNaN(v) || v < 1) return;
                    setConfig(c => c ? { ...c, batch_size: v } : null);
                  }}
                  onBlur={() => {
                    if (config && config.batch_size !== health?.batch_size) {
                      saveConfig({ batch_size: config.batch_size }, "batch");
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                />
                {saving === "batch" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Auto Upgrade</span>
              <button
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors cursor-pointer ${config?.auto_upgrade_enabled ? "bg-brand-green" : "bg-gray-300"}`}
                onClick={() => saveConfig({ auto_upgrade_enabled: !config?.auto_upgrade_enabled }, "upgrade")}
                disabled={!!saving}
              >
                <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${config?.auto_upgrade_enabled ? "translate-x-4.5" : "translate-x-1"}`} />
                {saving === "upgrade" && <span className="absolute -right-4 text-xs text-muted">…</span>}
              </button>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Auto Save</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={0} max={3600}
                  className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[52px] text-right bg-white text-ink"
                  value={config?.auto_save_interval_s ?? ""}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    if (isNaN(v) || v < 0) return;
                    setConfig(c => c ? { ...c, auto_save_interval_s: v } : null);
                  }}
                  onBlur={() => {
                    if (config && config.auto_save_interval_s !== health?.auto_save_interval_s) {
                      saveConfig({ auto_save_interval_s: config.auto_save_interval_s }, "autosave");
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                />
                <span className="text-[11px] text-stone">s</span>
                {saving === "autosave" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>Top-K</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={1} max={1000}
                  className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[56px] text-right bg-white text-ink"
                  value={config?.top_k ?? ""}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    if (isNaN(v) || v < 1) return;
                    setConfig(c => c ? { ...c, top_k: v } : null);
                  }}
                  onBlur={() => {
                    if (config && config.top_k !== (health as any)?.top_k) {
                      saveConfig({ top_k: config.top_k }, "topk");
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                />
                {saving === "topk" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
            <div className="border-t border-hairline pt-2 mt-2">
              <span className="text-[11px] font-medium text-stone uppercase tracking-wider">Search Filters</span>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>dHash ≤</span>
              <div className="flex items-center gap-1">
                <input type="number" min={0} max={64} className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[48px] text-right bg-white text-ink"
                  value={config?.dhash_threshold ?? ""}
                  onChange={(e) => { const v = parseInt(e.target.value); if (isNaN(v) || v < 0) return; setConfig(c => c ? { ...c, dhash_threshold: v } : null); }}
                  onBlur={() => { if (config && config.dhash_threshold !== (health as any)?.dhash_threshold) saveConfig({ dhash_threshold: config.dhash_threshold }, "dhash"); }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
                {saving === "dhash" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>pHash ≤</span>
              <div className="flex items-center gap-1">
                <input type="number" min={0} max={64} className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[48px] text-right bg-white text-ink"
                  value={config?.phash_threshold ?? ""}
                  onChange={(e) => { const v = parseInt(e.target.value); if (isNaN(v) || v < 0) return; setConfig(c => c ? { ...c, phash_threshold: v } : null); }}
                  onBlur={() => { if (config && config.phash_threshold !== (health as any)?.phash_threshold) saveConfig({ phash_threshold: config.phash_threshold }, "phash"); }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
                {saving === "phash" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className={labelCls}>SSIM ≥</span>
              <div className="flex items-center gap-1">
                <input type="number" min={0} max={1} step={0.01} className="border border-hairline-strong rounded px-2 py-0.5 text-xs font-mono w-[52px] text-right bg-white text-ink"
                  value={config?.ssim_threshold ?? ""}
                  onChange={(e) => { const v = parseFloat(e.target.value); if (isNaN(v) || v < 0) return; setConfig(c => c ? { ...c, ssim_threshold: v } : null); }}
                  onBlur={() => { if (config && config.ssim_threshold !== (health as any)?.ssim_threshold) saveConfig({ ssim_threshold: config.ssim_threshold }, "ssim"); }}
                  onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
                {saving === "ssim" && <span className="text-xs text-muted animate-pulse">…</span>}
              </div>
            </div>
          </div>
        </div>

        {/* Sync Summary */}
        <div className={cardCls}>
          <h3 className="text-xs font-semibold text-slate uppercase tracking-wider mb-3">Sync</h3>
          {syncStatus && syncStatus.total_files > 0 ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className={labelCls}>Status</span>
                <span className="flex items-center gap-1.5">
                  {dot(!syncStatus.running)}
                  <span className={valueCls}>{syncStatus.running ? "Running" : "Complete"}</span>
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className={labelCls}>Indexed</span>
                <span className={valueCls}>{syncStatus.indexed_files.toLocaleString()}</span>
              </div>
              {syncStatus.skipped_files > 0 && (
                <div className="flex justify-between items-center">
                  <span className={labelCls}>Skipped</span>
                  <span className={valueCls}>{syncStatus.skipped_files.toLocaleString()}</span>
                </div>
              )}
              {syncStatus.failed_files > 0 && (
                <div className="flex justify-between items-center">
                  <span className={labelCls}>Failed</span>
                  <span className="text-sm font-medium text-danger">{syncStatus.failed_files.toLocaleString()}</span>
                </div>
              )}
            </div>
          ) : (
            <p className="text-[13px] text-muted">No sync data yet. Sync runs on startup if images dir is populated.</p>
          )}
        </div>
      </div>

      {/* Single upload */}
      <section className="mb-6">
        <h3 className="text-base font-medium text-slate mb-2">Index Image</h3>
        <div className="flex gap-2 items-start">
          <input type="file" accept="image/*" ref={fileRef} className="text-sm text-steel" />
          <button className="bg-brand-green text-brand-teal-deep px-[22px] py-2.5 rounded-full text-sm font-semibold cursor-pointer disabled:opacity-50" onClick={handleSingle} disabled={loading}>
            {loading ? "Indexing..." : "Index"}
          </button>
        </div>
      </section>

      {/* Batch */}
      <section className="mb-6">
        <h3 className="text-base font-medium text-slate mb-2">Batch Index</h3>
        <p className="text-[13px] text-stone mb-2">Directory path on the server filesystem.</p>
        <div className="flex gap-2 items-start">
          <input
            type="text"
            placeholder="/absolute/path/to/images"
            value={directory}
            onChange={(e) => setDirectory(e.target.value)}
            className="border border-hairline-strong rounded-lg px-3 py-2.5 text-base text-ink h-11 max-w-[400px] w-full outline-none focus:border-brand-green-dark"
          />
          <button className="border border-hairline-strong text-ink px-[22px] py-2.5 rounded-full text-sm font-semibold cursor-pointer disabled:opacity-50" onClick={handleBatch} disabled={loading || !directory.trim()}>
            {loading ? "Indexing..." : "Index Batch"}
          </button>
        </div>
        {/* Batch progress bar */}
        {batchStatus?.running && (
          <div className="mt-3 px-4 py-3 bg-surface border border-hairline rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-accent-orange animate-pulse inline-block" />
              <span className="text-sm font-medium text-ink">Batch Indexing</span>
            </div>
            <div className="flex items-center gap-4 text-[13px] text-slate flex-wrap mb-2">
              <span>{batchStatus.indexed.toLocaleString()} / {batchStatus.total.toLocaleString()} indexed</span>
              {batchStatus.failed > 0 && <span className="text-danger">{batchStatus.failed} failed</span>}
              <span className="text-stone">{batchStatus.progress_pct.toFixed(1)}%</span>
              <span className="text-stone">Batch {batchStatus.current_batch}/{batchStatus.total_batches}</span>
              {batchStatus.rate_img_per_s > 0 && <span className="text-stone">{batchStatus.rate_img_per_s.toFixed(0)} img/s</span>}
              {batchStatus.eta_ms > 0 && <span className="text-stone">ETA: {(batchStatus.eta_ms / 1000).toFixed(0)}s</span>}
            </div>
            <div className="w-full bg-white border border-hairline rounded-full h-2 overflow-hidden">
              <div className="bg-brand-green h-full rounded-full transition-all duration-300" style={{ width: `${batchStatus.progress_pct}%` }} />
            </div>
          </div>
        )}
      </section>

      {/* Train */}
      <section className="mb-6">
        <h3 className="text-base font-medium text-slate mb-2">Train Index</h3>
        <p className="text-[13px] text-stone mb-2">
          Upgrade the Faiss index from Flat to IVF for faster search on large datasets.
          {health && <span className="text-slate"> Current type: <code className="bg-surface px-1 rounded text-xs">{health.index_type}</code>.</span>}
        </p>
        <button className="border border-hairline-strong text-ink px-[22px] py-2.5 rounded-full text-sm font-semibold cursor-pointer disabled:opacity-50" onClick={handleTrain} disabled={training || rebuilding}>
          {training ? "Training..." : "Train IVF"}
        </button>
      </section>
      {/* Clear */}
      <section className="border-2 border-dashed border-danger rounded-xl p-4 mb-6">
        <h3 className="text-base font-medium text-slate mb-2">Clear Index</h3>
        <p className="text-[13px] text-stone mb-3">Removes all {count} indexed images. This cannot be undone.</p>
        <button className="bg-danger text-white px-[22px] py-2.5 rounded-full text-sm font-semibold cursor-pointer disabled:opacity-40" onClick={handleClear} disabled={loading}>
          Clear All
        </button>
      </section>

      {error && <div className="bg-danger-soft border border-red-200 text-danger px-4 py-2.5 rounded-lg my-3 text-sm">{error}</div>}
      {result && <pre className="bg-brand-teal-deep text-white font-mono text-sm p-4 rounded-lg overflow-x-auto mt-3">{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
