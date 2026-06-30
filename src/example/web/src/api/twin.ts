import axios from "axios";

const BASE = "/api/v1";
const api = axios.create({ baseURL: BASE });

// ── Health ──────────────────────────────────────────────────────────
export interface HealthData {
  status: string;
  indexed_count: number;
  model_loaded: boolean;
  index_type: string;
  // CLIP runtime
  device: string;
  model_name: string;
  gpu_name: string;
  // Faiss runtime
  faiss_gpu_enabled: boolean;
  faiss_index_type: string;
  // Config
  batch_size: number;
  auto_upgrade_enabled: boolean;
  auto_save_interval_s: number;
}

export async function getHealth(): Promise<HealthData> {
  const { data } = await api.get<HealthData>("/health");
  return data;
}

// ── Search ──────────────────────────────────────────────────────────
export interface StageStats {
  in: number;
  out: number;
  elapsed_ms: number;
}

export interface SearchResult {
  id: number;
  filename: string;
  distance: number;
  match_level: "confirmed" | "suspected" | "none";
  stages_passed: number;
  dhash_distance: number;
  phash_distance: number;
  ssim_score: number;
  dhash_hex: string;
  phash_hex: string;
  path: string;
}

export interface SearchResponse {
  results: SearchResult[];
  count: number;
  query_time_ms: number;
  stages: Record<string, StageStats>;
}

export async function searchImage(file: File): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<SearchResponse>("/search", form);
  return data;
}

// ── Index (single) ──────────────────────────────────────────────────
export interface IndexStatus {
  status: string;
  id: number;
  filename: string;
}

export async function indexImage(file: File): Promise<IndexStatus> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<IndexStatus>("/index", form);
  return data;
}

// ── Index (batch) ───────────────────────────────────────────────────
export interface BatchIndexResult {
  status: string;
  total: number;
  indexed: number;
  failed: number;
  time_ms: number;
}

export async function indexBatch(directory: string): Promise<BatchIndexResult> {
  const { data } = await api.post<BatchIndexResult>("/index/batch", { directory });
  return data;
}

// ── Batch Status (polled during indexing) ───────────────────────────
export interface BatchStatus {
  running: boolean;
  total: number;
  indexed: number;
  failed: number;
  current_batch: number;
  total_batches: number;
  progress_pct: number;
  elapsed_ms: number;
  eta_ms: number;
  rate_img_per_s: number;
}

export async function getBatchStatus(): Promise<BatchStatus> {
  const { data } = await api.get<BatchStatus>("/index/batch/status");
  return data;
}

// ── Rebuild Status (polled during index type switch) ────────────────
export interface RebuildStatus {
  running: boolean;
  phase: string;
  n_vectors: number;
  elapsed_ms: number;
}

export async function getRebuildStatus(): Promise<RebuildStatus> {
  const { data } = await api.get<RebuildStatus>("/index/rebuild/status");
  return data;
}

// ── Clear ───────────────────────────────────────────────────────────
export async function clearIndex(): Promise<{ status: string }> {
  const { data } = await api.delete<{ status: string }>("/index");
  return data;
}

// ── Sync Status ─────────────────────────────────────────────────────
export interface SyncStatus {
  running: boolean;
  total_files: number;
  indexed_files: number;
  skipped_files: number;
  failed_files: number;
  progress_pct: number;
  elapsed_ms: number;
  eta_ms: number;
  rate_img_per_s: number;
}

export async function getSyncStatus(): Promise<SyncStatus> {
  const { data } = await api.get<SyncStatus>("/sync/status");
  return data;
}

// ── GPU Toggle ──────────────────────────────────────────────────────
export async function setGpuEnabled(enabled: boolean): Promise<{ status: string; index_type?: string; reason?: string }> {
  const { data } = await api.post("/index/gpu", { enabled });
  return data;
}

// ── Train Index ─────────────────────────────────────────────────────
export async function trainIndex(): Promise<{ status: string; index_type: string }> {
  const { data } = await api.post<{ status: string; index_type: string }>("/index/train");
  return data;
}

// ── Config ──────────────────────────────────────────────────────────
export interface RuntimeConfig {
  faiss_index_type: string;
  auto_upgrade_enabled: boolean;
  auto_save_interval_s: number;
  batch_size: number;
  top_k: number;
  nprobe: number;
  hnsw_ef_search: number;
  dhash_threshold: number;
  phash_threshold: number;
  ssim_threshold: number;
}

export type ConfigUpdate = Partial<RuntimeConfig>;

export async function getConfig(): Promise<RuntimeConfig> {
  const { data } = await api.get<RuntimeConfig>("/config");
  return data;
}

export async function updateConfig(updates: ConfigUpdate): Promise<RuntimeConfig> {
  const { data } = await api.patch<RuntimeConfig>("/config", updates);
  return data;
}

// ── Browse ──────────────────────────────────────────────────────────
export interface IndexedItem {
  id: number;
  filename: string;
  dhash: string;
  path: string;
}

export interface IndexListResponse {
  items: IndexedItem[];
  total: number;
  page: number;
  page_size: number;
}

export async function listIndex(page = 1, pageSize = 50): Promise<IndexListResponse> {
  const { data } = await api.get<IndexListResponse>("/index", {
    params: { page, page_size: pageSize },
  });
  return data;
}

export function imageUrl(filename: string): string {
  return `${BASE}/images/${encodeURIComponent(filename)}`;
}
