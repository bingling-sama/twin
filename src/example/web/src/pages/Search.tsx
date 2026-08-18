import { useState, useRef, useEffect, useCallback, type DragEvent, type ChangeEvent, type MouseEvent as RMouseEvent } from "react";
import { searchImage, imageUrl, type SearchResult, type SearchResponse } from "../api/twin";

/* ------------------------------------------------------------------ */
/* Zoomable image viewer                                              */
/* ------------------------------------------------------------------ */
interface ImageMeta { width: number; height: number }

function ZoomImage({ src, alt, onMeta }: { src: string; alt: string; onMeta?: (m: ImageMeta) => void }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const MIN = 0.25, MAX = 8, STEP = 0.5;

  function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

  function zoomTo(newZoom: number, cx = 0.5, cy = 0.5) {
    const z = clamp(newZoom, MIN, MAX);
    const ratio = z / zoom;
    setZoom(z);
    setPan((p) => {
      const rect = containerRef.current?.getBoundingClientRect();
      return rect
        ? { x: p.x * ratio + cx * rect.width * (1 - ratio), y: p.y * ratio + cy * rect.height * (1 - ratio) }
        : p;
    });
  }

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const h = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = (e.clientX - rect.left) / rect.width;
      const cy = (e.clientY - rect.top) / rect.height;
      zoomTo(zoom + (e.deltaY < 0 ? STEP : -STEP), cx, cy);
    };
    el.addEventListener("wheel", h, { passive: false });
    return () => el.removeEventListener("wheel", h);
  }, [zoom]);

  function onMouseDown(e: RMouseEvent) {
    if (zoom <= 1) return;
    setDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }

  function onMouseMove(e: RMouseEvent) {
    if (!dragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  }

  function onMouseUp() { setDragging(false); }

  function onLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    const img = e.currentTarget;
    onMeta?.({ width: img.naturalWidth, height: img.naturalHeight });
  }

  const cursor = zoom > 1 ? (dragging ? "grabbing" : "grab") : "zoom-in";
  const btn = "text-ink px-2 py-1 rounded text-base min-w-[32px] text-center cursor-pointer hover:bg-surface disabled:opacity-30 disabled:cursor-default";

  return (
    <div className="flex flex-col flex-1 min-h-0 mb-4">
      <div className="flex items-center gap-1 py-2 border-b border-hairline mb-3 shrink-0">
        <button className={btn} onClick={() => zoomTo(zoom - STEP)} disabled={zoom <= MIN}>−</button>
        <span className="font-mono text-[13px] text-slate min-w-[48px] text-center">{Math.round(zoom * 100)}%</span>
        <button className={btn} onClick={() => zoomTo(zoom + STEP)} disabled={zoom >= MAX}>+</button>
        <button className={btn} onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} disabled={zoom === 1}>Fit</button>
        <button className={btn} onClick={() => zoomTo(1)}>1:1</button>
      </div>
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden relative bg-surface rounded-lg min-h-[300px] flex items-center justify-center"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <img
          src={src}
          alt={alt}
          onLoad={onLoad}
          draggable={false}
          className="absolute origin-top-left select-none"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor }}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Search page                                                        */
/* ------------------------------------------------------------------ */
const LEVEL_MAP: Record<string, { text: string | null; cls: string }> = {
  confirmed: { text: "CONFIRMED", cls: "bg-brand-green-dark" },
  suspected: { text: "SUSPECT", cls: "bg-accent-orange" },
  none:      { text: null, cls: "" },
};

export default function Search() {
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [imageMeta, setImageMeta] = useState<ImageMeta | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const revokedRef = useRef<string | null>(null);

  const showPreview = useCallback((f: File) => {
    if (revokedRef.current) URL.revokeObjectURL(revokedRef.current);
    const url = URL.createObjectURL(f);
    revokedRef.current = url;
    setPreviewUrl(url);
  }, []);

  useEffect(() => () => { if (revokedRef.current) URL.revokeObjectURL(revokedRef.current); }, []);

  function accept(f: File | null) {
    if (!f) return;
    setFile(f);
    showPreview(f);
    setError(null);
    setResults(null);
  }

  function onDrop(e: DragEvent) { e.preventDefault(); setDragOver(false); accept(e.dataTransfer?.files?.[0] ?? null); }
  function onChange(e: ChangeEvent<HTMLInputElement>) { accept(e.target?.files?.[0] ?? null); }

  async function handleSearch() {
    if (!file) { setError("Select an image first."); return; }
    setLoading(true); setError(null); setResults(null);
    try { setResults(await searchImage(file)); } catch (e: any) { setError(e.response?.data?.detail ?? e.message); }
    finally { setLoading(false); }
  }

  const zoneCls = `border-2 rounded-xl p-8 text-center cursor-pointer transition-colors ${
    dragOver ? "border-brand-green bg-surface-feat" : file ? "border-hairline border-solid bg-white p-3" : "border-dashed border-hairline-strong bg-surface"
  }`;

  return (
    <div>
      <h2 className="text-[22px] font-medium text-ink mb-5">Search</h2>

      {/* Upload zone */}
      <section className="mb-6">
        <div
          className={zoneCls}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input type="file" accept="image/*" ref={fileRef} onChange={onChange} hidden />

          {previewUrl ? (
            <div className="flex flex-col items-center gap-2">
              <img src={previewUrl} alt="Preview" className="max-w-[260px] max-h-[200px] object-contain rounded-lg border border-hairline" />
              <span className="text-[13px] font-medium text-slate">{file?.name}</span>
            </div>
          ) : (
            <div className="pointer-events-none">
              <div className="text-[32px] text-muted mb-3 leading-none">↑</div>
              <p className="text-base text-slate">Drop an image here, or <span className="text-brand-green-dark font-medium">browse</span></p>
              <p className="text-[13px] text-stone mt-1">PNG, JPEG, WebP, BMP — any size</p>
            </div>
          )}
        </div>

        <div className="flex items-center gap-4 mt-4">
          <button className="bg-brand-green text-brand-teal-deep px-[22px] py-2.5 rounded-full text-sm font-semibold cursor-pointer disabled:opacity-50" onClick={handleSearch} disabled={loading || !file}>
            {loading ? "Searching..." : "Search"}
          </button>
          {file && <span className="text-[13px] text-stone">{file.name} ({(file.size / 1024).toFixed(0)} KB)</span>}
        </div>
      </section>

      {error && <div className="bg-danger-soft border border-red-200 text-danger px-4 py-2.5 rounded-lg my-3 text-sm">{error}</div>}

      {/* Results */}
      {results && (
        <div>
          <p className="text-sm text-slate mb-4 flex flex-wrap items-center gap-3">
            <span>{results.count} result{results.count !== 1 ? "s" : ""} in {results.query_time_ms}ms</span>
            {results.stages && (
              <span className="inline-flex flex-wrap gap-1 items-center">
                {(["faiss","dhash","phash","ssim"] as const).map((name, i, arr) => {
                  const s = results.stages[name];
                  if (!s) return null;
                  return [
                    <span key={name} className="inline-block bg-surface border border-hairline px-2 py-0.5 rounded font-mono text-xs text-slate">
                      {name}: {s.out}/{s.in} <small className="text-muted">{s.elapsed_ms}ms</small>
                    </span>,
                    i < arr.length - 1 && results.stages[arr[i + 1]] ? <span key={`s${i}`} className="text-muted">→</span> : null,
                  ];
                })}
              </span>
            )}
          </p>

          {results.results.length === 0 ? (
            <p className="text-muted mt-3 text-sm">No results. The index may be empty, or no images matched your filters.</p>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
              {results.results.map((r) => {
                const lv = LEVEL_MAP[r.match_level];
                return (
                  <div
                    key={r.id}
                    className={`bg-white border rounded-xl overflow-hidden cursor-pointer transition ${
                      r.match_level === "confirmed" ? "border-brand-green-mid shadow-sm" :
                      r.match_level === "suspected" ? "border-accent-orange" : "border-hairline"
                    }`}
                    onClick={() => { setSelected(r); setImageMeta(null); }}
                  >
                    <div className="relative aspect-square bg-surface">
                      <img src={imageUrl(r.filename)} alt={r.filename} loading="lazy" className="w-full h-full object-contain block" />
                      {lv.text && (
                        <span className={`absolute top-1 right-1 text-[11px] font-semibold tracking-wide px-2 py-0.5 rounded text-white ${lv.cls}`}>
                          {lv.text}
                        </span>
                      )}
                    </div>
                    <div className="px-2 py-1.5">
                      <span className="block text-xs font-medium text-ink truncate" title={r.filename}>{r.filename}</span>
                      <span className="block text-[11px] text-steel mt-0.5">L2: {r.distance.toFixed(4)}</span>
                      <span className="block text-[11px] text-steel">dHash: {r.dhash_distance} | pHash: {r.phash_distance} | SSIM: {r.ssim_score.toFixed(3)} | ✓{r.stages_passed}/3</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Fullscreen preview */}
      {selected && (
        <div
          className="fixed inset-0 bg-brand-teal-deep/60 flex items-center justify-center z-100"
          onClick={() => setSelected(null)}
          onKeyDown={(e) => { if (e.key === "Escape") setSelected(null); }}
        >
          <div className="bg-white rounded-xl p-6 w-[min(95vw,1100px)] max-h-[92vh] flex flex-col shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-0">
              <strong className="text-sm text-ink">{selected.filename}</strong>
              <button className="text-2xl text-steel hover:bg-surface px-2 py-0.5 rounded cursor-pointer leading-none" onClick={() => setSelected(null)}>×</button>
            </div>
            <ZoomImage src={imageUrl(selected.filename)} alt={selected.filename} onMeta={setImageMeta} />
            <table className="text-[13px] border-none mt-3">
              <tbody>
                {imageMeta && <tr><td className="text-stone font-normal pr-3 py-1">Dimensions</td><td className="py-1">{imageMeta.width} × {imageMeta.height} px</td></tr>}
                {selected.path && <tr><td className="text-stone font-normal pr-3 py-1">Path</td><td className="font-mono text-xs break-all py-1">{selected.path}</td></tr>}
                {selected.dhash_hex && <tr><td className="text-stone font-normal pr-3 py-1">dHash hex</td><td className="font-mono text-xs py-1">{selected.dhash_hex}</td></tr>}
                {selected.phash_hex && <tr><td className="text-stone font-normal pr-3 py-1">pHash hex</td><td className="font-mono text-xs py-1">{selected.phash_hex}</td></tr>}
                <tr><td className="text-stone font-normal pr-3 py-1">Match</td><td className="py-1">
                  <span className={`inline-block text-[13px] font-semibold px-2 py-0.5 rounded-sm text-white ${selected.match_level === "confirmed" ? "bg-brand-green-dark" : selected.match_level === "suspected" ? "bg-accent-orange" : ""}`}>
                    {selected.match_level}
                  </span>
                </td></tr>
                <tr><td className="text-stone font-normal pr-3 py-1">L2 Distance</td><td className="py-1">{selected.distance.toFixed(6)}</td></tr>
                <tr><td className="text-stone font-normal pr-3 py-1">dHash</td><td className="py-1">{selected.dhash_distance} / 64</td></tr>
                <tr><td className="text-stone font-normal pr-3 py-1">pHash</td><td className="py-1">{selected.phash_distance} / 64</td></tr>
                <tr><td className="text-stone font-normal pr-3 py-1">SSIM</td><td className="py-1">{selected.ssim_score.toFixed(4)}</td></tr>
                <tr><td className="text-stone font-normal pr-3 py-1">Filters passed</td><td className="py-1">{selected.stages_passed} / 3</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
