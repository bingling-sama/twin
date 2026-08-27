import { useState, useRef, useEffect, useCallback, type DragEvent, type ChangeEvent, type MouseEvent as RMouseEvent } from "react";
import { searchImage, searchText, imageUrl, type SearchResult, type SearchResponse, type TextSearchResultItem } from "../api/twin";

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

interface ResultPreviewModalProps {
  filename: string;
  path?: string;
  distance: number;
  dhash_hex?: string;
  phash_hex?: string;
  match_level?: string;
  dhash_distance?: number;
  phash_distance?: number;
  ssim_score?: number;
  stages_passed?: number;
  onClose: () => void;
  onMeta: (m: ImageMeta) => void;
  imageMeta: ImageMeta | null;
}

function ResultPreviewModal({
  filename,
  path,
  distance,
  dhash_hex,
  phash_hex,
  match_level,
  dhash_distance,
  phash_distance,
  ssim_score,
  stages_passed,
  onClose,
  onMeta,
  imageMeta,
}: ResultPreviewModalProps) {
  const isImageSearch = match_level !== undefined;

  return (
    <div
      className="fixed inset-0 bg-brand-teal-deep/60 flex items-center justify-center z-100"
      onClick={onClose}
      onKeyDown={(e) => { if (e.key === "Escape") onClose(); }}
    >
      <div
        className="bg-white rounded-xl p-6 w-[min(95vw,1100px)] max-h-[92vh] flex flex-col shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-0">
          <strong className="text-sm text-ink">{filename}</strong>
          <button
            className="text-2xl text-steel hover:bg-surface px-2 py-0.5 rounded cursor-pointer leading-none"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <ZoomImage src={imageUrl(filename)} alt={filename} onMeta={onMeta} />
        <table className="text-[13px] border-none mt-3">
          <tbody>
            {imageMeta && (
              <tr>
                <td className="text-stone font-normal pr-3 py-1">Dimensions</td>
                <td className="py-1">{imageMeta.width} × {imageMeta.height} px</td>
              </tr>
            )}
            {path && (
              <tr>
                <td className="text-stone font-normal pr-3 py-1">Path</td>
                <td className="font-mono text-xs break-all py-1">{path}</td>
              </tr>
            )}
            {dhash_hex && (
              <tr>
                <td className="text-stone font-normal pr-3 py-1">dHash hex</td>
                <td className="font-mono text-xs py-1">{dhash_hex}</td>
              </tr>
            )}
            {phash_hex && (
              <tr>
                <td className="text-stone font-normal pr-3 py-1">pHash hex</td>
                <td className="font-mono text-xs py-1">{phash_hex}</td>
              </tr>
            )}
            {isImageSearch ? (
              <>
                <tr>
                  <td className="text-stone font-normal pr-3 py-1">Match</td>
                  <td className="py-1">
                    <span
                      className={`inline-block text-[13px] font-semibold px-2 py-0.5 rounded-sm text-white ${
                        match_level === "confirmed"
                          ? "bg-brand-green-dark"
                          : match_level === "suspected"
                          ? "bg-accent-orange"
                          : ""
                      }`}
                    >
                      {match_level}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td className="text-stone font-normal pr-3 py-1">L2 Distance</td>
                  <td className="py-1">{distance.toFixed(6)}</td>
                </tr>
                {dhash_distance !== undefined && (
                  <tr>
                    <td className="text-stone font-normal pr-3 py-1">dHash</td>
                    <td className="py-1">{dhash_distance} / 64</td>
                  </tr>
                )}
                {phash_distance !== undefined && (
                  <tr>
                    <td className="text-stone font-normal pr-3 py-1">pHash</td>
                    <td className="py-1">{phash_distance} / 64</td>
                  </tr>
                )}
                {ssim_score !== undefined && (
                  <tr>
                    <td className="text-stone font-normal pr-3 py-1">SSIM</td>
                    <td className="py-1">{ssim_score.toFixed(4)}</td>
                  </tr>
                )}
                {stages_passed !== undefined && (
                  <tr>
                    <td className="text-stone font-normal pr-3 py-1">Filters passed</td>
                    <td className="py-1">{stages_passed} / 3</td>
                  </tr>
                )}
              </>
            ) : (
              <tr>
                <td className="text-stone font-normal pr-3 py-1">L2 Distance</td>
                <td className="py-1">{distance.toFixed(6)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Search() {
  const [searchMode, setSearchMode] = useState<"image" | "text">("image");
  const [textPrompt, setTextPrompt] = useState("");
  const [imageResults, setImageResults] = useState<SearchResponse | null>(null);
  const [textResults, setTextResults] = useState<TextSearchResponse | null>(null);
  const [activeResultMode, setActiveResultMode] = useState<"image" | "text">("image");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedImage, setSelectedImage] = useState<SearchResult | null>(null);
  const [selectedText, setSelectedText] = useState<TextSearchResultItem | null>(null);
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
    setImageResults(null);
    setTextResults(null);
  }

  function onDrop(e: DragEvent) { e.preventDefault(); setDragOver(false); accept(e.dataTransfer?.files?.[0] ?? null); }
  function onChange(e: ChangeEvent<HTMLInputElement>) { accept(e.target?.files?.[0] ?? null); }

  async function handleSearch() {
    if (searchMode === "image") {
      if (!file) { setError("Select an image first."); return; }
      setLoading(true); setError(null); setImageResults(null); setTextResults(null);
      try {
        const res = await searchImage(file);
        setImageResults(res);
        setActiveResultMode("image");
      } catch (e: any) {
        setError(e.response?.data?.detail ?? e.message);
      } finally {
        setLoading(false);
      }
    } else {
      if (!textPrompt.trim()) { setError("Enter a search prompt first."); return; }
      setLoading(true); setError(null); setImageResults(null); setTextResults(null);
      try {
        const res = await searchText(textPrompt.trim(), 50);
        setTextResults(res);
        setActiveResultMode("text");
      } catch (e: any) {
        setError(e.response?.data?.detail ?? e.message);
      } finally {
        setLoading(false);
      }
    }
  }

  const zoneCls = `border-2 rounded-xl p-8 text-center cursor-pointer transition-colors ${
    dragOver ? "border-brand-green bg-surface-feat" : file ? "border-hairline border-solid bg-white p-3" : "border-dashed border-hairline-strong bg-surface"
  }`;

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-[22px] font-medium text-ink">Search</h2>
        {/* Mode Toggle */}
        <div className="flex bg-surface rounded-full p-1 border border-hairline">
          <button
            className={`px-4 py-1.5 rounded-full text-xs font-semibold cursor-pointer transition ${
              searchMode === "image" ? "bg-white text-ink shadow-sm" : "text-slate hover:text-ink"
            }`}
            onClick={() => { setSearchMode("image"); setError(null); }}
          >
            Image Search
          </button>
          <button
            className={`px-4 py-1.5 rounded-full text-xs font-semibold cursor-pointer transition ${
              searchMode === "text" ? "bg-white text-ink shadow-sm" : "text-slate hover:text-ink"
            }`}
            onClick={() => { setSearchMode("text"); setError(null); }}
          >
            Text Search (CLIP)
          </button>
        </div>
      </div>

      {/* Query Form */}
      {searchMode === "image" ? (
        <div className="flex flex-col gap-3">
          <div
            className={zoneCls}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="hidden"
              onChange={onChange}
            />
            {previewUrl ? (
              <div className="flex items-center gap-3">
                <img src={previewUrl} alt="Query preview" className="w-16 h-16 object-cover rounded-lg border border-hairline" />
                <div className="text-left flex-1 min-w-0">
                  <p className="text-sm font-medium text-ink truncate">{file?.name}</p>
                  <p className="text-xs text-muted">{file ? (file.size / 1024).toFixed(1) + " KB" : ""}</p>
                </div>
                <button
                  type="button"
                  className="text-xs text-muted hover:text-ink px-2 py-1 rounded hover:bg-surface cursor-pointer"
                  onClick={(e) => { e.stopPropagation(); setFile(null); setPreviewUrl(null); }}
                >
                  Change
                </button>
              </div>
            ) : (
              <div>
                <p className="text-sm font-medium text-ink">Drop an image here or click to browse</p>
                <p className="text-xs text-muted mt-1">JPEG, PNG, WebP, BMP up to 50MB</p>
              </div>
            )}
          </div>
          <button
            className="w-full py-2.5 px-4 bg-brand-green text-ink font-semibold rounded-full hover:bg-brand-green-hover transition disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            onClick={handleSearch}
            disabled={!file || loading}
          >
            {loading ? "Searching..." : "Search Similar Images"}
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="relative">
            <input
              type="text"
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleSearch(); }}
              placeholder="e.g. A cat sitting on a laptop, red sports car, snowy mountain peak..."
              className="w-full px-4 py-3 bg-white border border-hairline rounded-xl text-sm text-ink placeholder:text-muted focus:outline-none focus:border-brand-green"
            />
          </div>
          <button
            className="w-full py-2.5 px-4 bg-brand-green text-ink font-semibold rounded-full hover:bg-brand-green-hover transition disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            onClick={handleSearch}
            disabled={!textPrompt.trim() || loading}
          >
            {loading ? "Searching..." : "Search with Text"}
          </button>
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Funnel Stage Timeline (Image Search) */}
      {activeResultMode === "image" && imageResults && (
        <div className="mt-6 mb-6">
          <div className="bg-surface rounded-xl p-4 border border-hairline">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate uppercase tracking-wider">Search Funnel Stages</span>
              <span className="text-xs font-mono text-steel">{imageResults.query_time_ms} ms total</span>
            </div>
            <div className="grid grid-cols-4 gap-2 text-center">
              {Object.entries(imageResults.stages).map(([stage, info]) => (
                <div key={stage} className="bg-white rounded-lg p-2.5 border border-hairline">
                  <div className="text-[11px] font-semibold text-slate uppercase">{stage}</div>
                  <div className="text-lg font-bold text-ink mt-0.5">{info.out}</div>
                  <div className="text-[10px] text-muted">{info.elapsed_ms}ms</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Image Search Results */}
      {activeResultMode === "image" && imageResults && (
        <div>
          <p className="text-sm text-slate mb-4">
            {imageResults.count} candidate{imageResults.count !== 1 ? "s" : ""} evaluated in {imageResults.query_time_ms}ms
          </p>
          {imageResults.results.length === 0 ? (
            <p className="text-muted mt-3 text-sm">No similar images found in the index.</p>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
              {imageResults.results.map((r) => {
                const badge = LEVEL_MAP[r.match_level];
                return (
                  <div
                    key={r.id}
                    className="bg-white border border-hairline rounded-xl overflow-hidden cursor-pointer hover:border-brand-green transition"
                    onClick={() => { setSelectedImage(r); setSelectedText(null); setImageMeta(null); }}
                  >
                    <div className="relative aspect-square bg-surface">
                      <img src={imageUrl(r.filename)} alt={r.filename} loading="lazy" className="w-full h-full object-contain block" />
                      {badge?.text && (
                        <span className={`absolute top-1.5 right-1.5 text-[10px] font-bold px-1.5 py-0.5 rounded text-white ${badge.cls}`}>
                          {badge.text}
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

      {/* Text Search Results */}
      {activeResultMode === "text" && textResults && (
        <div>
          <p className="text-sm text-slate mb-4 flex flex-wrap items-center gap-3">
            <span>{textResults.count} result{textResults.count !== 1 ? "s" : ""} for <strong className="text-ink">"{textResults.query}"</strong> in {textResults.query_time_ms}ms</span>
            <span className="inline-block bg-surface border border-hairline px-2 py-0.5 rounded font-mono text-xs text-brand-green-dark">
              CLIP Semantic Recall
            </span>
          </p>

          {textResults.results.length === 0 ? (
            <p className="text-muted mt-3 text-sm">No images matched your query.</p>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
              {textResults.results.map((r) => (
                <div
                  key={r.id}
                  className="bg-white border border-hairline rounded-xl overflow-hidden cursor-pointer hover:border-brand-green transition"
                  onClick={() => { setSelectedText(r); setSelectedImage(null); setImageMeta(null); }}
                >
                  <div className="relative aspect-square bg-surface">
                    <img src={imageUrl(r.filename)} alt={r.filename} loading="lazy" className="w-full h-full object-contain block" />
                  </div>
                  <div className="px-2 py-1.5">
                    <span className="block text-xs font-medium text-ink truncate" title={r.filename}>{r.filename}</span>
                    <span className="block text-[11px] text-steel mt-0.5">Distance: {r.distance.toFixed(4)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Result Preview Modal */}
      {selectedImage && (
        <ResultPreviewModal
          {...selectedImage}
          onClose={() => setSelectedImage(null)}
          onMeta={setImageMeta}
          imageMeta={imageMeta}
        />
      )}
      {selectedText && (
        <ResultPreviewModal
          {...selectedText}
          onClose={() => setSelectedText(null)}
          onMeta={setImageMeta}
          imageMeta={imageMeta}
        />
      )}
    </div>
  );
}
