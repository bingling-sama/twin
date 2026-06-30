import { useEffect, useState } from "react";
import { listIndex, imageUrl, type IndexedItem, type IndexListResponse } from "../api/twin";

const PAGE_SIZE = 50;

export default function Browse() {
  const [data, setData] = useState<IndexListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<IndexedItem | null>(null);

  async function load(p: number) {
    try { setError(null); setData(await listIndex(p, PAGE_SIZE)); setPage(p); }
    catch (e: any) { setError(e.message); }
  }

  useEffect(() => { load(1); }, []);

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 0;

  const btn = "text-ink px-3 py-1 rounded-lg text-xs font-semibold cursor-pointer hover:bg-surface disabled:opacity-40 disabled:cursor-default";

  return (
    <div>
      <h2 className="text-[22px] font-medium text-ink mb-5">Browse</h2>
      {error && <div className="bg-danger-soft border border-red-200 text-danger px-4 py-2.5 rounded-lg my-3 text-sm">{error}</div>}

      {data && (
        <>
          <div className="flex justify-between items-center mb-4 text-sm text-slate">
            <span>{data.total.toLocaleString()} images</span>
            <div className="flex items-center gap-3 text-[13px] text-slate">
              <button className={btn} disabled={page <= 1} onClick={() => load(page - 1)}>← Prev</button>
              <span>Page {page} / {totalPages}</span>
              <button className={btn} disabled={page >= totalPages} onClick={() => load(page + 1)}>Next →</button>
            </div>
          </div>

          {selected && (
            <div className="fixed inset-0 bg-brand-teal-deep/60 flex items-center justify-center z-100" onClick={() => setSelected(null)}>
              <div className="bg-white rounded-xl p-6 max-w-[520px] max-h-[85vh] overflow-y-auto text-center shadow-xl" onClick={(e) => e.stopPropagation()}>
                <img src={imageUrl(selected.filename)} alt={selected.filename} className="max-w-full max-h-[420px] object-contain rounded" />
                <div className="mt-4 text-left text-sm text-slate">
                  <p className="text-ink font-medium">{selected.filename}</p>
                  <p className="text-xs text-stone mt-1">ID: {selected.id} | dHash: {selected.dhash}</p>
                  {selected.path && <p className="text-xs text-stone">Path: {selected.path}</p>}
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-[repeat(auto-fill,minmax(130px,1fr))] gap-3">
            {data.items.map((item) => (
              <div key={item.id} className="bg-white border border-hairline rounded-xl overflow-hidden cursor-pointer hover:border-brand-green-mid hover:shadow-md transition" onClick={() => setSelected(item)}>
                <img src={imageUrl(item.filename)} alt={item.filename} loading="lazy" className="w-full aspect-square object-contain bg-surface block" />
                <span className="block px-2 py-1 text-xs text-steel truncate">{item.filename}</span>
              </div>
            ))}
          </div>

          {data.items.length === 0 && <p className="text-muted mt-3 text-sm">No images indexed yet.</p>}

          <div className="flex justify-center items-center gap-3 mt-5 text-[13px] text-slate">
            <button className={btn} disabled={page <= 1} onClick={() => load(page - 1)}>← Prev</button>
            <span>Page {page} / {totalPages}</span>
            <button className={btn} disabled={page >= totalPages} onClick={() => load(page + 1)}>Next →</button>
          </div>
        </>
      )}
    </div>
  );
}
