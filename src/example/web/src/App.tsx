import { useState, type FC } from "react";
import Search from "./pages/Search";
import Browse from "./pages/Browse";
import Manage from "./pages/Manage";

interface Tab {
  key: string;
  label: string;
  component: FC;
}

const TABS: Tab[] = [
  { key: "search", label: "Search", component: Search },
  { key: "browse", label: "Browse", component: Browse },
  { key: "manage", label: "Manage", component: Manage },
];

export default function App() {
  const [tab, setTab] = useState("search");
  const Active = TABS.find((t) => t.key === tab)?.component;

  return (
    <>
      <header className="sticky top-0 z-50 bg-white border-b border-hairline">
        <div className="max-w-[1280px] mx-auto px-8 py-4">
          <h1 className="text-[22px] font-medium text-ink">
            Twin <span className="text-brand-green-dark">·</span> Image Search
          </h1>
          <nav className="flex gap-0 mt-3">
            {TABS.map((t) => (
              <button
                key={t.key}
                className={`
                  px-4 py-3 text-sm font-medium border-b-2 transition-colors cursor-pointer
                  ${tab === t.key
                    ? "text-brand-green-dark border-brand-green-dark"
                    : "text-steel border-transparent"
                  }
                `}
                onClick={() => setTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <div className="max-w-[1280px] mx-auto px-8 mt-8">
        <main className="bg-white border border-hairline rounded-xl p-6 mb-8">
          {Active && <Active />}
        </main>
      </div>
    </>
  );
}
