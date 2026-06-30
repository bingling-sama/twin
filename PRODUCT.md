# Product

## Register

product

## Users

Developers and ML engineers building or testing image similarity search pipelines. The primary user is the developer running the system locally — they upload images, tune thresholds, inspect funnel metrics, and verify that the two-stage retrieval pipeline produces correct results. The web frontend is a verification tool for the API, not a polished end-user product.

Context: typically used at a desk, with the server running on the same machine or a local network. The user switches between the CLI (curl, pytest) and the browser UI for visual inspection.

## Product Purpose

Twin is an image similarity search and deduplication system built around a CLIP + Faiss two-stage pipeline with funnel-based perceptual verification (dHash → pHash → SSIM). It exists to:

1. Provide a fast, accurate way to find duplicate or near-duplicate images in large collections (10K–1M images)
2. Expose every step of the pipeline — embedding time, per-stage survivor counts, metric scores — so the user can tune thresholds with real data
3. Serve as a reference implementation for the two-stage retrieval pattern that can be adapted to other domains

Success is measured by: search results return in under 100ms, confirmed duplicates have zero false positives, and the funnel visualization makes it obvious which stage eliminated which candidates.

## Brand Personality

**Precision · Efficient · Transparent**

- **Precision**: Every number on screen is accurate. L2 distances, Hamming distances, SSIM scores — no rounding that hides information. The tool earns trust through exactness.
- **Efficient**: No animations that delay information. No "loading experience" that isn't a real progress indicator. Every click produces a result or an error message that explains what happened.
- **Transparent**: The funnel is visible. The user can see that 50 candidates entered dHash, 12 survived, 5 passed pHash, and 3 were confirmed by SSIM — with per-stage timing. Nothing is a black box.

Voice: direct, technical, no marketing. Button labels say exactly what they do ("Index", "Search", "Clear All"). Error messages name the thing that failed and suggest a fix when possible.

## Anti-references

- **No SaaS marketing aesthetic**: no hero banners, no gradient backgrounds, no oversized CTAs, no "Start your free trial" energy. This is a tool, not a landing page.
- **No enterprise dashboard bloat**: no multi-level sidebar navigation, no dashboard widgets with sparklines, no "Welcome back, User" headers. Every UI element must earn its space.
- **No black-box "magic"**: search results must never feel like a mystery. If something was rejected, the user can see why.

## Design Principles

1. **Show your work** — Every pipeline stage reports its input count, output count, and elapsed time. The user can trace a result from Faiss recall through to SSIM confirmation.
2. **Be fast, not fancy** — Performance is a feature. No decorative motion. Transitions exist only to prevent layout shift or flash-of-content. If an animation doesn't help the user understand state change, it doesn't belong.
3. **Exact numbers, not happy signals** — Show raw scores (L2 distance, Hamming distance, SSIM). The user decides what "close enough" means. The tool reports, it doesn't judge.
4. **One thing per screen** — Each tab (Health, Search, Browse, Index) does one job. No nested workflows, no wizards, no multi-step forms hidden behind modals.

## Accessibility & Inclusion

- Target WCAG 2.1 Level AA
- All text must meet 4.5:1 contrast ratio against backgrounds
- Reduced motion: all transitions disabled via `prefers-reduced-motion`
- Keyboard navigable: tab order follows visual order, focus indicators visible
- Error messages include both visual indicator (color) and text description
