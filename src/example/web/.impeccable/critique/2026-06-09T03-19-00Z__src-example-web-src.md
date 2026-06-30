---
target: src/example/web/src/ -- all pages
total_score: 21
p0_count: 1
p1_count: 3
timestamp: 2026-06-09T03-19-00Z
slug: src-example-web-src
---
# Design Critique: Twin Frontend (all pages)

**Target**: src/example/web/src/ | **Date**: 2026-06-09 | **Score**: 21/40

## Heuristic Scores

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | No progress indication for long operations |
| 2 | Match Between System and Real World | 2 | Jargon without legend (SSIM, pHash, ✓2/3) |
| 3 | User Control and Freedom | 3 | Clean tab switching, confirm on destructive actions |
| 4 | Consistency and Standards | 3 | MongoDB tokens respected, minor semantic overlap |
| 5 | Error Prevention | 1 | Silent fail on no-file upload |
| 6 | Recognition Rather Than Recall | 3 | All tabs visible, results persistent |
| 7 | Flexibility and Efficiency of Use | 1 | Zero keyboard shortcuts, no drag-drop, no sort/filter |
| 8 | Aesthetic and Minimalist Design | 3 | Clean and restrained, but contrast failures |
| 9 | Help Users Recover from Errors | 2 | Server errors shown but no recovery suggestions |
| 10 | Help and Documentation | 1 | No tooltips, help, onboarding, or explanations |
| **Total** | | **21/40** | Acceptable |

## Anti-Patterns
- LLM: LOW slop. Detector: 0 findings.
- Near-miss: 11px uppercase match-tags (functional, not decorative)

## Cognitive Load
- 2 of 8 failures: progressive disclosure (jargon without tooltips), feedback (silent fail + no progress)

## Priority Issues
- P0: Silent failure on no-file action (Search/Index buttons)
- P1: No focus indicators on interactive elements
- P1: Muted/stone text colors fail WCAG AA contrast
- P1: No reduced-motion support despite PRODUCT.md commitment
- P2: No progress indication for long operations
- P2: Query preview appears too late + URL.createObjectURL memory leak
- P2: No pagination keyboard shortcuts
- P3: Batch index has no path validation

## Persona Red Flags
- Alex: No keyboard shortcuts, no drag-drop, no sort/filter, excessive rounding
- Jordan: Jargon wall, intimidating Clear All, no empty-state guidance
- Sam: Contrast failures, no focus indicators, no reduced-motion, no dark mode, no ARIA tab semantics

## Strengths
1. Pipeline transparency (stage chain with per-stage timing)
2. Restrained visual system (MongoDB tokens, zero decoration)
3. Clean IA with tab-level isolation
