# deepseek-tech-notes

📖 **[在线阅读 · Read online（mdBook）](https://fooSynaptic.github.io/deepseek-tech-notes/)** — 与本地 IDE Preview 渲染一致；在线请用 Pages，**勿用 GitHub blob 预览**。

> **Smooth, deep notes on frontier DeepSeek tech** — bidirectional navigation from V1 through **V4** (DSA, MoE, DSpark, Engram, and beyond): formulas, walkthroughs, and infra patches in wiki + book form. Unofficial; not affiliated with DeepSeek.
>
> **丝滑阅读 × 深度拆解 × 前沿跟进** — 非官方 DeepSeek 技术笔记（V1→V4）：双向引用 wiki + 成书读本，机制、公式与推理 infra 逐级展开。

> **Primary language: Chinese.** Most articles, the book layout, and deep-dive notes in this repo are written in **简体中文**, aimed at the **Chinese-speaking community**.  
> **→ [中文导读 · 在线成书](https://fooSynaptic.github.io/deepseek-tech-notes/)** · [源稿 docs/README.md](docs/README.md)

### Recommended reading

These notes form a **bidirectional wiki** — every article links back at the top and forward in the body. To get the most out of that navigation, use one of these (not GitHub’s in-repo blob preview):

| Mode | When | Navigation |
|------|------|------------|
| **IDE Preview** (VS Code / Cursor) | Cloned repo, deep reading or editing | Click `←` back-links and in-text links; split preview or preview history — **best for forward / back references** |
| **[GitHub Pages (mdBook)](https://fooSynaptic.github.io/deepseek-tech-notes/)** | Online, no clone | Same math/diagram rendering as the IDE; use the browser **Back / Forward** buttons to retrace your reading path |

**Either IDE Preview or Pages works.** Edit and PR in `docs/` as usual.

### Why an online book (not GitHub Preview)?

**Local IDE Preview** (VS Code / Cursor) and **GitHub’s in-repo `.md` Preview** use different Markdown + math renderers — blockquotes, `$...$` / `$$...$$`, and inline math inside links often look wrong on GitHub even when they look fine in the IDE. I **do not change source Markdown** to chase GitHub Preview; instead, `docs/` is built into an **[mdBook site on GitHub Pages](https://fooSynaptic.github.io/deepseek-tech-notes/)** (KaTeX, same `$...$` source as the IDE). Use that for online reading; use the repo for editing and PRs.

**为何单独建在线 Pages？** 本地 **IDE Preview** 与 **GitHub 仓库内 Markdown 预览** 的渲染引擎不同——引用块、行内/块级公式、链接里的 `$...$` 等在 GitHub 上常会错位，在 IDE 里却正常。源稿 `.md` **不为迁就 GitHub Preview 而改写法**；改为用 mdBook + KaTeX 部署 **[在线成书](https://fooSynaptic.github.io/deepseek-tech-notes/)**，与 IDE 阅读体验对齐。在线请点 Pages；改稿、提 PR 仍走本仓库。**

---

## What this repo is

I follow DeepSeek's open-model line **V1 → V2 → V3 → R1 / V3.2 → V4**, and unpack **most** (not every) major technical reports into readable walkthroughs: architecture changes, training/inference tricks, formulas, and how versions relate.

Coverage includes:

- **Core DeepSeek releases** — MLA, MoE routing, MTP, DSA, CSA/HCA, mHC, Hash MoE, V4 KV layout, etc.
- **V4 inference stack** — **[DSpark](docs/versions/dspark-speculative-decoding.md)** speculative decoding, HiSparse, disk prefix cache.
- **Adjacent infra work** layered on DeepSeek checkpoints — **[Index Share / IndexCache](docs/versions/index-share.md)** (Tsinghua + Zhipu) and **[ESS](docs/versions/ess-latent-cache-offload.md)** latent-cache offload (Baidu BaiGe), with a dedicated **infrastructure** thread alongside algorithm and MoE.

Organized as wiki-style articles, SVG diagrams, and a book-style layout under [《ds-技术报告》/](《ds-技术报告》/01-总览/01-版本演进总览.md). For full navigation and article list, use the **[Chinese docs home](docs/README.md)** or the **[online mdBook](https://fooSynaptic.github.io/deepseek-tech-notes/)**.

### Why reading here feels smooth

I built this repo for **bidirectional navigation**: every article, deep-dive, and Q&A page links **back** to where you came from — the Chinese home, the English homepage, the evolution hub, or the parent section. Follow a link into DSA logic, MTP fusion, or Engram notes; when you are done, one click returns you to the article or index you started from. No dead ends, no guessing how to resume the thread.

In **[IDE Preview](#recommended-reading)**, click links to jump; on **[Pages](https://fooSynaptic.github.io/deepseek-tech-notes/)**, use browser Back / Forward for the same effect.

> **Work in progress.** Summaries, mirroring, links, and diagrams are still being updated. Prefer arXiv / official PDFs cited at the top of each article. Broken links or errors — **issues welcome**.

---

## Start here

| | |
|--|--|
| **Online book（Pages）** | **[fooSynaptic.github.io/deepseek-tech-notes](https://fooSynaptic.github.io/deepseek-tech-notes/)** — or clone and use **IDE Preview** |
| **Chinese home（源稿）** | [docs/README.md](docs/README.md) |
| **Evolution hub** | [Version lineage overview](docs/reports/deepseek-version-lineage-20260625.md) — algorithm / infrastructure / MoE threads |
| **Book mirror (repo)** | [《ds-技术报告》/01-总览/01-版本演进总览.md](《ds-技术报告》/01-总览/01-版本演进总览.md) |
| **PNG figures** | [`png/`](png/) — raster exports of all SVG diagrams (images only) |

<img src="./diagrams/deepseek-version-lineage.svg" alt="DeepSeek version timeline: V3–V4 algorithm evolution and Index Share / ESS / DSpark / HiSparse infra patches" width="920"/>

[Diagram details](./diagrams/deepseek-version-lineage.svg)

<img src="./diagrams/mtp-fusion-scheme.svg" alt="MTP fusion: one main-network forward per step; MTP chain supplies draft tokens without K full forwards" width="920"/>

[Diagram details](./diagrams/mtp-fusion-scheme.svg) · [DSpark speculative decoding](docs/versions/dspark-speculative-decoding.md) · [MTP fusion scheme (qa)](docs/versions/qa/mtp-fusion-scheme.md)

---

## Repository layout

| Path | Role |
|------|------|
| [`docs/`](docs/) | **Source of truth** — edit articles here |
| [`《ds-技术报告》/`](《ds-技术报告》/) | **Book mirror** — generated by `build_book.py` (do not hand-edit) |
| [`book.toml`](book.toml) + [`theme/`](theme/) | mdBook config & CSS for GitHub Pages |
| [`scripts/build_pages.sh`](scripts/build_pages.sh) | `build_book` → `SUMMARY.md` → `mdbook build` |
| [`.github/workflows/pages.yml`](.github/workflows/pages.yml) | Deploy mdBook to GitHub Pages on push to `main` |

**Reading:** [Recommended reading](#recommended-reading) — **IDE Preview** or **[GitHub Pages mdBook](https://fooSynaptic.github.io/deepseek-tech-notes/)**; not GitHub blob preview. See [Why an online book](#why-an-online-book-not-github-preview) for rendering details.

---

## Contributing & book layout

**Source of truth is `docs/`.** The folder [《ds-技术报告》/](《ds-技术报告》/) is a **generated book mirror** — do not edit those Markdown files by hand; they are overwritten by `build_book.py`.

When you add or move content:

1. **Write the article** under `docs/` (e.g. `docs/versions/`, `docs/dsa/`, `docs/reports/`, `docs/versions/qa/`).
2. **Register it for the book** in [`《ds-技术报告》/build_book.py`](《ds-技术报告》/build_book.py):
   - `CHAPTER_MAP` — map `docs/...` → book chapter path;
   - `READING_ORDER` — prev/next chapter navigation;
   - `QA_DESTINATIONS` — if it is a Q&A page (may mirror to multiple book folders);
   - `ASSET_MAP` — only if new figures need copying into the book tree.
3. **Add navigation** — blockquote top bar with `←` links back to the parent section / index (see existing articles); link the new page from the relevant overview or index.
4. **Rebuild & check** (from repo root):

```bash
python3 《ds-技术报告》/build_book.py
python3 scripts/validate_refs.py
python3 scripts/validate_backlinks.py
```

Or run the full gate: `bash scripts/doc_series_gate.sh`.

To preview the **mdBook site** locally (requires [mdBook](https://rust-lang.github.io/mdBook/)):

```bash
bash scripts/build_pages.sh
# open mdbook-out/index.html
```

Wiki-style reading in `docs/` works without the book step; run `build_book.py` when the chapter should appear in [《ds-技术报告》](《ds-技术报告》/01-总览/01-版本演进总览.md) with rewritten links and chapter nav. Push to `main` rebuilds GitHub Pages automatically.

---

## License

| Scope | License |
|-------|---------|
| Notes, diagrams, book layout | [CC BY 4.0](LICENSE) |
| `scripts/` | [MIT](LICENSE-MIT) |
| `docs/engram/` | [Apache 2.0](docs/engram/LICENSE) |
| `docs/material/` mirrors | upstream / original paper terms |

DeepSeek papers, weights, and official code remain under **their own** licenses.
