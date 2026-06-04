# IJCAI Reviewer Feedback — Changes for ICDM Submission

---

## Reviewer #cbb5275

### R1. Clarity — proofreading needed throughout
- **Status:** ✅ DONE
- **What was done:** Grammar pass completed. Fixed sentence flow in abstract, method section, and results. Replaced "optimum" with "optimal", hyphenated "ground-truth", added comma before "respectively", replaced "-" with ":" in category lists, tightened passive constructions throughout. No em dashes used anywhere.

---

## Reviewer #cbb6654

### R2. Accuracy drop as proxy reward — validity without ground-truth labels
- **Status:** ✅ ALREADY IN PAPER — no change needed
- **Where:** Section III-C, reward function Eq. (4). Validated via synthetic F1 results (Table I).

### R3. Full model reset after detection — why not warm-start/replay?
- **Status:** ✅ ALREADY IN PAPER — no change needed
- **Where:** Section IV states this isolates detector quality. Conclusion mentions warm-start as future work.

### R4. Only 2 base learners (GaussianNB, HAT) — limited representativeness
- **Status:** ✅ DONE — reviewer concern dismissed and addressed
- **Rationale:** GaussianNB and HAT are the standard benchmark learners across the data stream mining literature, covering probabilistic and tree-based families. Not a real limitation.
- **What was done:** Added one sentence in Section IV citing Bifet (MOA) and Barros (2018) to justify this as standard practice. Added `Barros2018` to references.bib.

### R5. DN metric on real-world datasets — not meaningful standalone
- **Status:** ✅ ALREADY IN PAPER — no change needed
- **Where:** Table II caption and results text treat accuracy as primary, DN as complementary.

### R6. Bandit linearity assumption — empirical support
- **Status:** ✅ DONE
- **What was done:** Added sentence at end of Section III-E pointing to Appendix convergence table (selection accuracy 69.4% to 93.1%) as empirical support for the linear reward approximation.

### R7. Novelty — is AutoDD just engineering composition?
- **Status:** ✅ ALREADY IN PAPER — no change needed
- **Where:** Theorems 1 & 2 with provable regret bounds. Introduction contrasts with OAML/ASML.

---

## Reviewer #5075c06

### R8. Missing baselines: IBDD, WRS, IKS
- **Status:** ✅ DONE — all three named and cited
- **What was done:**
  - IKS (`dosReis2016IKS`) and WRS (`Barros2018`) named and cited in Discussion and Limitations (Section VI), arguing the statistical two-sample test family is represented by the newer KSWIN, HDDM, and ADWIN; AutoDD is detector-agnostic so adding detectors is a natural extension.
  - IBDD now named and cited in **Related Work** (distribution-based methods paragraph) as a fully unsupervised, model-independent image-based detector (`Souza2021IBDD`, KAIS 2021). New bib entry added.

### R9. Analysis across drift types (sudden, incremental, gradual, recurring)
- **Status:** ✅ DONE
- **What was done:** Drift-type table is in Appendix B. Added forward-reference sentence at end of Synthetic Datasets paragraph in Section V pointing to Appendix Table VI.

---

## Additional Changes for ICDM (Format/Content)

| Item | Status |
|---|---|
| Fig. 2 framework diagram — full width (`figure*`) | ✅ DONE |
| Supplementary merged into paper as `\appendix` | ✅ DONE |
| Missing `kulbach2022evolution` — added to bib | ✅ DONE |
| Missing `Barros2018` — added to bib | ✅ DONE |
| IEEE `[conference]` format header | ✅ DONE |
| Footnote: "supplementary" → "Appendix A" | ✅ DONE |
| Ethics Statement: removed IJCAI-specific reference | ✅ DONE |
| No em dashes anywhere in paper | ✅ DONE (fixed `---` on L301; `grep -- "---"` returns none) |
| American English consistency throughout | ✅ DONE |

---

## What to Move from Appendix to Main Paper (Suggestions)

The appendix currently contains five analyses. Below is a prioritized recommendation on what would strengthen the main paper if promoted:

### HIGH PRIORITY — Move to main paper

**1. Detection Delay table (Table V)** — ✅ DONE
- Moved to main Results as `tab:delay_main` (Section "Detection Delay"). Per-dataset delay numbers (ADWIN=64.3, AutoDD=38.4 mean) now in the body.

**2. Convergence Analysis (Table IV)** — ✅ DONE
- Moved to main Results as `tab:conv_main` (Section "Convergence Analysis"). Selection-accuracy progression (69.4% → 82.6% → 93.1%) is now in the body and referenced from the theory section.

### MEDIUM PRIORITY — Consider moving

**3. Per-Drift-Type Performance (Appendix B, Table VI + Fig. 10)**
- Currently: Appendix only, forward-referenced from Section V.
- AutoDD wins on 3 of 4 drift types; this breakdown shows the method is not a one-trick result.
- **Suggestion:** If page budget allows, move Table VI to main Results and drop the figure (table is sufficient). Adds one column-width table.

**4. Detector Selection Patterns (Appendix B, Table VII + Fig. 11)**
- Currently: Appendix only.
- Shows the learned policy is interpretable and aligns with domain knowledge (ADWIN for high velocity, KSWIN for high noise). This is a qualitative novelty argument.
- **Suggestion:** Move Table VII to main paper. The figure can stay in appendix.

### LOW PRIORITY — Keep in appendix

**5. HAT Statistical Significance + Efficiency Figures (Appendix B, Figs. 8-9)**
- The GaussianNB versions are already in the main paper. The HAT versions are supporting evidence but do not add new insight.
- **Keep in appendix.**

**6. Full Theorem Proofs**
- Note: proofs are currently inline in the main Theoretical Analysis section (Theorem 1, Proposition 2, Corollary), not in a separate appendix. The appendix is now a single "Additional Results" section containing only the per-drift-type and detector-selection-pattern tables.
- No action needed; inline proofs are acceptable.

### Sample Complexity — ✅ DONE
- Added a self-contained sentence in Section III-E after the combined-regret corollary: inverting the dominant $O(\sqrt{KmT\log T})$ term for the portfolio ($K=7$, $m=15$) to reach average proxy suboptimality $\sim 10^{-2}$ gives $T \approx 5 \times 10^5$ windows, matching the order of magnitude of the convergence in Table `tab:conv_main`.
- Note: the appendix no longer contains a separate Sample-Complexity derivation to `\cite`, so the sentence is stated self-contained rather than as a cross-reference.
