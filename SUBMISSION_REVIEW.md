# Submission review

Reviewed on 24 October 2026.

**Score: 6.8/10. Verdict: hold final submission.**

The local interface, text analysis, request validation, and development image flow work in the checks below. The live website serves the expected static preview. Real SD-Turbo generation did not finish within the four-minute review limit, so generation readiness remains unverified. The image evaluator also accepts incompatible classifier outputs in a way that can misreport emotion recovery.

The project is a readable prototype with a clear pipeline. It is not yet a validated demonstration of emotion control. The README correctly says that the current image-classifier evaluation is unmeasured.

## Review scope

The review covered all 56 tracked files: 36 Python files, the browser code, tests, scripts, configuration, README, citation metadata, and license. All text source was read. The three pipeline SVGs were parsed and their wording checked against the implementation. The walkthrough's format, duration, and captions were checked; every video frame was not independently reviewed.

No application source was changed during this review. This report records the current behavior and the remaining work.

## Verification

| Check | Result |
|---|---|
| Maintained Python suite | 7 tests passed. |
| Maintained JavaScript suite | 1 test passed. |
| Ruff | Passed error, import, upgrade, and bug checks with Python 3.11 as the target. |
| Real text analysis | HTTP 200 with all seven emotion scores, using the cached pretrained model. |
| Real image generation | Inconclusive. A 512 × 512, two-step SD-Turbo request on Apple MPS exceeded 240 seconds, including model loading, and was stopped. No completed image was obtained. |
| Local browser flow | Passed analysis, generation, image/report downloads, variations, four-image history, history restoration, and invalid-seed handling using the null backend and a synthetic analyzer. This verifies the interface, not diffusion quality. |
| Browser layout | Chrome at 1440 px and 390 px showed no horizontal overflow. No uncaught JavaScript exceptions were observed. |
| Live website | Checked page, script, stylesheet, video page, and captions returned HTTP 200 and matched repository content. Generation controls were disabled with the static-preview notice visible. |
| README and pipeline | Local links and page anchors resolved. Diagram stages match the code and distinguish heuristic text affect from classifier-label priors. |
| Exact paired binary test | Matched exhaustive paired randomization for all 5,461 binary datasets with zero through six pairs. The two-pair counterexample returns p = .5. |
| Checkpoint recovery | Retained complete records, recovered incomplete final JSON and UTF-8, preserved a complete final record without a newline, and rejected earlier corruption. |
| Seed construction | No collisions in the checked domain of 1,030 subjects × 7 emotions × 13 seeds: 93,730 seeds. |
| Historical regression checks | Sentence/clause negation boundaries, incompatible text-classifier labels, nonfinite values, and deeply nested request data behaved as intended. These targeted probes supplement the maintained tests. |
| Security checks | The same-day dependency audit reported no known vulnerabilities among 94 checked packages. File and local-history secret scans reported no findings. Source review confirmed public-mode authentication, host checks, request limits, and a hosted-generation deadline. This is not a penetration test. |

The runtime used Python 3.12.13, PyTorch 2.14.1, and Pillow 13.0.0. Real inference used cached models offline. The generator revision was `b261bac6fd2cf515557d5d0707481eafa0485ec2`; the text-classifier revision was `0e1cd914e3d46199ed785853e12b57304e04178b`.

Docker was unavailable. Hosted inference, CUDA, Windows, Safari, and physical mobile devices were not exercised. A browser phone-width check does not verify a physical phone. No new recovery accuracy, human-rating result, or performance benchmark was produced.

## Remaining findings

### High: incompatible image-classifier labels can change the reported result

[`HFImageClassifierProbe`](novavision/eval/probes.py) accepts a classifier when any output label overlaps the project's taxonomy. Recovery then drops unsupported labels before selecting the largest remaining score.

A verified synthetic classifier with `POSITIVE: 0.99` and `joy: 0.01` was accepted and reported as `joy`, with valence 0.8 and arousal 0.7. The strongest prediction disappeared. This affects the separate image evaluator; the text analyzer rejects an incompatible sentiment taxonomy correctly.

Require every output label to map unambiguously to supported emotions. Accept canonical labels directly and require an explicit mapping otherwise. Reject unsupported outputs rather than silently dropping them. If a study deliberately uses fewer emotion classes, declare that coverage and restrict the evaluation accordingly. Expose required mappings through the experiment and validation commands.

### Medium: real generation still needs a completed readiness check

The diffusion test loaded its pipeline components but did not return an image before the review deadline. This does not establish a code defect, and it does not establish acceptable generation latency. Check memory use and progress on the intended submission machine, complete a cold request and a repeat request, and retain the actual settings and timings.

### Medium: a documented script command fails in a fresh checkout

The header in [`affect_lexicon.tsv`](novavision/data/affect_lexicon.tsv) recommends `python scripts/download_lexicon.py`. Direct execution cannot import `novavision` in a clean checkout without an incidental import-path adjustment. Packaging metadata is absent.

The equivalent `python -m scripts.download_lexicon` works from the repository root. All seven project-importing script entry points passed their module-style help commands; their direct-file forms failed the isolated import check. Use module-style commands consistently.

### Research gaps: provenance and uncertainty

[`validate_probe.py`](novavision/eval/validate_probe.py) and [`build_benchmark.py`](novavision/data/build_benchmark.py) can retain a missing or mutable dataset revision. Resolve and save the dataset commit, as the model-resolution code already does for models. Validation code records decoded-image hashes; retain them.

[`experiments/run.py`](novavision/experiments/run.py) reuses subjects across emotions and seeds while calculating uncertainty over individual images or pairs. An inference about new subjects needs subject-level resampling or another suitable clustered analysis. Otherwise, limit conclusions to the fixed prompts and sampled settings. The exact paired binary calculation itself passed the exhaustive numerical check.

### Low: citation metadata is stale

[`CITATION.cff`](CITATION.cff) mentions archived pilot charts and tables that are no longer included. Its abstract should describe the current unmeasured evaluator.

## Score

This is a review judgment about submission readiness, not model performance.

| Area | Score | Reason |
|---|---:|---|
| Application behavior | 19/25 | Browser flow and real analysis pass; completed diffusion generation remains unverified. |
| Code organization | 16/20 | Focused modules, clear names, input checks, and bounded hosted requests; image-label handling needs correction. |
| Evaluation methods | 12/20 | Exact paired tests, checkpoint recovery, and study identity safeguards are useful; taxonomy and sampling scope need attention. |
| Evidence and reproducibility | 4/15 | No current measured recovery results; dataset commits and tested dependency constraints are incomplete. |
| Documentation and interface | 9/10 | Clear purpose, honest limitations, and a consistent pipeline; one broken command and stale citation remain. |
| Security and verification | 8/10 | Authentication and request controls are present; scans and focused checks passed within the stated scope. |
| **Total** | **68/100** | **6.8/10** |

## Five improvements

1. **Correct image-label validation.** Validate every classifier output and explicit mapping before experiments. Keep regression assertions for unsupported, duplicate, and nonfinite predictions in the existing evaluation tests.
2. **Complete real generation checks.** Finish cold and repeat SD-Turbo requests on the target machine, then test a complete browser session with that backend. Report actual timings and failures.
3. **Measure the current evaluator.** Validate the chosen classifier on labeled images, then run paired prompt tiers with recorded settings. Publish traceable CSV measurements and plots. Use human ratings for claims about human perception.
4. **Make runs reproducible.** Record immutable dataset commits, preserve tested dependency constraints, and retain model revisions and input fingerprints in submission exports. Match uncertainty calculations to the subject sampling design.
5. **Finish release cleanup.** Replace direct script commands with module commands, update the citation abstract, and preserve the important failure checks within the existing test files.

## Final verdict

The repository is suitable for presenting the prototype and its design with these limitations stated. Hold a final submission that promises working image generation or measured emotion controllability until a real generation request completes and the image-label defect is fixed. No current evidence supports a stronger result claim.
