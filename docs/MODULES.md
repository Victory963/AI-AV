# Module Reference

One section per module: what it is for, its main public API, the design decisions that are easy to get wrong, and how to run its self-test. For how the modules fit together, read [ARCHITECTURE.md](ARCHITECTURE.md) first.

Every module under `src/longfilm/` carries its own self-test:

```bash
export PYTHONPATH=$PWD/src
python -m longfilm.<module>            # e.g. python -m longfilm.router
python -m longfilm.cli selftest        # all 26 modules; prints a ✓/✗ table
python -m longfilm.cli selftest qc     # only modules whose name matches "qc"
```

Self-tests need no GPU and no API keys. They need ffmpeg (auto-detected; see [§ Tools](#tools-and-environment)) and, for anything that renders text, a CJK font (see [../assets/fonts/README.md](../assets/fonts/README.md)).

**Contents**

1. [Contract](#1-contract) — `schema`
2. [Authoring](#2-authoring) — `storyboard_gen`, `charbible`, `previz3d`, `animate`
3. [Compilation](#3-compilation) — `audio_first`, `refpack`, `prompt_os`, `chain`
4. [Execution](#4-execution) — `router`, `providers/*`, `qc`, `queue`, `cache`
5. [Post-production & delivery](#5-post-production--delivery) — `grade`, `stitch`, `upscale`, `timeline`, `compliance`
6. [Training flywheel](#6-training-flywheel) — `distill`, `lora`, `dpo`
7. [Infrastructure](#7-infrastructure) — `_proc`, `_fonts`, `cli`
8. [Scripts](#8-scripts) and [deploy/](#9-deploy)

---

## 1. Contract

### `schema.py` — the single contract (446 lines)

Pydantic v2 models that every other module reads and writes. Nothing else is shared between modules.

| Type | Purpose |
|---|---|
| `ShotSize`, `CameraMove`, `Transition`, `ContentRating`, `EngineHint` | Enums. `ShotSize`/`CameraMove` values are English film terms because they are spliced straight into prompts. `ContentRating`: `g`, `pg13`, `r_violence`, `r_suggestive`, `blocked`. |
| `ImageRef`, `VideoRef`, `AudioRef`, `RefPack` | Multimodal references. `RefPack` validates the 9 image / 3 video / 3 audio ceiling. `by_role()`, `slots_used()`. |
| `Appearance`, `VoiceProfile`, `LoRASpec`, `CharacterBible` | Character identity. `CharacterBible.is_fictional` is `Literal[True]`; `age_statement` is mandatory and validated non-empty. |
| `DialogueLine` | A line of dialogue. `start_s`/`end_s` are **relative to the shot**, never absolute. |
| `Continuity`, `Grade` | Cross-shot constraints (first/last-frame locks, screen direction, overlap) and target grade. |
| `Shot`, `Scene`, `AudioPlan`, `DeliverySpec`, `Storyboard` | The storyboard tree. `Shot.fingerprint()` hashes everything except output fields. `Storyboard.validate_continuity()`, `load()`, `save()`, `all_shots()`, `character()`. |
| `json_schema()` | JSON Schema export, used as a structured-output constraint when an LLM writes storyboards. |

Pitfall: the 180° axis check tracks direction **per character** (`last_dir: dict[str, tuple[str, str]]`). Tracking only the previous shot flags every shot/reverse-shot pair as an axis break.

---

## 2. Authoring

### `storyboard_gen.py` — beat sheet → storyboard (1829 lines)

Director knowledge written as allocation rules and a library of shot-grammar templates.

- `Beat`, `BeatFunction` — one beat and its narrative function (drives the default template and a shot-length modifier).
- `plan_beats(beats, target_duration_s=, atomic_shot_s=)` → `[BeatAllocation]` — splits total runtime into per-beat duration and shot count, **with the reason for every number** (`BeatAllocation.render()`). Intensity and density are weighted, so the cut is deliberately not uniform.
- `ShotPattern`, `PatternSlot`, `apply_pattern(...)` — reusable shot skeletons (establishing → medium → shot/reverse-shot, and so on).
- `assign_screen_directions(characters)` — one fixed screen direction per character for the whole film.
- `beat_sheet_to_scenes(...)` — one beat becomes one scene.
- `llm_prompt_for_storyboard(logline, characters, constraints)` — builds an LLM prompt with every hard constraint written as an explicit clause (`StoryboardConstraints`).
- `validate_and_repair(raw_json, ...)` — turns LLM output into a valid `Storyboard` **and reports every repair it made**.
- `pacing_report(storyboard, genre=)` — shot-length, shot-size and camera-move distributions against empirical ranges.

### `charbible.py` — character bible builder (1368 lines)

Turns a one-line character description into identity-anchor assets ready for production.

- `CharacterBibleBuilder` — fluent builder: `describe()`, `with_voice()`, `with_lora()`, `with_negative()`, `with_motion_ref()`, `age_statement()`, `build()`. `shot_list()` returns the makeup-test matrix (`View × Light × Expression × Framing`, one `PortraitSpec` per cell, each with the reason it must exist). `Tier` controls matrix size.
- `emit_portrait_prompts(character, level=)` → `[PortraitJob]` — text-to-image jobs for FLUX/SDXL batch scripts.
- `score_anchor(text)` → `AnchorScore`; `consistency_anchors(character, k=)` — picks the most distinctive appearance phrases for `prompt_os`. A phrase is *strong* only with ≥ 2 qualifier kinds (colour / shape / size), so "long hair" alone is weak.
- `inspect_portrait(uri, root=)` → `ImageQuality` — reference-image health check:
  - short side < 512 px is blocking; < 1024 px is advisory;
  - aspect ratio must be within 0.6–1.7;
  - more than 5% transparent pixels is flagged;
  - suspected UI overlays (long axis-aligned edges in the outer 25% of the frame) are flagged.
- `validate_bible(character, check_images=True, asset_root=)` — completeness check; an empty list means ready.
- `coverage_report(character, level=)` — which matrix cells are missing and which shot types they will hurt.
- `demo_cast()` — three complete fictional adult characters, all passing `validate_bible()`.

### `previz3d.py` — 3D previz (1222 lines)

Uses grey-box renders to pin down geometry for difficult shots.

- `solve_camera(shot, aspect=)` → `CameraSolve` — converts shot size / camera move / focal length into camera position, look-at and keyframes (Blender XYZ Euler via `look_at_euler`).
- `judge(shot)` → `PrevizVerdict`; `PrevizPlan.build(storyboard)` — decides which shots deserve previz, as a weighted score with reasons. `cost_benefit(shot)` gives the cost/benefit in plain language.
- `emit_blender_script(shot, storyboard, out_dir=)` — headless Blender script that renders grey / depth / normal sequences.
- `render_previz(script, out_dir)` → `PrevizRender` — **a missing Blender is a normal result, not an exception.**
- `previz_to_control(previz_dir, shot_id=)` → `ControlBundle` — packs the sequences as ControlNet inputs (`to_video_refs()`, `comfy_values()`).

### `animate.py` — motion transfer (1044 lines)

Drives a fictional character with a performance video.

- `SourceKind` — **no default**; the capture side must label where the driving footage came from. `MotionSource.validate()` rejects footage without source or licence.
- `PoseExtractor` → `DWPoseExtractor` (COCO-WholeBody 133), `OpenPoseExtractor` (BODY_25), `PrerenderedPoseSource` (the only path that runs without a GPU).
- `MethodSpec`, `atomic_seconds(method, pipeline_max_s=)` — per-method clip limits, never hard-coded at call sites.
- `motion_energy`, `smooth`, `slice_motion(motion, max_s)` → `[MotionSlice]` — cuts long motion into atomic lengths **at low-velocity points**.
- `retarget(motion, character, method=, shot=, ...)` → `AnimateJob` — ComfyUI injection values aligned with `WanAnimateToVideo` port names.

---

## 3. Compilation

### `audio_first.py` — audio-first pipeline (1106 lines)

Nail down the dialogue track first, then derive picture timing from measured durations.

- Backends (`TTSBackend`):
  - `EdgeTTSBackend` — online, no key.
  - `HttpTTSBackend` — self-hosted IndexTTS-2 / CosyVoice2 / GPT-SoVITS; field names configurable via `HttpTTSSpec`; presets `gpt_sovits()`, `openai_compatible()`.
  - `SilentTTSBackend` — offline placeholder.
  - `pick_backend(prefer)` always falls back to a working one.
- `synthesize_dialogue(storyboard, backend, out_dir, ...)` → `DialogueTrack` — synthesizes every line, **measures** its duration (`probe_duration_s`, decoded byte count, not an estimate) and writes `start_s`/`end_s` back onto each `DialogueLine`. Cached by `TTSTask.cache_key()`.
- `DialogueCue` stores both `shot_offset_s` and `abs_start_s`; `DialogueTrack.of_shot()`, `required_shot_duration_s()`.
- `retime_shots_to_audio(storyboard, track, min_s=, max_s=, ...)` — shot lengths from dialogue; splits a shot at a breath when a line does not fit. Returns a report.
- `build_audio_refs(track, storyboard, stem_dir=)` — one dialogue stem per shot, attached as `RefPack.audios` with `role="dialogue"`. The stem note begins with `source=ai_generated`.
- `estimate_duration_s` exists **only** for the silent placeholder backend.

### `refpack.py` — reference-slot orchestration (740 lines)

Ranks references by value and cuts them to each engine's budget.

- `ROLE_PRIORITY` / `VIDEO_ROLE_PRIORITY` / `AUDIO_ROLE_PRIORITY`; `role_priority(kind, role)` (unknown role → 0).
- `RefBudget.from_capabilities(caps)` — budget from the target engine's declared limits.
- `build_refpack(shot, storyboard, budget=, prev_last_frame_uri=, drops_out=)` — assembles from character bible + scene + shot, then cuts to budget.
- `fit_to_budget(images, videos, audios, budget)` → kept lists + `[DroppedRef]`. A dropped identity anchor sets `critical=True`.
- `pick_identity_anchors(char, shot_size, k)`, `preferred_views(shot_size)`, `allocate_identity_quota(subject_ids, slots)` — multi-character quota so the first character cannot take every identity slot.
- `binding_directives(pack, lang=)` — one "controls X / do not inherit Y" sentence per occupied slot; `slot_tag`, `slot_manifest` for humans.

### `prompt_os.py` — deterministic prompt compiler (639 lines)

- `compile_prompt(shot, storyboard, dialect=, include_dialogue=, negative_groups=)` → `CompiledPrompt` (frozen; `fingerprint()`, `layer(name)`). Same input → byte-identical output.
- `PromptDialect` — `wan` (long English sentences), `seedance` (short sentences + `@image` tags), `generic`.
- `camera_syntax(shot, style=)`, `audio_directive(shot, sb)`, `compile_negative(shot, sb, groups=)` with the grouped `NEGATIVE_BANK`.
- `estimate_tokens(text)`, `diff_prompts(a, b)` for A/B attribution.

### `chain.py` — long-take continuation (1135 lines)

Joins 5–15 s atomic segments into one long take without the identity collapsing.

- Strategies (`ChainStrategy`: `supported_by`, `applicable`, `prepare`), with base drift:

  | Strategy | Base drift |
  |---|---|
  | `OfficialExtendChain` | 0.18 |
  | `JobContinueChain` | 0.20 |
  | `OverlapChain` | 0.26, 6 overlap frames by default |
  | `LastFrameChain` | 0.30, the only one that works across providers |

- `DriftBudget` — explicit error budget; `estimate()`, `would_exceed()`, `spend()`, `reanchor()`, `explain()`. Threshold 1.0 is calibrated so the steadiest strategy chains exactly 4 segments before a **forced re-anchor**; the docstring explains how to recalibrate for a new engine.
- `ChainPlanner.plan(scene, storyboard, router)` → `[ChainStep]` (calls `router.plan` only; sends nothing); `LinkContext`, `LinkPlan`, `explain_chain()`.
- `pick_tail_frame(video, out_dir, max_back=)` → `TailFrame` — rejects a blurry final frame (Laplacian variance, `frame_sharpness`) and steps back. `extract_tail_frames` seeks from the end with `-sseof`.

---

## 4. Execution

### `router.py` — dual-engine routing (1242 lines)

Decisions read only `Capabilities`; execution only understands `FailureKind`.

- `RoutingPolicy(bias=, episode_budget_usd=, quotas={name: ProviderQuota(max_concurrency, max_calls, max_usd)}, rating_kinds=, unmoderated_ratings=, hero_min_tier=)`.
- `QualityBias` — `quality` / `cost` / `speed`.
- `content_gate(shot, storyboard, policy)` → `GateDecision`:
  - `blocked` is refused;
  - every on-screen character must exist in the bible with an adult statement;
  - the rating decides which engine kinds are allowed;
  - R ratings go only to self-hosted engines, and the gate never probes vendor moderation.
- `Router(registry, policy, poll_interval_s=, job_timeout_s=, backoff_base_s=)`:
  - `.plan(shot, sb)` → `RoutingPlan` (`primary`, `fallbacks`, `request`, `degradations`, `scores`, `rejected`, `explain()`). It goes content gate → hard reject → penalty scoring.
  - `.compile_request(shot, sb, provider)` → `GenRequest`.
  - `.execute(shot, sb, max_attempts=)` walks the fallback chain:
    - backoff on retryable failures;
    - immediate failover on quota, auth or moderation;
    - raise on bad request.
- `CostLedger` — records failed calls too; `ensure_affordable()` raises `BudgetExceeded` before money is spent; `by_shot()`, `by_provider()`, `report()`.

### `providers/` — engine adapters

| File | Class | Notes |
|---|---|---|
| `base.py` (313) | `VideoProvider`, `Capabilities`, `GenRequest`, `GenResult`, `JobStatus`, `FailureKind`, `ProviderError`, `ProviderRegistry`, `REGISTRY` | `FailureKind.retryable` / `.should_failover`; `Capabilities.fits_duration()`, `clamp_duration()`, `nearest_resolution()`; `REGISTRY.capable(...)` filters by need. `VideoProvider.wait()` polls to completion. |
| `ark_seedance.py` (923) | `ArkSeedanceProvider`, `classify_error`, `register_defaults` | Volcengine Ark. Model catalog: Seedance 1.0 pro/lite, 2.0 (`doubao-seedance-2-0-260128`, 4–15 s, 9 images, native audio, extend via `reference_video`). Vendor error codes → `FailureKind`. Key: `ARK_API_KEY`. |
| `dashscope_wan.py` (894) | `DashScopeWanProvider`, `WanLoRATrainingClient` | Alibaba Bailian Wanxiang 2.2 / 2.5 / 2.6 / 2.7 (i2v, kf2v). The training client runs official LoRA fine-tuning: `upload_dataset`, `create`, `checkpoints`, `deploy`. Weights stay on the platform. Key: `DASHSCOPE_API_KEY`. |
| `comfy_local.py` (703) | `ComfyProvider`, `WorkflowTemplate`, `inject`, `chain_loras`, `drop_node`, `wan_frame_count` | Self-hosted ComfyUI (`COMFY_URL`). A workflow template plus a semantic-name → node-path binding table, so exported workflows can change without code changes. Wan frame counts must be 4k+1. N LoRAs are chained; with 0 the LoRA node is bypassed. |
| `aggregator.py` (893) | `FalProvider`, `ReplicateProvider`, `failover_chain` | Third fallback. fal is the only aggregator that can load a LoRA in the cloud. Keys: `FAL_KEY`, `REPLICATE_API_TOKEN`. |
| `mock.py` (914) | `MockProvider`, `register_defaults` | Offline engine that renders **real mp4s** (zoompan camera moves, Pillow overlays), supports first-frame lock and extend, and injects deterministic faults. Three tiers: clean, jittery, always-refuses. |
| `_http.py` (217) | `request_json`, `post_multipart`, `download`, `status_to_failure`, `redact` | Minimal stdlib HTTP client. Streaming download; `redact()` masks keys before anything is logged. |

### `qc.py` — quality gate (1682 lines)

- Default metrics (`QCGate.default()`):
  - blocking: `FreezeDetect`, `BlackDetect`, `DurationFpsConformance`;
  - weighted: `TemporalFlicker`, `ColorDrift`, `MotionSanity`, `SeamCheck`.
- Optional plugins (`OptionalMetric`): `IdentityDrift` (face embeddings), `AestheticScore` (VBench aesthetic_quality protocol), `LipSyncScore` (SyncNet LSE-D/LSE-C). `register_plugin()`, `plugin_availability()`, `available_plugins()` — unavailable plugins are left out, not reported as SKIP.
- `QCContext(shot=, storyboard=, expect_duration_s=, expect_fps=, expect_resolution=)` — expectations must come from the **request actually sent**, not the storyboard. One decode is shared by all metrics (`FrameStats`, `DetectEvents`).
- `QCGate.evaluate(video, shot, sb, context=)` → `QCReport` (`score`, `verdict` ∈ `Verdict.PASS/RETAKE/ESCALATE`, `advice`, `failures()`, `table()`).
- `run_batch(...)`, `batch_qc(...)` — batch reports sorted worst first.

### `queue.py` — production queue (1484 lines)

A reliable single-machine queue. All state lives in SQLite, so the process can die at any time.

- `JobQueue(db_path, lease_s=600)`:
  - `enqueue` is idempotent on the shot fingerprint;
  - `claim` takes a lease; `heartbeat` renews it; `complete` / `fail` / `requeue` / `block` / `cancel`;
  - `reclaim_expired` recovers stuck jobs;
  - `mark_cache_hit`.
- `JobRecord` — one row holds everything needed to decide the next step after a power loss.
- `RetakeStrategy` — take 1: new seed; take 2: soften the camera move or shorten, and add artifact negatives; then: change engine. `derived_seed(fingerprint, take)` is deterministic, so a resumed run computes the same seed.
- `run_episode(storyboard, router, db_path=, concurrency=, resume=, qc_fn=, retake=, max_take=)` → `EpisodeResult`. It runs plan → enqueue → concurrent execution → QC → retake → summary, and **re-checks the cache just before execution**.
- `FileLock` (fcntl), `ConsoleProgress`, `QueueStats`.

### `cache.py` — content-addressed artifacts (702 lines)

- `ArtifactStore(root)` — sha256-addressed store: `put`, `put_bytes`, `get`, `has`, `link`, `stats`, `gc(roots)`. `iter_storyboard_uris(sb)` supplies the GC roots.
- `PromptCache(db)` — prompt + parameter fingerprint → artifact hash: `key_for`, `lookup`, `put`, `prune`, `verify`.
- `hash_file` streams, never reading whole videos into memory. `open_sqlite()` is shared with `queue.py`. `_ThreadLocalDB.close()` closes only the calling thread's connection.

---

## 5. Post-production & delivery

### `grade.py` — cross-shot colour matching (1009 lines)

- `sample_stats(video, frames=)` → `ColorStats` (`luma_mean`, `cct_k`, per-channel mean/std, …; `drift_from()`).
- `scene_reference(clips)` — per-field **median** across the scene. The mean is dragged off by one broken shot.
- `compile_match(src, ref, strength=, max_gain=)` → `MatchParams` — Reinhard mean/std → per-channel affine; emits a `curves` or `eq+colorbalance` filter.
- `match_to_reference(video, ref_stats, out, src_stats=, strength=)`, `generate_lut(src, ref, out_cube, size=)` (`.cube` for lut3d / DaVinci / OBS), `grade_report(...)`.

### `stitch.py` — atomic-shot assembly (1405 lines)

- `FFmpeg` — the single exit for ffmpeg calls: `run`, `run_capture`, `probe` → `MediaInfo`, `exact_duration`. `LONGFILM_FFMPEG` / `LONGFILM_FFPROBE` override the binaries.
- `extract_first_frame`, `extract_last_frame(back_off_frames=)`.
- `VideoTarget.from_delivery / from_clips` — the normalization target.
- `ClipSpec`, `resolve_clips(...)`.
- `concat_hard(...)` — concat demuxer + `-c copy` when parameters match.
- `concat_with_transitions(...)` — xfade chain.
- `overlap_blend(a, b, out, overlap_s=, mode=)`.
- `clamp_transition_s(want, left, right)` — the **only** clamping rule for transition length.
- `expected_duration(clips, durations)` — for assertions.
- `assemble_episode(clips, out, audio_tracks=, delivery=, audio_plan=, subtitles=, transition_s=, use_clip_audio=, two_pass_loudnorm=)` — picture + mix + loudnorm + optional subtitle burn-in + encode, in one `filter_complex`. `measure_loudness` is the first loudnorm pass.

### `upscale.py` — upscaling & frame interpolation (1241 lines)

- `Hardware.detect()` (`has_cuda`, `vram_gb`, `cpu_cores`, `ram_gb`, `comfy_url`).
- Upscalers: `FFmpegUpscaler` (CPU baseline: zimg + unsharp), `RealESRGANUpscaler`, `SeedVRUpscaler` (quality ceiling, VRAM-hungry), `ComfyUpscaler` (remote GPU).
- Interpolators: `FFmpegMinterpolate`, `RifeInterpolator` (`RIFE_DIR`).
- `plan_postprocess(delivery, hw, source_wh=, source_fps=, duration_s=, prefer=)` → `PostPlan` (`steps`, `warnings` with every rejected option and its reason, `est_total_s`, `describe()`, `run()`).
- `plan_chunks(duration_s, chunk_s=, overlap_s=)`, `chunked_process(video, fn, out, ...)` — invariant: Σwindows − (n−1)·overlap = duration.

### `timeline.py` — tracks and NLE interchange (1162 lines)

- `Timeline`, `Track`, `Clip`, `TrackKind`; `build_timeline(storyboard, renders, dialogue=, default_transition_s=, fps=)`.
- Exports: `export_edl` (CMX3600), `export_otio` (OpenTimelineIO JSON), `export_ass_subtitles(timeline, font=, style=)` with `wrap_cjk` line breaking, `export_ffmpeg_concat`.
- `ass_burn_filter(ass_path, fonts_dir=)` — filtergraph-safe escaping.
- `sync_report(timeline, tolerance_s=)` — audio/video alignment within about one frame.
- Timecode helpers: `seconds_to_frames`, `frames_to_timecode(drop_frame=)`, `seconds_to_timecode`.

### `compliance.py` — labelling & provenance (1145 lines)

- `add_visible_disclosure(video, out, text=, position=, font=, height_ratio=0.05, start_s=, end_s=None)` — burns the label with libass; the label stays on screen for the whole film by default. Refuses to run without a CJK font rather than rendering boxes.
- `write_metadata(video, out, manifest)` + `read_metadata(video)` — implicit label. Written metadata must be read back before it counts as written.
- `ProvenanceManifest.from_storyboard(sb, content_producer=, producer_code=, model_versions=, asset_root=)`. Per shot (`ShotProvenance`) it records:
  - provider, model version, job id, seed, shot fingerprint, LoRA;
  - `RefDigest` per reference: sha256 plus `source=` type;
  - age statements.

  `implicit_label()`, `save()`, `load()`.
- `c2pa_stub(manifest)` — unsigned C2PA manifest definition.
- `compliance_check(storyboard)` — release gate; an empty list means clear to publish. `parse_source_type(ref)`, `iter_refs(sb)`.

---

## 6. Training flywheel

### `distill.py` — teacher dataset (1197 lines)

- `LicenseFact`, `license_of(provider)` — vendor terms with `source` and `checked_on`. Unregistered providers are **unknown**, never allowed by default.
- `RenderRecord` → `harvest(renders, qc_reports, min_score=, allowed_ratings=)` → `DistillDataset` (only QC-passed shots; `trainable()`, `license_breakdown()`, `dataset_hash()`).
- `to_training_set(dataset, out_dir, fps=, resolution=, clip_s=, caption_style=, trigger=)` → `TrainingSetManifest` — clips + captions laid out for musubi-tuner / diffusion-pipe. `musubi_target_frames` enforces 4k+1.
- `diversity_metrics`, `diversity_report`, `mix_general_data(dataset, general_dir, ratio=)` → `MixPlan` (the `num_repeats` needed to keep general data at the target share).

### `lora.py` — LoRA job generator (1192 lines)

- `LoRATrainingJob` — the complete, reproducible job. Fields include `stack` (musubi / diffusion-pipe), `base_model` (default `wan2.2-i2v-a14b`), `dataset_dir` / `dataset_hash` / `dataset_remote`, `rank` / `alpha`, `lr` (0 = stack default), `epochs`, `MemoryOpts`, `save_state=True` and `resume_from`.
- `emit_musubi_config`, `musubi_cache_argv`, `musubi_train_argv` (adds `--save_state` / `--resume`), `emit_diffusion_pipe_config`, `emit_diffusion_pipe_dataset`, `emit_train_script(job, platform=)`.
- `plan_resume(job)` → `(state_path, remaining_epochs)`:
  - picks the latest `*-state` directory **by progress in its name, not mtime** (handles both epoch- and step-style names);
  - computes the remaining epochs, because musubi's `--resume` would rerun the full `max_train_epochs`.
- `estimate_cost(job, gpu, tier=)` → `CostEstimate` (always a range), `cheapest_viable_gpu(job, tier=)`. Always warns for 14B models on cards under 40 GB.
- `recommend_character_lora(character, ...)` → `CharacterLoRARecipe`; `dual_anchor_config(character, shot)` → `DualAnchorPlan` (LoRA strength vs reference-image weight).

### `dpo.py` — preference alignment & flywheel health (1769 lines)

- `PreferencePair`, `PreferenceDataset` (`filter_margin`, `merge`, `reweight_videodpo`, `to_distill_dataset`, …). `PairSource` must be recorded: automatic and human labels are biased in different directions.
- `build_pairs_from_qc(renders, qc_reports, min_margin=, ...)`, `build_pairs_from_human(ratings_csv, ...)`, `export_for_training(dataset, out_dir, fmt)`.
- `audit_static_bias(dataset)` → `StaticBiasAudit` — do winners skew static? (Automatic QC tends to reward still footage.)
- `FlywheelMonitor` — per-round diversity snapshot (`effective_modes` = exp(Shannon entropy) over shot sizes and camera moves) with collapse alerts (`AlertLevel`, `CollapseAlert`).

---

## 7. Infrastructure

### `_proc.py` — subprocess hardening (74 lines)

`run(cmd, **kw)` and `popen(cmd, **kw)` are drop-in `subprocess` replacements. `PREEXEC` sets `prctl(PR_SET_PDEATHSIG, SIGKILL)`, so the kernel kills the child ffmpeg even when the parent is SIGKILLed. This prevents orphaned transcodes; one such orphan once ran for 18 hours. Every ffmpeg spawn in the package goes through it (`stitch` passes `PREEXEC` directly).

### `_fonts.py` — CJK font lookup (85 lines)

`cjk_font()` → first existing of:

1. `LONGFILM_FONT`
2. `assets/fonts/` (preferred names, then any font file)
3. System Noto CJK / WenQuanYi / Microsoft YaHei on the WSL host / PingFang

It returns `None` when none is found. `require_cjk_font()` raises with instructions instead. `family_name(path)` reads the family from the file, which libass needs.

### `cli.py` — command line (109 lines)

Installed as `longfilm` (`pip install -e .`), or run with `python -m longfilm.cli`:

| Command | Does |
|---|---|
| `storyboard` | Build the 30-shot demo storyboard (two fictional adult characters) |
| `run [...]` | Run the end-to-end pipeline (forwards to `scripts/run_pipeline.py`) |
| `demo [...]` | Render the latest run into a demo video (`scripts/make_demo.py`) |
| `cost [...]` | Full-API vs self-hosted cost comparison (`deploy/cost_calculator.py`) |
| `selftest [pattern]` | Run module self-tests |
| `modules` | List modules |

---

## 8. Scripts

| Script | Purpose |
|---|---|
| `build_demo_storyboard.py` | Writes `configs/demo_episode.json`: 30 shots, 2 fictional adult characters (林岚, 凯 K-7), about 4.3 min, with source-type notes on every reference image. |
| `run_episode.py` | `setup_engines()` registers the `PricedMock` channels (`mock-official`, `mock-open`, `mock-open-spot`) so routing and the ledger have real prices without real vendors: an expensive, moderated tier-5 channel with stepped durations up to 15 s, and a cheap, unmoderated tier-2 channel with LoRA support. Also runs route → render → ledger for one episode on its own. |
| `run_pipeline.py` | The six-stage production pipeline (`audio → render → grade → assemble → post → deliver`). Flags: `--work`, `--resume`, `--stages`, `--limit`, `--scale`. `STAGE_DEPS` auto-adds `audio` before `assemble`/`deliver`; `_assemble_batched` builds in batches of 6. |
| `make_demo.py` | Renders a pipeline run into an explanatory demo video (18 segments). `--quick`, `--reuse`. |
| `demo_visuals.py`, `demo_render.py` | Pillow components and the ffmpeg timeline (xfade offset accumulation, ambient bed, mux) used by `make_demo.py`. |

## 9. deploy/

| File | Purpose |
|---|---|
| `README.md` | Cloud GPU manual: picking a card, real prices (RunPod / Vast / Paperspace), boot, render, train, known pitfalls. |
| `bootstrap_gpu.sh` | One-shot pod setup: health check → model download by profile → ComfyUI + nodes → smoke test. One exit code per failure. |
| `models.yaml` | Model manifests per profile (smoke / production / training). Sizes come from the HuggingFace API; sha256 is marked unverified until checked on download. |
| `prices.yaml`, `cost_calculator.py` | Price table (checked 2026-09-20; on-demand / spot / Paperspace plus subscriptions) and the calculator (`--tier paperspace` adds the monthly subscription). |
| `train_lora_pod.sh` | Sync data → train → sync back; auto-resumes via `plan_resume` and syncs state to `OUT_REMOTE` every 600 s, so spot instances are safe. |
| `runpod_serverless_handler.py` | RunPod Serverless entry; returns large outputs via S3. |
| `Dockerfile`, `docker-compose.yml` | Container image and compose file (ComfyUI + control plane). |
| `comfy_workflows/` | API-format workflow templates with their semantic binding tables. |
| `_miniyaml.py` | Dependency-free YAML subset reader for the manifests. |

---

## Tools and environment

| Variable | Default / lookup order |
|---|---|
| `LONGFILM_FFMPEG`, `LONGFILM_FFPROBE` | `bin/ffmpeg` → `PATH` → `imageio-ffmpeg`. A static build with libass is recommended; drawtext is **not** required. |
| `LONGFILM_FONT` | See `_fonts.py` above. |
| `ARK_API_KEY`, `DASHSCOPE_API_KEY`, `COMFY_URL`, `FAL_KEY`, `REPLICATE_API_TOKEN` | Engine credentials. Unset means that engine is simply not registered or healthy; the mock engine always works. |
| `HF_TOKEN`, `AWS_REGION` | Training downloads, serverless outputs. |

See [../.env.example](../.env.example) for the full list.
