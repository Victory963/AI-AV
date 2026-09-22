# Character reference images (not included in the repo)

The pipeline makes only **fictional adult characters**. Reference images live in `assets/cast/<character_id>/`. The demo storyboard (`scripts/build_demo_storyboard.py`) uses file names such as `lin_lan/front_neutral.png` and `kai_k7/front_neutral.png`. The mock engine does not need these files to exist, so the whole pipeline runs without them.

Before real production:

- **Each reference image must have a source type** (`source=ai_generated|licensed|original|...`) in the `note` field of `ImageRef`. `compliance.compliance_check()` blocks unlabeled images before release.
- **Do not use real people's likenesses.** `CharacterBible.is_fictional` is fixed to `True` and `age_statement` is a required field. `router.content_gate` rejects a shot whose on-screen character is not in the character bible or lacks an adult statement, before scoring even starts.
- Check image quality with `charbible.inspect_portrait()` / `validate_bible(check_images=True)`:
  - The short side must be at least 1024 (below 512 is a blocking error).
  - The aspect ratio must be between 0.6 and 1.7.
  - No transparent background.
  - No UI overlays such as screenshot borders or buttons.
- For the definitive list of reference images to shoot, see the output of `CharacterBibleBuilder.shot_list()`, which gives views × lighting setups × expressions × shot sizes.
