import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const F = require("../sneakey/live_features.js");

// The model and validation clips are generated from local recordings by analysis/export_model.py and are not
// committed (analysis_output/ is git-ignored). Without them this parity test cannot run, so it is skipped.
const MODEL = new URL("../analysis_output/live_model/model.json", import.meta.url);
const VAL = new URL("../analysis_output/live_model/validation.json", import.meta.url);
const SKIP = !(existsSync(MODEL) && existsSync(VAL))
  ? "needs analysis_output/live_model/ - run `.venv/bin/python analysis/export_model.py` first" : false;
const model = SKIP ? null : JSON.parse(readFileSync(MODEL));
const val = SKIP ? null : JSON.parse(readFileSync(VAL));

test("JS features match Python features on validation clips", { skip: SKIP }, () => {
  const extract = F.makeExtractor(model.feature_config, model.sr_train, model.clip_samples);
  let maxAbs = 0, maxRel = 0;
  for (const c of val.clips) {
    const seg = Float64Array.from(c.samples_i16, (v) => v / 32768);
    const f = extract(seg);
    assert.equal(f.length, c.features.length, "feature length");
    for (let i = 0; i < f.length; i++) {
      const a = Math.abs(f[i] - c.features[i]);
      maxAbs = Math.max(maxAbs, a);
      maxRel = Math.max(maxRel, a / (Math.abs(c.features[i]) + 1e-6));
    }
  }
  console.log(`  feature max abs diff ${maxAbs.toExponential(2)}, max rel diff ${maxRel.toExponential(2)}`);
  assert.ok(maxAbs < 1e-3, `feature max abs diff ${maxAbs}`);
});

test("JS classifier reproduces Python top-5 probabilities and labels", { skip: SKIP }, () => {
  const extract = F.makeExtractor(model.feature_config, model.sr_train, model.clip_samples);
  const predict = F.makeClassifier(model);
  let maxP = 0, labelMismatch = 0;
  for (const c of val.clips) {
    const seg = Float64Array.from(c.samples_i16, (v) => v / 32768);
    const probs = predict(extract(seg));
    for (let j = 0; j < 5; j++) {
      assert.equal(probs[j][0], c.top5[j][0], `rank ${j}: ${probs[j][0]} vs ${c.top5[j][0]}`);
      maxP = Math.max(maxP, Math.abs(probs[j][1] - c.top5[j][1]));
    }
    if (probs[0][0] !== c.top5[0][0]) labelMismatch++;
  }
  console.log(`  top-5 prob max abs diff ${maxP.toExponential(2)}, argmax mismatches ${labelMismatch}/${val.clips.length}`);
  assert.ok(maxP < 1e-3, `prob diff ${maxP}`);
  assert.equal(labelMismatch, 0);
});
