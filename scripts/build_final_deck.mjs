import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  FileBlob,
  PresentationFile,
} from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(`SHORT_ERROR: ${error.message}`);
  process.exit(1);
});
process.on("unhandledRejection", (error) => {
  console.error(`SHORT_ERROR: ${error?.message ?? error}`);
  process.exit(1);
});

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const workspace = path.join(root, "tmp/artifacts/final_deck");
const starter = path.join(
  root,
  "deliverables/presentation/FMRG_Final_Template_Starter.pptx",
);
const output = path.join(
  root,
  "deliverables/presentation/FMRG_Final_Submission_Audited.pptx",
);
const resultImage = path.join(
  root,
  "results/final_submission/figures/track21_held_out_comparison.png",
);
const importanceImage = path.join(
  root,
  "results/final_submission/figures/feature_importance.png",
);

const presentation = await PresentationFile.importPptx(await FileBlob.load(starter));
const inventory = (await fs.readFile(
  path.join(root, "scripts/final_deck_template_inventory.ndjson"),
  "utf8",
))
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

async function findOne(slideNumber, oldText) {
  const records = inventory.filter(
    (record) => record.slide === slideNumber && record.text?.includes(oldText),
  );
  if (records.length !== 1) {
    throw new Error(`Expected one inventory record on slide ${slideNumber} for ${oldText}; found ${records.length}`);
  }
  const slide = presentation.slides.getItem(slideNumber - 1);
  const matches = slide.shapes.items.filter((shape) => shape.name === records[0].name);
  if (matches.length !== 1) {
    throw new Error(`Expected one match on slide ${slideNumber} for ${oldText}; found ${matches.length}`);
  }
  return matches[0];
}

async function rewrite(slideNumber, oldText, newText) {
  const target = await findOne(slideNumber, oldText);
  target.text.replace(oldText, newText);
}

async function forceRewrite(slideNumber, oldText, newText) {
  const target = await findOne(slideNumber, oldText);
  target.text.set(newText);
}

async function imageBytes(path) {
  const bytes = await fs.readFile(path);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function replaceImage(slideNumber, imageName, path, alt) {
  const slide = presentation.slides.getItem(slideNumber - 1);
  const matches = slide.images.items.filter((image) => image.name === imageName);
  if (matches.length !== 1) {
    throw new Error(`Expected one image named ${imageName} on slide ${slideNumber}; found ${matches.length}`);
  }
  const image = matches[0];
  const oldFrame = image.frame;
  const oldCrop = image.crop;
  const oldFit = image.fit;
  const oldGeometry = image.geometry;
  const oldBorderRadius = image.borderRadius;
  const oldRotation = image.rotation;
  const oldFlipHorizontal = image.flipHorizontal;
  const oldFlipVertical = image.flipVertical;
  const oldLockAspectRatio = image.lockAspectRatio;
  image.replace({
    blob: await imageBytes(path),
    contentType: "image/png",
    alt,
    ...(oldFit ? { fit: oldFit } : {}),
  });
  image.frame = oldFrame;
  image.crop = oldCrop;
  image.geometry = oldGeometry;
  image.borderRadius = oldBorderRadius;
  image.rotation = oldRotation;
  image.flipHorizontal = oldFlipHorizontal;
  image.flipVertical = oldFlipVertical;
  image.lockAspectRatio = oldLockAspectRatio;
}

const edits = [
  [1, "NSF FUTURE MANUFACTURING DATA CHALLENGE  |  FINALIST PRESENTATION", "NSF FUTURE MANUFACTURING DATA CHALLENGE  |  FINAL SUBMISSION"],
  [1, "Predicting DED track geometry\nvia multi-modal sensor fusion", "Predicting local DED track width\nfrom thermal history"],
  [1, "Fusing in-situ thermal history with substrate surface awareness to predict local track height", "Audited physical alignment, local boundaries, uncertainty, and held-out evaluation"],
  [1, "Team Submission\nJuly 31, 2026   |   github.com/alphons3t/nsf-chuds", "Team Submission\nJuly 31, 2026   |   Reproduced from the official Zenodo release"],

  [2, "Can we predict part geometry during printing?", "Can thermal history predict local geometry after printing?"],
  [2, "▪  Directed Energy Deposition (DED) builds metal parts track by track — but final geometry varies along every track.\n▪  In-situ sensors watch the melt pool in real time; post-process metrology reveals what was actually built — hours later.\n▪  The challenge: connect what the camera sees during printing to the local geometric variation measured after printing.\n▪  If we can do this, we unlock real-time quality prediction and closed-loop control for metal additive manufacturing.", "▪  A DED track is not a uniform line: width and both boundaries vary along x.\n▪  Thermal frames record the moving melt pool; profilometry records the final geometry.\n▪  The modeling task is local: map each frame and short history to the matching segment.\n▪  A credible result must generalize across laser power and quantify uncertainty."],
  [2, "Predict local track height along the laser path from in-situ monitoring data", "Predict local width and boundary position from thermal history"],
  [2, "Ground truth:  leveled 3D profilometer height maps\nInputs:  thermal video + SEM substrate imagery\nScope:  4 tracks spanning different laser powers & scan speeds", "Ground truth:  left/right boundaries from 3D profilometry\nInputs:  thermal video; masked SEM is tested separately\nSplit:  develop on Tracks 8, 10, 14; hold out Track 21"],

  [3, "Three sensors, three languages: time, texture, and space", "Three modalities, one physical x-axis, one held-out condition"],
  [3, "In-situ melt pool dynamics", "400 active frames per track"],
  [3, "–  400 frames per track  ·  50 fps\n–  14 µm/pixel  ·  5.6 × 5.6 mm field of view\n–  Captures melt pool length, width, area\n–  Lives in the TIME domain", "–  50 fps at 10 mm/s = 0.2 mm per frame\n–  Size, bounding box, intensity, gradients\n–  Asymmetry, deltas, lags, rolling history\n–  Lives in the TIME domain"],
  [3, "Substrate surface morphology", "Substrate-only texture candidate"],
  [3, "–  Tiled .tif images along each track\n–  Reveals gouges, texture & roughness\n–  Used only outside the track (no leakage)\n–  Lives in the TEXTURE domain", "–  13-14 tiles placed from 100 mm to 20 mm\n–  Central 30% band excluded before features\n–  Texture candidate tested, not assumed useful\n–  Lives in the TEXTURE domain"],
  [3, "Post-process ground truth", "Local width and boundaries"],
  [3, "–  Bruker/Wyko full-field height maps\n–  x, y in mm  ·  z resolved to nanometers\n–  Defines the true deposited track height\n–  Lives in the SPACE domain", "–  Robust cross-track substrate detrending\n–  Connected 30%-height boundary crossings\n–  0.40 mm physical smoothing window\n–  Missing acquisition gaps stay excluded"],
  [3, "The core difficulty:  these modalities share no common coordinate system — dynamic video frames must be fused with static spatial maps before any model can learn.", "Evaluation is cross-condition: grouped leave-one-track-out development on 8, 10, 14; Track 21 is scored after the pipeline is locked."],

  [4, "From three raw sensors to one physical dataset", "From raw frames to leakage-safe local predictions"],
  [4, "Level", "Extract"],
  [4, "Fit & subtract a 2D plane from the raw 3D scan to isolate true track height", "Detrend each cross-section and find connected left/right crossings"],
  [4, "Align", "Align"],
  [4, "Map 400 thermal frames onto the 20–100 mm spatial axis of the track", "Use physical x in mm; never normalized row position"],
  [4, "Engineer", "Select"],
  [4, "Build thermal-history and substrate-awareness features at each location", "Compare fixed model families by grouped development CV"],
  [4, "Predict", "Test"],
  [4, "Train a Random Forest on 4 tracks to predict local track height", "Fit Tracks 8, 10, 14; evaluate width and boundaries on Track 21"],
  [4, "Why this matters:  every downstream feature is expressed in physical millimeter coordinates along the track — the model learns physics, not sensor artifacts.\nGenerative AI (LLM) was used transparently as a collaborative coding tool: designing the leveling math, debugging dataframe alignment, and refining visualizations.", "Audit rule: geometry extraction, feature sets, model selection, and uncertainty calibration are derived without Track 21 labels. The final score is a cross-power generalization test."],

  [5, "Building a common coordinate system", "Ground truth first: robust boundaries, then physical alignment"],
  [5, "STEP 1 — MATHEMATICAL 3D LEVELING", "STEP 1 - LOCAL GEOMETRY EXTRACTION"],
  [5, "Recover the true track from a tilted, warped substrate", "Measure one connected bead in every valid cross-section"],
  [5, "▪  Raw profilometer scans carry significant planar tilt.\n▪  We isolate the bare substrate edges (non-track regions) and fit a 2D plane via linear regression.\n▪  Subtracting the fitted plane levels the substrate to ≈ 0, exposing the true deposited height — our ground truth.", "▪  Interpolate missing y pixels only within each cross-section.\n▪  Fit substrate slope outside the bead shoulders.\n▪  Locate connected 30%-of-peak crossings around the central maximum.\n▪  Smooth left and right boundaries over 0.40 mm."],
  [5, "STEP 2 — SPATIAL–TEMPORAL ALIGNMENT", "STEP 2 - PHYSICAL TIME-SPACE ALIGNMENT"],
  [5, "Fuse video time with physical space", "Match each thermal frame to measured x in millimeters"],
  [5, "▪  Melt pool length, width & area extracted from each thermal frame by intensity thresholding (value = 2000).\n▪  Constant scan speed (10 mm/s @ 50 fps) → each frame advances 0.2 mm along the track.\n▪  Linear interpolation maps all 400 frames onto the 20–100 mm coordinates of the leveled height profile.", "▪  Detect laser shutoff and retain the prior 400 active frames.\n▪  Convert frame index to x using 10 mm/s at 50 fps.\n▪  Interpolate geometry at each frame's physical coordinate.\n▪  Reject frames farther than 0.10 mm from valid profilometry."],

  [6, "Feature engineering: physical memory and context", "Model selection: rich thermal history wins without SEM"],
  [6, "THERMAL HISTORY", "THERMAL DESCRIPTORS"],
  [6, "DED is a continuous thermal process — not independent frames", "Short histories describe pool shape, energy, and change"],
  [6, "▪  Rolling averages of melt pool dimensions over 5- and 10-frame windows.\n▪  These act as a proxy for heat accumulation and cooling rate at each point along the path.\n▪  Captures the longitudinal variation a single frame cannot see.", "▪  Area, equivalent diameter, bounding box, and elongation.\n▪  Peak/percentile temperature, hot-region mean, thermal mass.\n▪  Gradients, asymmetry, deltas, one-frame lags, rolling means.\n▪  Physical x harmonics capture repeatable spatial structure."],
  [6, "SUBSTRATE AWARENESS", "GROUPED MODEL SELECTION"],
  [6, "The surface the laser lands on shapes what it builds", "Three development tracks choose the model without Track 21"],
  [6, "▪  Localized SEM intensity and SEM roughness matrices extracted from masked SEM imagery.\n▪  Quantifies the topographical foundation prior to deposition — gouges, texture, reflectivity.\n▪  Track region masked out to strictly avoid output leakage.", "▪  Compare Ridge, Extra Trees, and histogram boosting.\n▪  Leave one whole track out in every development fold.\n▪  Ridge (alpha 10) + thermal-only features minimizes CV MAE.\n▪  Worst-track conformal residual sets the 90% interval."],

  [7, "The model tracks real variation across four recipes", "Audited model reduces held-out Track 21 MAE by 12.3%"],
  [7, "EXHIBIT 1 — ACTUAL VS PREDICTED TRACK HEIGHT", "EXHIBIT 1 - HELD-OUT LOCAL TRACK WIDTH"],
  [7, "21.18 µm\nMean Absolute Error", "139 µm\nHeld-out MAE"],
  [7, "0.41  R²\nheld-out test data", "12.3% lower\nvs. notebook baseline"],
  [7, "▪  Random Forest Regressor — 150 estimators, max depth 12, 80/20 train–test split.\n▪  Trained jointly on Tracks 8, 10, 14 & 21 to generalize across laser powers and scan speeds.\n▪  Captures longitudinal variation via thermal rolling averages and constrains boundaries via melt pool width.\n▪  Validated directly against leveled profilometer ground truth in physical µm.", "▪  Baseline MAE: 0.159 mm; audited Ridge MAE: 0.139 mm.\n▪  RMSE improves from 0.187 mm to 0.167 mm.\n▪  Development grouped-CV MAE improves from 0.137 mm to 0.088 mm.\n▪  Point prediction improves, but held-out R² remains negative (-0.58)."],

  [8, "The substrate — not the melt pool — drives prediction", "Thermal history generalizes better than masked SEM"],
  [8, "EXHIBIT 2 — RANDOM FOREST FEATURE IMPORTANCE", "EXHIBIT 2 - VALIDATION PERMUTATION IMPORTANCE"],
  [8, "WHAT THE FOREST LEARNED", "WHAT THE AUDIT FOUND"],
  [8, "SEM intensity is the single most dominant feature", "Hot-region temperature and thermal mass lead"],
  [8, "1.  Substrate texture governs heat dissipation and liquid metal flow.\n2.  Thermal history (rolling melt pool areas) ranks next — heat buildup matters.\n3.  Instantaneous melt pool size alone is the weakest signal.", "1.  Hot-region mean temperature produces the largest validation MAE increase.\n2.  Thermal mass and its change capture process memory beyond pool size.\n3.  Adding masked SEM worsens grouped CV MAE: 0.114 mm vs. 0.088 mm.\n4.  With four tracks, process and substrate effects are not yet identifiable."],

  [9, "Limitations & uncertainty: physics pushes back", "Improved, but not reliable enough for closed-loop control"],
  [9, "▪  Thermal warping: the metal substrate bows during printing, so a perfectly rigid 2D plane fit cannot capture all deformation.\n▪  Deep gouges in the substrate produce some negative ground-truth heights (down to −40 µm) after leveling.\n▪  Model uncertainty grows in regions of extreme substrate deformation — exactly where the plane assumption weakens.\n▪  R² = 0.41 reflects genuine stochasticity in DED, not just model error — a meaningful signal in a noisy physical process.", "▪  Held-out R² is -0.58: the mean/condition shift still dominates local fit.\n▪  Nominal 90% intervals cover 76.5% of Track 21 points.\n▪  Track 21 profilometry is incomplete; only valid gaps are scored.\n▪  Mean left/right boundary MAE is 0.174 mm.\n▪  Four process conditions cannot cleanly separate substrate from process."],
  [9, "IF WE HAD MORE TIME", "WHAT WOULD CLOSE THE GAP"],
  [9, "–  Replace the rigid plane with a flexible spline / polynomial surface to absorb warping.\n–  Quantify prediction intervals per location (e.g., quantile forests).\n–  Extend from height to full width & boundary-position descriptors.\n–  Validate across additional tracks and unseen laser recipes.", "–  Add more powers, plates, repeats, and substrate pre-scans.\n–  Register SEM before processing rather than infer substrate from post-scan tiles.\n–  Calibrate intervals by laser-power regime or hierarchical conformal methods.\n–  Predict a full contour distribution, not only center plus width."],

  [10, "Thermal monitoring alone is not enough", "The audit changed both the model and the claim"],
  [10, "Fusion is the answer", "Better held-out width"],
  [10, "Only by combining thermal history with substrate surface awareness can DED geometry be predicted reliably.", "Track 21 MAE improves from 159 to 139 µm; RMSE improves from 187 to 167 µm."],
  [10, "Physics-grounded pipeline", "Leakage controls matter"],
  [10, "Automated 3D leveling + spatial–temporal alignment turned three incompatible sensors into one physical dataset.", "Connected boundaries, physical x alignment, grouped CV, and a masked-SEM ablation make the result defensible."],
  [10, "Quantified & validated", "Uncertainty remains open"],
  [10, "MAE of 21.18 µm and R² of 0.41 against leveled profilometer ground truth, generalized over 4 laser recipes.", "Coverage is 76.5% and R² remains negative, so this is a better benchmark - not a deployment-ready controller."],
  [10, "“The foundation a part is built on matters as much as the energy used to build it.”", "“A smaller honest result is more useful than a larger unsupported one.”"],
  [10, "Thank you — Questions?    |    github.com/alphons3t/nsf-chuds", "Questions?  |  github.com/alphons3t/nsf-chuds  |  github.com/joeyperez1-debug/nsfsubmit"],
];

for (const [slide, oldText, newText] of edits) {
  await rewrite(slide, oldText, newText);
}

// artifact-tool's rich-text replacement is a no-op for a few inherited,
// multi-run text boxes. Set those frames explicitly so stale audit claims
// cannot survive the template import.
await forceRewrite(7, "21.18 µm\nMean Absolute Error", "139 µm\nHeld-out MAE");
await forceRewrite(7, "0.41  R²\nheld-out test data", "12.3% lower\nvs. notebook baseline");
await forceRewrite(
  7,
  "▪  Random Forest Regressor — 150 estimators, max depth 12, 80/20 train–test split.",
  "Baseline MAE: 0.159 mm; audited Ridge MAE: 0.139 mm.\nRMSE improves from 0.187 mm to 0.167 mm.\nDevelopment grouped-CV MAE improves from 0.137 mm to 0.088 mm.\nPoint prediction improves, but held-out R² remains negative (-0.58).",
);
await forceRewrite(
  9,
  "▪  Thermal warping: the metal substrate bows during printing",
  "Held-out R² is -0.58: the condition shift still dominates local fit.\nNominal 90% intervals cover 76.5% of Track 21 points.\nTrack 21 profilometry is incomplete; only valid regions are scored.\nMean left/right boundary MAE is 0.174 mm.\nFour conditions cannot cleanly separate substrate from process.",
);
await forceRewrite(
  9,
  "–  Replace the rigid plane with a flexible spline / polynomial surface to absorb warping.",
  "Add more powers, plates, repeats, and substrate pre-scans.\nRegister SEM before processing instead of inferring substrate from post-scan tiles.\nCalibrate intervals by laser-power regime or hierarchical conformal methods.\nPredict a full contour distribution, not only center plus width.",
);

await replaceImage(
  7,
  "Picture 7",
  resultImage,
  "Held-out Track 21 measured width, notebook baseline, audited model, and 90% interval",
);
await replaceImage(
  8,
  "Picture 7",
  importanceImage,
  "Validation permutation importance for the selected thermal Ridge model",
);

const notes = [
  "Opening: this revision reports reproduced local width and boundary results, not the unsupported height claims in the prior deck.",
  "Challenge framing: local geometry is a spatial signal and generalization is across laser-power conditions.",
  "Data: official Zenodo archives were checksum-verified before analysis.",
  "Pipeline: Track 21 labels do not participate in feature or model selection.",
  "Geometry: isolated dust and missing profilometer pixels do not count as track width.",
  "Modeling: masked SEM is a tested ablation, not a presumed source of lift.",
  "Result: emphasize the 12.3% MAE improvement and the still-negative R2 together.",
  "Interpretation: thermal descriptors win this split; the data do not support a causal substrate claim.",
  "Limitations: uncertainty under-covers and more experimental conditions are required.",
  "Close: the contribution is a more defensible benchmark and pipeline, not a control-ready model.",
];
const sources = [
  "https://doi.org/10.5281/zenodo.21285367",
  "https://arxiv.org/abs/2607.07965",
  "results/final_submission/metrics.json",
];
for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.getItem(index);
  slide.speakerNotes.textFrame.setText(
    `${notes[index]}\n\n[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}\n[/Sources]`,
  );
  slide.speakerNotes.setVisible(true);
}

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.mkdir(workspace, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);

for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.getItem(index);
  const stem = `final-slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${workspace}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${workspace}/${stem}.layout.json`, await layout.text());
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${workspace}/final-montage.webp`, new Uint8Array(await montage.arrayBuffer()));
