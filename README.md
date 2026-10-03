# CentreWatch prototype

A browser-local prototype for SIH 26245, with administrator, institution-head, and trainer workspaces.

## Run locally

```sh
npm start
```

Open `http://127.0.0.1:4173`. No dependency installation or build is required. `npm run check` checks the three JavaScript modules.

## Review the workflow

1. Start as **Institution head**. Add a trainer if needed, then create a class and upload a JPEG, PNG, WebP, MP4, or WebM file (maximum 75 MB).
2. Switch the top-right role selector to **Trainer** and select the assigned trainer. Submit the classroom count.
3. The count is stored immediately. Uploaded media runs experimental COCO-SSD person detection in the browser. Photos use one observation; videos use a median across at most seven sampled frames. This is a visible-person estimate, not a unique or identity-based attendance measure.
4. Switch back to **Institution head** to review a mismatch or an unclear result. Record an acceptance note or request follow-up.
5. Switch to **Administrator** for institution-level oversight and daily CSV reports.

The eight seeded classes use labelled sample observations and a classroom illustration. They are explicitly marked as samples. A seeded class's check replays a fixed sample count; it does not run inference on the illustration. Uploaded media never receives a simulated fallback result.

## Boundaries

- Roles are prototype views, not authenticated or secure access controls.
- Records use localStorage; uploads use IndexedDB. Data belongs to the current browser/origin and is not shared across users or devices.
- First inference downloads TensorFlow.js, COCO-SSD, and model weights from their public hosting services. Images and videos are not uploaded by the app.
- Confidence labels are review heuristics, not calibrated accuracy. Low-confidence and failed checks remain unverified.
- Declared equipment is recorded, but this prototype does not verify equipment, machine operability, CCTV feeds, or PTZ control.
- Reports are available on demand; there is no background email or end-of-day scheduler.
- Closing the browser interrupts an active check. Reloaded interrupted checks become unverified and can be retried.
- Sample/trainer input, previous submissions, and analysis/review history remain separate from model estimates. An aligned aggregate count cannot verify trainee identities.

## Files

- `dist/app.js`: interface, role navigation, forms, attendance/review workflow, reports.
- `dist/store.js`: sample records, persistence, and media storage.
- `dist/vision.js`: actual browser inference, video sampling, quality flags, timeouts.
- `dist/styles.css`: responsive dashboard styling.

## Model reference

Uses the official [TensorFlow.js COCO-SSD model](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd). Libraries are pinned to TensorFlow.js 4.22.0 and COCO-SSD 2.2.3. Target classroom performance needs a separate labelled accuracy evaluation.

## Verification performed

- Browser walkthrough: create institution, create trainer, upload media and create class, switch to assigned trainer, submit attendance, review as institution head, inspect administrator report.
- Actual JPEG smoke test using a public two-person sample: submitted 3, detected 2, review alert produced.
- Actual four-second H.264 smoke test of the same image: three frames sampled, median count 2, submitted count 2 aligned. This validates video decoding and sampling, not performance on moving classroom footage.
- Labelled sample class: submitted 24, seeded result 24, sample status and provenance displayed.
- The smoke-test files and records are not part of the distributed seed dataset.
