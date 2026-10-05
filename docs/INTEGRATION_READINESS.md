# Integration readiness

Verified on 4 October 2026 using Python 3.12, the installed dependencies recorded in `platform/requirements.lock.txt`, and the prepared local YOLO-World artifact. The lock file is an exact snapshot of the tested environment; deployments on other architectures should run the same checks after installation.

The complete test suite passed **44 tests**, including **15 API tests**. The API tests cover cookie authentication, CSRF/origin rejection, institution isolation, assigned trainer access, decoded media uploads, password changes and recovery, intake assignment, persistent queue recovery, submission revision races, failure handling, review records, idempotent edge events, retention boundaries, and safe CSV exports. Model calls in unit tests are isolated test stubs. Production has no simulated inference fallback.

A separate real HTTP check launched Uvicorn with a temporary database, randomly isolated test accounts, and the actual prepared model. It completed this workflow:

1. Administrator created an institution and assigned an incoming development image.
2. Institution head created a trainer and attached the assigned image to a class.
3. Trainer submitted 5; actual local inference observed 2 visible people.
4. The system produced a review alert and an abstract evidence diagram.
5. The head recorded follow-up; the admin downloaded the daily CSV report.
6. A compact observation survived closing and reopening the SQLite outbox, synced over HTTP to the server, and was acknowledged once. Sending the same event again returned a duplicate acknowledgment without creating another observation.

That single metadata event was **338 bytes of serialized JSON**. This excludes HTTP/TLS headers and transport overhead and is not a guaranteed deployment bandwidth figure. The machine-readable run record is `integration-check.json`.

This check used one development photograph. It does **not** establish classroom accuracy, video accuracy, inventory accuracy, performance on rural cameras, or live RTSP compatibility with a particular camera. Run the provided dataset evaluation on the user's independently labelled classroom footage to measure those results.

## Repeat the integration check

From `centre-watch/platform`:

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/check_http_integration.py \
  --image /absolute/path/to/development-image.jpg \
  --expected-visible-count 2 \
  --submitted-count 5
```

Supply the manually verified visible-person count for the development image. Choose a different submitted count to exercise discrepancy review. The script uses a temporary runtime, starts a loopback HTTP server, and stops it after the check. It does not create accounts or classes in the installed workspace.

## Footage intake

Put user-provided videos in `centre-watch/data/incoming/`. In the web app, the administrator assigns each filename to an institution. Only that institution's head can list and attach the file. Incoming originals remain under the user's control; automatic retention deletes application-owned uploads and evidence only.

## Full browser workflow and final regression check

The browser flow was also exercised against isolated QA records: institution-head sign-in, create a class using an assigned video and inventory, trainer sign-in and submit 3, actual observed count 2 with an observed chair shortfall (0 versus 1), head follow-up, administrator dashboard and dataset evaluation. All source pixels were absent from the derived evidence diagram. The public fixture is a timestamped still image and does not establish real classroom performance.

`PIPELINE_ASSESSMENT.json` preserves the actual evaluation: three manually counted timestamps, two visible people each, submitted attendance 3, true chair count 0, sanctioned chair count 1. It measures the evaluator and alert path on this fixture. It has no negative alert cases, no independent test rooms and no box annotations; it must not be presented as an accuracy or specificity estimate for training centres.

The final suite passed **48 tests and 5 subtests** on 4 October 2026. The only warning was an upstream Starlette/httpx deprecation. The installed application uses its own database and opens on first-admin setup; temporary QA accounts and observations were not copied into it.
