# CentreWatch privacy design

## What the system observes

The local detector observes people, their apparent seated/standing posture, and configured equipment classes. Classes default to **seated people only**; a head can explicitly choose all visible people instead. Seated mode excludes standing detections and sends uncertain posture for review. It compares aggregate counts with the trainer's submitted attendance count. A seated trainer may be counted, and a standing student is excluded: posture does **not** establish a person's role or verify the identities of trainees on a register.

The system does not identify faces, create face templates, compare individuals across cameras, retain appearance embeddings for people, transcribe audio or infer protected attributes. Prepared models can contain text embeddings for equipment or posture vocabulary; these are class prompts, not representations of individual trainees. Temporary spatial tracks within one observation window help preserve recently observed seated presence through brief occlusion. They do not accumulate every person who ever appeared into one attendance total or match people across sessions. Detection boxes and posture observations are technical evidence, not persistent person identities.

## Observation duration and uncertainty

Each check requests 300 seconds by default, configurable from 300 to 900 seconds. Live capture needs actual elapsed observation time. Uploaded recordings are sampled from their available duration; a ten-second clip never becomes five minutes of evidence. The dashboard records the requested duration, observed duration, source duration where known, sampling interval and whether the requested window was completed. A photograph cannot demonstrate continued attendance.

Visible seated detections and temporally supported seated presence are separate measurements. A missing detection can be supported for **at most 3 seconds** after recent confirmed seated evidence. Separately, a person who remains **visible and spatially stable** may use seated posture confirmed within the previous **30 seconds** when the current posture classification is unclear; the dashboard reports these uses of recent posture history. This longer period never permits an absent person to remain counted for 30 seconds. A standing detection is excluded immediately. New uncertain candidates, prolonged absence, ambiguous position matching and camera movement require fresh evidence or review.

The result represents simultaneously supported presence during the observation, not the maximum count or a union of visitors across the recording. The pretrained detector and posture classifier have not been newly trained to prove student/trainer roles or perfect classroom accuracy.

After a person has been observed standing, a later crop that merely resembles sitting cannot immediately restore seated status. The transition requires visible shoulder/hip/knee posture evidence and repeated confirmation. Until that evidence is available, it remains excluded or uncertain and can trigger `seating_transition_unverified`. A desk hiding the lower body can therefore prevent verification even when the person has actually sat down; human review is required.

The camera still records people and the uploaded source remains potentially personal data. Aggregate output is not a claim that the entire deployment is anonymous or legally compliant.

## Processing and transmission

1. A head configures a room, sanctioned inventory, assigned trainer, class time and local video/photo or camera source.
2. In the centre deployment, the Python service reads the source on the local machine and runs the prepared model locally. Runtime inference requires no cloud vision API and performs no model download.
3. The optional edge client sends only explicitly selected metadata to the central service: device/class IDs, event UUID and timestamp, count, evidence-quality category, equipment quantities/status and short quality flags. It never transfers video, original images, camera credentials, local paths, face crops or per-person embeddings.
4. The dashboard presents submitted counts, observed counts, discrepancies, review notes and restricted camera snapshots for analysis performed by that service. Each snapshot shows the actual sampled frame with numbered person boxes. Observations received through metadata-only edge sync do not include snapshots. The reviewed decision and original submission remain auditable.

**Deployment distinction:** the browser's upload button sends the selected file to the service the browser is connected to. For rural deployment, run that service at the institution and use the metadata edge client for central reporting. Uploading directly to a remote server transfers the complete source video to that server; it does not qualify as the metadata-only mode.

## Camera snapshots and detection numbers

For dashboard analysis, the saved evidence contains up to three actual sampled camera frames with numbered person boxes, a timestamp and the visible count for that frame. In seated mode, boxes are drawn only for supported seated people, including a visible, spatially stable person supported by recent seated posture history. Standing people and uncertain candidates without that support are excluded from those boxes. They can still be visible in the original camera image. These snapshots retain source pixels and may show recognizable faces or other personal information. They are not anonymized or face-blurred. They use the same authenticated class access checks as the source footage and are served with `Cache-Control: no-store`.

“Person 1”, “Person 2” and subsequent labels identify detection boxes within one snapshot only. Numbers can refer to different people in another frame; they are not identities, trainee-register entries or persistent tracking IDs. Representative frames and count extremes support review without claiming that every individual detection is correct.

The inference library retains an abstract-diagram mode that removes every source pixel; older saved observations may use it. The optional edge sender still sends only count/status metadata, never these snapshots or the raw footage.

Administrators can inspect institutions, institution heads can access their institution and trainers can access their assigned class media and snapshots. Standardised notice and appropriate permissions for filming must be handled by the institution before the demonstration or deployment.

## Retention and deletion

| Data | Implemented treatment |
|---|---|
| Files copied through the upload API | Locally stored on the service host; configurable `uploadRetentionHours`, default 24 hours. Cleanup expires the source and preserves the class, observations and audit history. |
| User originals in `data/incoming/` | User-managed originals. The service never automatically deletes these files. The institution must apply its own retention policy. |
| Annotated camera snapshots and legacy evidence diagrams | Configurable `retentionDays`, default 30 days; this is independent of the original upload's expiry. Snapshots can remain after the original video expires. |
| Counts, submissions, reviews and audit events | Retained for the monitoring record; no automatic policy-driven deletion is presently claimed. Define the required scheme-specific retention period before deployment. |
| Local edge outbox | SQLite metadata queue survives restarts. `prune --days 30` removes acknowledged events older than the chosen period; pending/failed events are retained for investigation. |
| Evaluation outputs | Local JSON/CSV reports and predictions; no video copied into a report. Bounding boxes are included when evaluating labelled boxes. Operator manages their retention. |

Deletion from the running application is not a guarantee of erasure from operating-system backups. Upload expiry makes the class source unavailable for fresh analysis; supply a new source to continue. Derived observations remain available for audit.

## Access and integrity

The service authenticates separate administrator, institution-head and trainer accounts. Permission checks run on the server for institution and class access; switching a displayed role is not authorization. Passwords are hashed, session cookies are protected, and state-changing browser requests use CSRF protection. Device keys are scoped to a registered institution and must be kept off public clients. Edge ingestion validates class scope and deduplicates the same device/event UUID, preserving the original capture timestamp during delayed delivery.

Use HTTPS for remote deployment. The edge client refuses clear-text remote URLs, refuses redirects that might forward its key and allows HTTP only for an explicitly enabled loopback development test. SQLite files and local uploads need operating-system access controls; full-disk encryption and protected backups are deployment responsibilities. Camera credentials must stay in the authorised centre service configuration, not in screenshots, public repositories or exported reports.

## Evidence limitations and human review

- People entirely outside the field of view cannot be detected. Crowding, overlap, low resolution and lighting can produce missed detections.
- Seated posture does not establish student identity. A seated trainer or visitor can contribute to the seated count; a standing student is excluded. A configured trainee zone can reduce some of this ambiguity but cannot prove identity.
- Standing/seated classification can be uncertain when a desk hides the legs or the person bends over. Temporary position tracking cannot recover someone never detected or establish presence throughout an unobserved interval.
- Unknown visibility, unsupported equipment classes, poor-quality frames and camera failure must produce insufficient evidence, not a fabricated count or an automatic accusation.
- A camera can observe an item and sometimes activity. This prototype does not certify that a machine functions; operability is recorded as unverified.
- A still image does not demonstrate continued attendance. Replayed footage is not fully defeated by frozen-frame checks, and predictable sampling can be gamed.
- A flagged discrepancy is a request for review. It is not by itself evidence of fraud or grounds for a penalty.

Before a real scheme rollout, agree the notice, lawful purpose, retention, reviewer permissions, incident process and grievance mechanism with the responsible authority. This design note documents the implementation; it is not a legal compliance certificate.
