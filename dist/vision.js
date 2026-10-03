/**
 * Local person detection for the CentreWatch prototype.
 * Media is decoded and analysed in the browser. Only runtime/model downloads
 * use the network; images and video frames are never uploaded by this module.
 *
 * API reference: https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd
 * The confidence label below is a review heuristic, not measured accuracy.
 */

const TF_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const COCO_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js';
const PERSON_THRESHOLD = 0.45;
const MAX_BOXES = 100;
const MAX_FRAME_EDGE = 1280;
const scriptLoads = new Map();
let modelPromise;
let modelState = 'idle';
let lastModelError = '';
// COCO-SSD changes the TF backend during suppression, so serialize inference.
let inferenceTail = Promise.resolve();

export function getModelStatus() {
  return { state: modelState, error: lastModelError, model: 'COCO-SSD MobileNet' };
}

function abortError() {
  return new DOMException('Analysis was cancelled.', 'AbortError');
}

function checkAbort(signal) {
  if (signal?.aborted) throw abortError();
}

/** Stop waiting cleanly; an already executing GPU inference cannot be cancelled. */
function bounded(promise, { signal, timeout = 30000, message = 'This step took too long. Please try again.' } = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer;
    const finish = (handler, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', cancel);
      handler(value);
    };
    const cancel = () => finish(reject, abortError());
    Promise.resolve(promise).then(value => finish(resolve, value), error => finish(reject, error));
    signal?.addEventListener('abort', cancel, { once: true });
    if (signal?.aborted) return cancel();
    timer = setTimeout(() => finish(reject, new Error(message)), timeout);
  });
}

function loadScript(url, ready) {
  if (ready()) return Promise.resolve();
  if (scriptLoads.has(url)) return scriptLoads.get(url);
  const promise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    let timer;
    const fail = () => {
      clearTimeout(timer);
      script.remove();
      reject(new Error('The browser AI library could not load. Check your internet connection and allow cdn.jsdelivr.net, then retry.'));
    };
    script.src = url;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.onload = () => {
      clearTimeout(timer);
      script.onload = script.onerror = null;
      if (ready()) resolve();
      else fail();
    };
    script.onerror = fail;
    timer = setTimeout(fail, 45000);
    document.head.append(script);
  }).catch(error => {
    scriptLoads.delete(url);
    throw error;
  });
  scriptLoads.set(url, promise);
  return promise;
}

function loadModel() {
  if (modelPromise) return modelPromise;
  modelState = 'loading';
  lastModelError = '';
  modelPromise = (async () => {
    await loadScript(TF_URL, () => Boolean(globalThis.tf));
    await loadScript(COCO_URL, () => Boolean(globalThis.cocoSsd));
    try {
      await globalThis.tf.ready();
    } catch {
      await globalThis.tf.setBackend('cpu');
      await globalThis.tf.ready();
    }
    const loading = globalThis.cocoSsd.load({ base: 'mobilenet_v2' });
    let timedOut = false;
    // If a timed-out network request eventually completes, free its model.
    loading.then(model => { if (timedOut) model.dispose(); }, () => {});
    let model;
    try {
      model = await bounded(loading, {
        timeout: 120000,
        message: 'The AI model download timed out. Check your internet connection and retry. No attendance count was produced.'
      });
    } catch (error) {
      timedOut = true;
      throw error;
    }
    modelState = 'ready';
    return model;
  })().catch(error => {
    modelState = 'error';
    lastModelError = error?.message || 'The AI model could not be loaded.';
    modelPromise = undefined;
    throw new Error(`${lastModelError} Analysis is unavailable until the model loads.`, { cause: error });
  });
  return modelPromise;
}

function mediaReady(media, event, ready, signal, start) {
  return new Promise((resolve, reject) => {
    let timer;
    const cleanup = () => {
      clearTimeout(timer);
      media.removeEventListener(event, complete);
      media.removeEventListener('error', fail);
      signal?.removeEventListener('abort', cancel);
    };
    const complete = () => { cleanup(); resolve(); };
    const fail = () => {
      cleanup();
      reject(new Error('This file could not be decoded in your browser. Try a JPEG/PNG photo or an MP4 video encoded with H.264.'));
    };
    const cancel = () => { cleanup(); reject(abortError()); };
    media.addEventListener(event, complete, { once: true });
    media.addEventListener('error', fail, { once: true });
    signal?.addEventListener('abort', cancel, { once: true });
    timer = setTimeout(() => {
      cleanup();
      reject(new Error('Reading the uploaded media timed out. Try a smaller file or a different video format.'));
    }, 20000);
    if (signal?.aborted) return cancel();
    try {
      start?.();
      if (ready()) complete();
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

function fileKind(blob) {
  const name = String(blob.name || '').toLowerCase();
  if (blob.type === 'image/svg+xml' || name.endsWith('.svg')) {
    throw new Error('Please upload a photograph in JPEG, PNG, WebP, or another supported raster image format.');
  }
  if (blob.type.startsWith('image/') || /\.(jpe?g|png|webp|avif|gif|bmp)$/.test(name)) return 'image';
  if (blob.type.startsWith('video/') || /\.(mp4|m4v|mov|webm|ogv)$/.test(name)) return 'video';
  throw new Error('Choose a photo or video file to analyse. JPEG, PNG, MP4, and WebM are supported in most browsers.');
}

async function runDetection(model, canvas, signal) {
  const run = inferenceTail.then(() => {
    checkAbort(signal);
    return model.detect(canvas, MAX_BOXES, PERSON_THRESHOLD);
  });
  inferenceTail = run.catch(() => {});
  return bounded(run, {
    signal,
    timeout: 60000,
    message: 'Person detection took too long on this device. Try a smaller image or retry with other browser tabs closed.'
  });
}

/**
 * @param {Blob|File} blob Local media, for example a File restored from IndexedDB.
 * @param {{onProgress?: Function, signal?: AbortSignal}} options
 * @returns {Promise<object>} A visible-person snapshot estimate, never identity attendance.
 */
export async function analyzeMedia(blob, { onProgress = () => {}, signal } = {}) {
  checkAbort(signal);
  if (!(blob instanceof Blob) || !blob.size) throw new Error('The media file is missing or empty. Please upload it again.');
  const kind = fileKind(blob);
  const notify = (stage, progress, message) => {
    // UI updates must not interrupt or corrupt an otherwise valid analysis.
    try { onProgress({ stage, progress, message }); } catch { /* Consumer callback. */ }
  };
  notify('loading', 5, modelState === 'ready' ? 'Preparing the local person detector…' : 'Loading the browser AI model. Your media stays on this device…');
  const model = await bounded(loadModel(), {
    signal,
    timeout: 220000,
    message: 'The AI model could not load in time. Check your connection and try again.'
  });
  checkAbort(signal);
  notify('sampling', 20, 'Reading the uploaded media on this device…');
  const url = URL.createObjectURL(blob);
  const media = document.createElement(kind === 'video' ? 'video' : 'img');
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { willReadFrequently: false });
  const observations = [];
  try {
    if (!context) throw new Error('Your browser could not create an image canvas. Please try another browser.');
    if (kind === 'video') {
      media.muted = true;
      media.playsInline = true;
      media.preload = 'auto';
      await mediaReady(media, 'loadedmetadata', () => media.readyState >= 1, signal, () => {
        media.src = url;
        media.load();
      });
      await mediaReady(media, 'loadeddata', () => media.readyState >= 2, signal);
    } else {
      await mediaReady(media, 'load', () => media.complete && media.naturalWidth > 0, signal, () => { media.src = url; });
    }

    const width = kind === 'video' ? media.videoWidth : media.naturalWidth;
    const height = kind === 'video' ? media.videoHeight : media.naturalHeight;
    if (!width || !height) throw new Error('This media has no readable image dimensions. Please choose another file.');
    const scale = Math.min(1, MAX_FRAME_EDGE / Math.max(width, height));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    let times = [0];
    if (kind === 'video') {
      if (!Number.isFinite(media.duration) || media.duration <= 0) {
        throw new Error('The video duration could not be read. Export it as an MP4 or upload a classroom photo.');
      }
      const sampleCount = media.duration < 1 ? 1 : Math.min(7, Math.max(3, Math.ceil(media.duration / 3)));
      const last = Math.max(0, media.duration - Math.min(0.15, media.duration / 10));
      times = Array.from({ length: sampleCount }, (_, index) => sampleCount === 1 ? 0 : last * index / (sampleCount - 1));
    }

    for (let index = 0; index < times.length; index++) {
      checkAbort(signal);
      const time = times[index];
      if (kind === 'video' && (Math.abs(media.currentTime - time) > 0.02 || media.seeking)) {
        await mediaReady(media, 'seeked', () => !media.seeking && Math.abs(media.currentTime - time) < 0.02 && media.readyState >= 2,
          signal, () => { media.currentTime = time; });
      }
      if (kind === 'video' && media.readyState < 2) {
        await mediaReady(media, 'loadeddata', () => media.readyState >= 2, signal);
      }
      notify('detecting', Math.round(25 + 65 * index / times.length),
        kind === 'video' ? `Counting visible people in sample ${index + 1} of ${times.length}…` : 'Counting visible people in your photo…');
      context.drawImage(media, 0, 0, canvas.width, canvas.height);
      let predictions;
      try {
        predictions = await runDetection(model, canvas, signal);
      } catch (error) {
        if (error.name === 'AbortError') throw error;
        throw new Error(`The browser model could not complete detection. ${error.message || 'Please retry on another device.'}`, { cause: error });
      }
      checkAbort(signal);
      const boxes = predictions.filter(item => item.class === 'person' && item.score >= PERSON_THRESHOLD).map(item => ({
        bbox: [
          item.bbox[0] * width / canvas.width,
          item.bbox[1] * height / canvas.height,
          item.bbox[2] * width / canvas.width,
          item.bbox[3] * height / canvas.height
        ],
        score: item.score
      }));
      observations.push({ count: boxes.length, boxes, time, atLimit: predictions.length >= MAX_BOXES });
    }

    const samples = observations.map(item => item.count);
    const sorted = [...samples].sort((a, b) => a - b);
    // The lower median is an observed integer count, including for even samples.
    const count = sorted[Math.floor((sorted.length - 1) / 2)];
    const representative = observations.find(item => item.count === count);
    const allBoxes = observations.flatMap(item => item.boxes);
    const averageScore = allBoxes.length ? allBoxes.reduce((sum, box) => sum + box.score, 0) / allBoxes.length : 0;
    const spread = Math.max(...samples) - Math.min(...samples);
    const lowResolution = width < 640 || height < 360;
    const variable = spread > Math.max(2, count * 0.2);
    const smallPeople = allBoxes.some(box => box.bbox[3] / height < 0.06);
    const edgePeople = allBoxes.some(box => box.bbox[0] < 2 || box.bbox[1] < 2 || box.bbox[0] + box.bbox[2] > width - 2 || box.bbox[1] + box.bbox[3] > height - 2);
    const qualityFlags = [];
    if (lowResolution) qualityFlags.push('Low image resolution may hide distant or partially visible people.');
    if (smallPeople) qualityFlags.push('Some detected people appear very small in the frame; distant people may be missed.');
    if (edgePeople) qualityFlags.push('A person appears at the frame edge. Review the camera coverage.');
    if (!allBoxes.length) qualityFlags.push('No people were detected. This does not establish that the classroom was empty.');
    else if (averageScore < 0.7) qualityFlags.push('Person detections have limited confidence. A manual review is recommended.');
    if (variable) qualityFlags.push(`The visible count changes across video samples (${Math.min(...samples)}–${Math.max(...samples)}). Review movement and obstruction.`);
    if (observations.some(item => item.atLimit)) qualityFlags.push('The detection limit was reached in a sample; the count may be incomplete.');
    if (kind === 'image') qualityFlags.push('A photo verifies one moment only; it cannot establish attendance throughout the class.');
    qualityFlags.push('People outside the frame or fully obscured cannot be counted. Trainers and visitors may be included.');
    const confidence = !count || lowResolution || variable || averageScore < 0.65
      ? 'low'
      : kind === 'video' && averageScore >= 0.82 && !smallPeople && !edgePeople ? 'high' : 'medium';
    const result = {
      count,
      confidence,
      boxes: representative.boxes,
      width,
      height,
      samples,
      qualityFlags,
      source: 'browser-model',
      model: 'COCO-SSD MobileNet',
      method: kind === 'video' ? `Median visible-person count from ${samples.length} sampled video frames; not a unique-person total.` : 'Visible-person detections in one photo; not individual attendance verification.',
      representativeTime: representative.time,
      sampleTimes: times,
      detectionThreshold: PERSON_THRESHOLD,
      confidenceNote: 'A heuristic based on detection scores and consistency; not a measured attendance accuracy.'
    };
    notify('complete', 100, `Analysis complete: ${count} visible ${count === 1 ? 'person' : 'people'} estimated.`);
    return result;
  } finally {
    if (kind === 'video') {
      media.pause();
      media.removeAttribute('src');
      media.load();
    } else {
      media.removeAttribute('src');
    }
    URL.revokeObjectURL(url);
    canvas.width = canvas.height = 1;
  }
}
