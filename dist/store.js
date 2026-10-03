/** Local prototype data. Seed observations are illustrative, never live inference. */
const STORAGE_KEY = 'centre-watch.state.v1';
const RECOVERY_KEY = 'centre-watch.state.recovery';
const DB_NAME = 'centre-watch-media';
const DB_VERSION = 1;
const MEDIA_STORE = 'files';

let memoryState = null;
let databasePromise = null;

const clone = (value) => JSON.parse(JSON.stringify(value));

export function uid(prefix = 'id') {
  const token = globalThis.crypto?.randomUUID?.()
    || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 11)}`;
  return `${prefix}-${token}`;
}

function localDate(date = new Date()) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function makeSeed() {
  const date = localDate();
  const timestamp = (time) => new Date(`${date}T${time}:00`).toISOString();
  const institutions = [
    { id: 'inst-01', name: 'Chennai Skill Development Centre', code: 'TN-CHN-024', city: 'Chennai, Tamil Nadu', head: 'Ananya Rao', email: 'ananya@chennai.example', capacity: 120, color: '#28685d' },
    { id: 'inst-02', name: 'Pune Technical Academy', code: 'MH-PUN-017', city: 'Pune, Maharashtra', head: 'Vikram Joshi', email: 'vikram@pune.example', capacity: 96, color: '#537baf' },
    { id: 'inst-03', name: 'Coimbatore Vocational Institute', code: 'TN-CBE-031', city: 'Coimbatore, Tamil Nadu', head: 'Meera Krishnan', email: 'meera@coimbatore.example', capacity: 80, color: '#a77540' },
  ];
  const trainers = [
    { id: 'tr-01', institutionId: 'inst-01', name: 'Kavya Iyer', email: 'kavya@chennai.example', specialty: 'Electrical installation' },
    { id: 'tr-02', institutionId: 'inst-01', name: 'Arjun Menon', email: 'arjun@chennai.example', specialty: 'Digital skills' },
    { id: 'tr-03', institutionId: 'inst-02', name: 'Neha Kulkarni', email: 'neha@pune.example', specialty: 'Mechanical assembly' },
    { id: 'tr-04', institutionId: 'inst-02', name: 'Rohan Deshmukh', email: 'rohan@pune.example', specialty: 'Welding techniques' },
    { id: 'tr-05', institutionId: 'inst-03', name: 'Priya Nair', email: 'priya@coimbatore.example', specialty: 'Apparel and tailoring' },
  ];
  const seedClass = ({ id, institutionId, trainerId, name, course, room, startTime, endTime, capacity, count, submittedCount, status, equipment }) => ({
    id,
    institutionId,
    trainerId,
    name,
    course,
    room,
    date,
    startTime,
    endTime,
    capacity,
    expectedEquipment: equipment,
    media: { id: null, name: 'Illustrated classroom sample', type: 'image/svg+xml', size: 0, source: 'sample' },
    submittedCount,
    submittedAt: submittedCount === null ? null : timestamp(startTime),
    observedCount: submittedCount === null ? null : count,
    status,
    analysis: submittedCount === null ? null : { source: 'sample', count, confidence: 'high', qualityFlags: [], model: 'Sample dataset' },
    review: null,
    sampleCount: count,
  });
  const classes = [
    seedClass({ id: 'cl-01', institutionId: 'inst-01', trainerId: 'tr-01', name: 'Electrical Installation', course: 'Electrician · Batch A', room: 'Workshop 01', startTime: '09:00', endTime: '11:00', capacity: 30, count: 28, submittedCount: 28, status: 'verified', equipment: [{ name: 'Workbenches', quantity: 7 }, { name: 'Electrical training kits', quantity: 14 }] }),
    seedClass({ id: 'cl-02', institutionId: 'inst-01', trainerId: 'tr-02', name: 'Digital Fundamentals', course: 'Data Entry Operator · Batch B', room: 'Computer Lab', startTime: '09:30', endTime: '11:30', capacity: 30, count: 20, submittedCount: 26, status: 'review', equipment: [{ name: 'Desktop computers', quantity: 30 }, { name: 'Chairs', quantity: 30 }] }),
    seedClass({ id: 'cl-03', institutionId: 'inst-01', trainerId: 'tr-01', name: 'Circuit Design Basics', course: 'Electrician · Batch C', room: 'Workshop 02', startTime: '13:00', endTime: '15:00', capacity: 28, count: 24, submittedCount: null, status: 'pending', equipment: [{ name: 'Workbenches', quantity: 7 }, { name: 'Multimeters', quantity: 14 }] }),
    seedClass({ id: 'cl-04', institutionId: 'inst-01', trainerId: 'tr-02', name: 'Workplace Communication', course: 'Employability Skills · Batch A', room: 'Classroom 03', startTime: '11:30', endTime: '12:30', capacity: 24, count: 22, submittedCount: 22, status: 'verified', equipment: [{ name: 'Chairs', quantity: 24 }, { name: 'Projector', quantity: 1 }] }),
    seedClass({ id: 'cl-05', institutionId: 'inst-02', trainerId: 'tr-03', name: 'Mechanical Assembly', course: 'Fitter · Batch A', room: 'Assembly Workshop', startTime: '09:00', endTime: '11:00', capacity: 28, count: 18, submittedCount: 24, status: 'review', equipment: [{ name: 'Workbenches', quantity: 7 }, { name: 'Bench vises', quantity: 14 }] }),
    seedClass({ id: 'cl-06', institutionId: 'inst-02', trainerId: 'tr-04', name: 'Welding Essentials', course: 'Welder · Batch B', room: 'Welding Workshop', startTime: '13:00', endTime: '15:00', capacity: 20, count: 16, submittedCount: null, status: 'pending', equipment: [{ name: 'Welding stations', quantity: 10 }, { name: 'Workbenches', quantity: 5 }] }),
    seedClass({ id: 'cl-07', institutionId: 'inst-03', trainerId: 'tr-05', name: 'Apparel Production', course: 'Sewing Machine Operator · Batch A', room: 'Apparel Lab', startTime: '09:00', endTime: '11:00', capacity: 24, count: 20, submittedCount: 20, status: 'verified', equipment: [{ name: 'Sewing machines', quantity: 24 }, { name: 'Cutting tables', quantity: 4 }] }),
    seedClass({ id: 'cl-08', institutionId: 'inst-03', trainerId: 'tr-05', name: 'Pattern Making', course: 'Self Employed Tailor · Batch B', room: 'Design Studio', startTime: '11:30', endTime: '13:30', capacity: 20, count: 18, submittedCount: 18, status: 'verified', equipment: [{ name: 'Cutting tables', quantity: 5 }, { name: 'Chairs', quantity: 20 }] }),
  ];
  return {
    version: 1,
    institutions,
    trainers,
    classes,
    activity: [
      { id: 'act-01', type: 'attendance', message: 'Sample comparison: Digital Fundamentals needs review — 26 submitted, 20 observed.', institutionId: 'inst-01', classId: 'cl-02', time: timestamp('09:35') },
      { id: 'act-02', type: 'attendance', message: 'Sample comparison: Electrical Installation counts match at 28.', institutionId: 'inst-01', classId: 'cl-01', time: timestamp('09:05') },
      { id: 'act-03', type: 'attendance', message: 'Sample comparison: Mechanical Assembly needs review — 24 submitted, 18 observed.', institutionId: 'inst-02', classId: 'cl-05', time: timestamp('09:05') },
      { id: 'act-04', type: 'class', message: 'Circuit Design Basics is ready for attendance submission.', institutionId: 'inst-01', classId: 'cl-03', time: timestamp('08:50') },
      { id: 'act-05', type: 'system', message: 'Demo workspace prepared with 3 institutions and 8 sample classes.', institutionId: null, classId: null, time: timestamp('08:30') },
    ],
    ui: { role: 'head', institutionId: 'inst-01', trainerId: 'tr-01', page: 'overview' },
  };
}

function normalizeState(value) {
  if (!value || typeof value !== 'object' || (value.version !== undefined && value.version !== 1)) {
    throw new Error('Unsupported workspace data');
  }
  for (const key of ['institutions', 'trainers', 'classes']) {
    if (!Array.isArray(value[key]) || value[key].some((entry) => !entry || typeof entry.id !== 'string')) {
      throw new Error(`Invalid ${key} data`);
    }
  }
  const state = clone(value);
  state.version = 1;
  state.activity = Array.isArray(state.activity) ? state.activity : [];
  state.ui = { role: 'head', institutionId: state.institutions[0]?.id || '', trainerId: state.trainers[0]?.id || '', page: 'overview', ...state.ui };
  if (!['admin', 'head', 'trainer'].includes(state.ui.role)) state.ui.role = 'head';
  state.classes = state.classes.map((entry) => ({
    expectedEquipment: [],
    submittedCount: null,
    submittedAt: null,
    observedCount: null,
    analysis: null,
    review: null,
    ...entry,
    status: ['verified', 'review', 'pending', 'unverified', 'reviewed', 'processing'].includes(entry.status) ? entry.status : 'pending',
  }));
  return state;
}

function persist(state) {
  memoryState = clone(state);
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(state));
    return true;
  } catch {
    // Private-browsing/storage restrictions still allow an in-memory demo.
    return false;
  }
}

export function loadState() {
  if (memoryState) return clone(memoryState);
  let raw = null;
  try {
    raw = globalThis.localStorage?.getItem(STORAGE_KEY);
    if (raw) {
      memoryState = normalizeState(JSON.parse(raw));
      return clone(memoryState);
    }
  } catch {
    // Retain one recoverable copy when storage contains malformed or old data.
    if (raw) {
      try { globalThis.localStorage?.setItem(RECOVERY_KEY, raw); } catch { /* Storage is unavailable. */ }
    }
  }
  const state = makeSeed();
  persist(state);
  return clone(state);
}

/** Returns false if browser storage is unavailable; data remains in memory. */
export function saveState(state) {
  return persist(normalizeState(state));
}

/** Reset workspace records. Call clearMedia() too to remove uploaded blobs. */
export function resetState() {
  const state = makeSeed();
  persist(state);
  return clone(state);
}

function openDatabase() {
  if (!globalThis.indexedDB) return Promise.reject(new Error('This browser cannot save uploads locally.'));
  if (databasePromise) return databasePromise;
  databasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(MEDIA_STORE)) request.result.createObjectStore(MEDIA_STORE);
    };
    request.onerror = () => reject(request.error || new Error('Unable to open local media storage.'));
    request.onblocked = () => reject(new Error('Close other prototype tabs and try uploading again.'));
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => { db.close(); databasePromise = null; };
      resolve(db);
    };
  }).catch((error) => { databasePromise = null; throw error; });
  return databasePromise;
}

async function mediaOperation(mode, operation) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(MEDIA_STORE, mode);
    let result;
    const request = operation(transaction.objectStore(MEDIA_STORE));
    request.onsuccess = () => { result = request.result; };
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error || request.error || new Error('Media could not be saved.'));
    transaction.onabort = () => reject(transaction.error || new Error('Media storage was interrupted.'));
  });
}

export async function saveMedia(file) {
  if (!(file instanceof Blob)) throw new TypeError('Choose a photo or video file.');
  const id = uid('media');
  await mediaOperation('readwrite', (store) => store.put(file, id));
  return { id, name: file.name || 'Classroom media', type: file.type || 'application/octet-stream', size: file.size };
}

export async function getMedia(id) {
  if (!id) return null;
  const value = await mediaOperation('readonly', (store) => store.get(id));
  return value instanceof Blob ? value : null;
}

export async function deleteMedia(id) {
  if (!id) return;
  await mediaOperation('readwrite', (store) => store.delete(id));
}

export async function clearMedia() {
  await mediaOperation('readwrite', (store) => store.clear());
}
