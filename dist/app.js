import { loadState, saveState, resetState, uid, saveMedia, getMedia, deleteMedia, clearMedia } from './store.js';
import { analyzeMedia } from './vision.js';

const $ = (s, el = document) => el.querySelector(s);
const escape = (s = '') => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; };
const dateLabel = (d = today()) => new Date(`${d}T12:00:00`).toLocaleDateString('en-IN', { weekday:'short', day:'numeric', month:'long', year:'numeric' });
const timeLabel = (t) => t ? new Date(t).toLocaleTimeString('en-IN', {hour:'2-digit',minute:'2-digit'}) : '—';
const initials = s => String(s || '').split(' ').filter(Boolean).slice(0,2).map(x => x[0]).join('');
const paths = {
  overview:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  building:'<path d="M4 21V5l8-3 8 3v16M2 21h20M9 21v-4h6v4M8 7h1m6 0h1M8 11h1m6 0h1"/>',
  classes:'<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4M7 8h10M7 12h6"/>',
  people:'<circle cx="9" cy="7" r="3"/><path d="M3 21v-2a6 6 0 0 1 12 0v2M16 4a3 3 0 0 1 0 6m2 5a5 5 0 0 1 3 4v2"/>',
  check:'<path d="m5 12 4 4L19 6"/>',
  shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
  alert:'<path d="m10.3 4.7-8 14A1.5 1.5 0 0 0 3.6 21h16.8a1.5 1.5 0 0 0 1.3-2.3l-8-14a1.6 1.6 0 0 0-2.8 0Z"/><path d="M12 9v4m0 4h.01"/>',
  report:'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8m-8 4h5"/>',
  calendar:'<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18m-13 4h2m4 0h2"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',
  arrow:'<path d="M5 12h14m-5-5 5 5-5 5"/>',
  down:'<path d="m6 9 6 6 6-6"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',
  search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/>',
  bell:'<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
  upload:'<path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
  download:'<path d="M12 3v13m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  camera:'<path d="M8 5 6 8H3v12h18V8h-3l-2-3Z"/><circle cx="12" cy="13" r="3.5"/>',
  play:'<path d="m9 5 11 7-11 7Z"/>',
  clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  scan:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M6 12h12"/>',
  menu:'<path d="M4 6h16M4 12h16M4 18h16"/>',
  logout:'<path d="M9 5H4v14h5m5-14 7 7-7 7M9 12h12"/>',
  image:'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 4-6 5 7"/>',
  video:'<rect x="3" y="5" width="13" height="14" rx="2"/><path d="m16 10 5-3v10l-5-3"/>',
};
const icon = (name, cls = '') => `<svg class="icon ${cls}" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.classes}</svg>`;
let state = loadState();
let search = '', statusFilter = 'all', reportDate = today(), activeModal = null, uploading = false;
let draftMedia = null, draftURL = null, detailURL = null;
const jobs = new Map();
const roleLabels = { admin:'Administrator', head:'Institution head', trainer:'Trainer' };
const navNames = { overview:'Overview', institutions:'Institutions', classes:'Classrooms', attendance:'My attendance', trainers:'Trainers', reviews:'Review queue', reports:'Daily report' };
let recovered = false;
for (const c of state.classes) if (c.status === 'processing') { c.status = 'unverified'; c.analysis = { source:'interrupted', qualityFlags:['Analysis was interrupted. Run verification again.'] }; recovered = true; }
if (recovered) saveState(state);
const inst = id => state.institutions.find(i => i.id === (id || state.ui.institutionId));
const trainer = id => state.trainers.find(t => t.id === (id || state.ui.trainerId));
const classesFor = () => state.classes.filter(c => state.ui.role === 'admin' || (c.institutionId === state.ui.institutionId && (state.ui.role !== 'trainer' || c.trainerId === state.ui.trainerId)));
const unresolved = c => ['review','unverified'].includes(c.status);
const scopeActivity = () => state.activity.filter(a => state.ui.role === 'admin' || (a.institutionId === state.ui.institutionId && (state.ui.role !== 'trainer' || classesFor().some(c => c.id === a.classId))));
function persist() { try { if (!saveState(state)) toast('Your browser could not save these changes. Keep this tab open.', 'error'); } catch { toast('Your browser could not save these changes. Keep this tab open.', 'error'); } }
function log(type, message, classId, institutionId) { state.activity.unshift({id:uid('event'),type,message,classId:classId || null,institutionId:institutionId || state.ui.institutionId,time:new Date().toISOString()}); state.activity = state.activity.slice(0,150); }
function toast(message, type = 'success') { const el = document.createElement('div'); el.className = `toast ${type}`; el.setAttribute('role','status'); el.innerHTML = `${icon(type==='error'?'alert':'check')}<span>${escape(message)}</span>`; $('#toast-root').append(el); setTimeout(() => el.remove(), 5500); }
const count = v => v == null ? '—' : String(v);
const sum = (items, key) => items.reduce((n,c) => n + (Number(c[key]) || 0), 0);
function badge(c) {
  const names = { verified:'Count aligned', review:'Needs review', pending:'Not submitted', unverified:'Unable to verify', processing:'Analyzing', reviewed:'Reviewed' };
  return `<span class="badge ${c.status === 'reviewed' ? 'neutral' : c.status}"><span class="badge-dot"></span>${names[c.status] || 'Pending'}</span>`;
}
function navigate(page) { state.ui.page = page; search = ''; statusFilter = 'all'; persist(); document.body.classList.remove('nav-open'); render(); window.scrollTo(0,0); }
function persona() { return state.ui.role==='admin' ? {name:'Arjun Mehta',caption:'Scheme monitoring unit'} : state.ui.role==='head' ? {name:inst()?.head || 'Institution head',caption:inst()?.name || ''} : {name:trainer()?.name || 'Trainer',caption:trainer()?.specialty || 'Class trainer'}; }
function render() {
  const role = state.ui.role, p = persona(), reviewCount = classesFor().filter(unresolved).length;
  const nav = role==='admin' ? ['overview','institutions','classes','reviews','reports'] : role==='head' ? ['overview','classes','trainers','reviews','reports'] : ['overview','attendance','classes'];
  if (!nav.includes(state.ui.page)) state.ui.page='overview';
  const navIcons = {overview:'overview',institutions:'building',classes:'classes',attendance:'people',trainers:'people',reviews:'shield',reports:'report'};
  $('#app').innerHTML = `<div class="app-shell">
    <button class="mobile-overlay" data-action="close-nav" aria-label="Close navigation"></button>
    <aside class="sidebar" aria-label="Main navigation">
      <a href="#overview" class="brand" data-page="overview"><span class="brand-mark">${icon('scan')}</span><span class="brand-text">CentreWatch<span>TRAINING ASSURANCE</span></span></a>
      <div class="workspace-card"><span class="workspace-symbol">${icon(role==='admin'?'building':'classes')}</span><div><span class="workspace-title">${role==='admin'?'National workspace':escape(inst()?.city || 'Centre workspace')}</span><span class="workspace-caption">${role==='admin'?'All institutions':'Institution workspace'}</span></div><span class="workspace-dot"></span></div>
      <div class="nav-section-label">WORKSPACE</div><nav class="nav-list">${nav.map(n=>`<a href="#${n}" class="nav-item ${state.ui.page===n?'active':''}" data-page="${n}" ${state.ui.page===n?'aria-current="page"':''}><span class="nav-icon">${icon(navIcons[n])}</span><span>${navNames[n]}</span>${n==='reviews' && reviewCount ? `<span class="nav-count">${reviewCount}</span>`:''}</a>`).join('')}</nav>
      <div class="sidebar-bottom"><div class="sidebar-help"><span class="help-orbit">${icon('shield')}</span><strong>Clarity in every count.</strong><p>See attendance and verification in one place.</p><button class="button ghost small" data-action="guide">Explore the prototype ${icon('arrow')}</button></div><div class="prototype-badge"><span></span>INTERACTIVE PROTOTYPE</div><div class="user-card"><span class="avatar">${escape(initials(p.name))}</span><div><strong>${escape(p.name)}</strong><span>${roleLabels[role]}</span></div><button class="icon-button" data-action="guide" aria-label="Prototype information">${icon('info')}</button></div></div>
    </aside>
    <div class="main-shell"><header class="topbar"><div class="breadcrumb"><button class="icon-button mobile-menu-button" data-action="open-nav" aria-label="Open navigation">${icon('menu')}</button><span>Workspace</span><span class="breadcrumb-separator">/</span><strong>${navNames[state.ui.page]}</strong></div><div class="topbar-actions"><span class="demo-switch-label">VIEW AS</span><label class="role-switcher">${icon(role==='admin'?'building':role==='head'?'shield':'people')}<select id="role-select" aria-label="Switch demo role"><option value="head" ${role==='head'?'selected':''}>Institution head</option><option value="trainer" ${role==='trainer'?'selected':''}>Trainer</option><option value="admin" ${role==='admin'?'selected':''}>Administrator</option></select>${icon('down')}</label><button class="icon-button notification-button" data-action="notifications" aria-label="View notifications">${icon('bell')}${reviewCount?'<span class="notification-dot"></span>':''}</button><span class="avatar avatar-small">${escape(initials(p.name))}</span></div></header>
    <main id="main-content" class="page-content" tabindex="-1">${pageContent()}</main><footer class="app-footer"><span>CentreWatch <span class="footer-dot">·</span> Every classroom, accounted for.</span><span>${icon('shield')} Media stays in this browser</span></footer></div></div>`;
}
function heading(eyebrow, title, description, actions='') { return `<section class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p class="page-description">${description}</p></div><div class="heading-actions">${actions}</div></section>`; }
function contextControls() {
  if(state.ui.role==='admin')return '';
  return `<div class="context-controls"><label>Demo institution<select id="institution-select" class="filter-select" aria-label="Select demo institution">${state.institutions.map(i=>`<option value="${i.id}" ${i.id===state.ui.institutionId?'selected':''}>${escape(i.name)}</option>`).join('')}</select></label>${state.ui.role==='trainer'?`<label>Demo trainer<select id="trainer-select" class="filter-select" aria-label="Select demo trainer">${state.trainers.filter(t=>t.institutionId===state.ui.institutionId).map(t=>`<option value="${t.id}" ${t.id===state.ui.trainerId?'selected':''}>${escape(t.name)}</option>`).join('')}</select></label>`:''}</div>`;
}
function stat(label,value,foot,ico,tone='teal') { return `<article class="stat-card"><div class="stat-top"><span class="stat-label">${label}</span><span class="stat-icon ${tone}">${icon(ico)}</span></div><strong class="stat-value">${value}</strong><div class="stat-foot">${foot}</div></article>`; }
function pageContent() {
  if(state.ui.page==='overview') return overview();
  if(state.ui.page==='classes'||state.ui.page==='attendance')return classPage();
  if(state.ui.page==='institutions')return institutionPage();
  if(state.ui.page==='trainers')return trainerPage();
  if(state.ui.page==='reviews')return reviewsPage();
  return reportPage();
}
function overview() {
  const all = classesFor(), items = all.filter(c=>c.date===today()), role = state.ui.role;
  const submitted=items.filter(c=>c.submittedCount!=null), analyzed=items.filter(c=>c.observedCount!=null), aligned=items.filter(c=>c.status==='verified');
  const attention=all.filter(unresolved), title = role==='admin'?'Every centre. One clear picture.':role==='trainer'?`Let's make today count, ${escape(persona().name.split(' ')[0])}.`:'Your centre, at a glance.';
  const desc=role==='admin'?'A connected view of attendance and verification across your institutions.':role==='trainer'?"Your classes, attendance submissions, and verification updates.":"Keep a pulse on your classrooms. Focus on what needs your attention.";
  const actions = role==='admin'?`<button class="button secondary" data-action="export">${icon('download')} Export report</button><button class="button primary" data-action="new-institution">${icon('plus')} Add institution</button>`:role==='head'?`<button class="button secondary" data-page="reports">${icon('report')} Daily report</button><button class="button primary" data-action="new-class">${icon('plus')} Create class</button>`:`<button class="button primary" data-page="attendance">${icon('people')} Mark attendance</button>`;
  const focus=attention[0]||items.find(c=>c.status==='pending')||items[0];
  return heading(dateLabel().toUpperCase(),title,desc,actions)+contextControls()+`<section class="stats-grid" aria-label="Today's summary">${stat(role==='admin'?'Institutions':'Classes today',role==='admin'?state.institutions.length:items.length,role==='admin'?`${items.length} classes scheduled today`:`${submitted.length} attendance submissions`,'classes')}${stat('Reported attendance',sum(submitted,'submittedCount'),`Across ${submitted.length} submitted classes`,'people','blue')}${stat('Counts aligned',analyzed.length?`${Math.round(aligned.length/analyzed.length*100)}<small>%</small>`:'—',`${aligned.length} of ${analyzed.length} analyzed classes`,'shield')}${stat('Needs attention',attention.length,attention.length?'Review discrepancies and unclear views':'All submitted checks are up to date','alert','amber')}</section>
  <section class="dashboard-grid"><article class="card verification-card"><div class="card-header"><div><h2 class="card-title">Attendance, in perspective</h2><p class="card-subtitle">Today's reported and observed counts</p></div><span class="badge neutral">${icon('calendar')} Today</span></div>${comparisonChart(items)}<div class="chart-bottom-note">${icon('info')} Observed counts estimate visible people, not individual identities.</div></article><article class="card"><div class="card-header"><div><h2 class="card-title">${role==='trainer'?'Your next steps':'Attention board'}</h2><p class="card-subtitle">${role==='trainer'?'Keep your classes up to date':'The right follow-up, at the right time'}</p></div><span class="counter-circle">${role==='trainer'?items.filter(c=>c.status==='pending').length:attention.length}</span></div><div class="activity-list attention-list">${(role==='trainer'?items.filter(c=>c.status==='pending'):attention).slice(0,3).map(c=>`<button class="attention-item" data-action="${role==='trainer'?'attendance':'detail'}" data-id="${c.id}"><span class="attention-icon ${c.status==='pending'?'teal':'amber'}">${icon(c.status==='pending'?'clock':'alert')}</span><span class="activity-content"><strong>${escape(c.name)}</strong><span>${c.status==='pending'?'Attendance is ready to submit':c.status==='unverified'?'The media needs a closer look':`${c.submittedCount} reported · ${count(c.observedCount)} observed`}</span><small>${escape(inst(c.institutionId)?.city)} · ${escape(c.room)}</small></span>${icon('arrow')}</button>`).join('')||`<div class="empty-state compact">${icon('shield')}<strong>You're all caught up</strong><p>No classes need your attention.</p></div>`}</div><div class="card-bottom"><button class="button ghost full-width" data-page="${role==='trainer'?'attendance':'reviews'}">${role==='trainer'?'View my attendance':'Open review queue'} ${icon('arrow')}</button></div></article></section>
  <section class="card mt-24"><div class="card-header"><div><h2 class="card-title">Today's classrooms</h2><p class="card-subtitle">A shared record from submission to review</p></div><button class="button ghost small" data-page="classes">View all classes ${icon('arrow')}</button></div>${classTable(items.slice(0,5))}</section>
  <section class="dashboard-grid mt-24"><article class="card monitor-card"><div class="card-header"><div><h2 class="card-title">Classroom spotlight</h2><p class="card-subtitle">${focus?escape(focus.name):'Your next class will appear here'}</p></div><span class="badge neutral">${focus?.media?.source==='sample'?'Sample scene':'Media overview'}</span></div>${focus?`${scene(focus)}<div class="monitor-footer"><div><span class="monitor-footer-label">REPORTED</span><strong>${count(focus.submittedCount)}</strong></div><div><span class="monitor-footer-label">OBSERVED</span><strong>${count(focus.observedCount)}</strong></div><button class="button secondary small" data-action="detail" data-id="${focus.id}">View class ${icon('arrow')}</button></div>`:'<div class="empty-state">No classes yet.</div>'}</article><article class="card"><div class="card-header"><div><h2 class="card-title">Recent activity</h2><p class="card-subtitle">The latest across your workspace</p></div>${icon('clock')}</div><div class="activity-list">${scopeActivity().slice(0,4).map(a=>`<div class="activity-item"><span class="activity-dot ${a.type==='review'?'amber':''}"></span><div class="activity-content"><p>${escape(a.message)}</p><span class="activity-time">${timeLabel(a.time)}</span></div></div>`).join('')||'<div class="empty-state compact">Activity will appear as you use the prototype.</div>'}</div></article></section>`;
}
function comparisonChart(items) {
  const values=items.slice(0,7), max=Math.max(10,...values.flatMap(c=>[c.submittedCount||0,c.observedCount||0]));
  if(!values.length)return '<div class="empty-state">No classes scheduled today.</div>';
  return `<div class="chart-legend"><span><i class="legend-dot submitted"></i>Trainer submitted</span><span><i class="legend-dot observed"></i>Visually observed</span></div><div class="comparison-chart"><div class="chart-scale"><span>${Math.ceil(max/10)*10}</span><span>${Math.ceil(max/10)*5}</span><span>0</span></div><div class="bar-chart">${values.map((c,i)=>`<div class="bar-group" title="${escape(c.name)}: ${count(c.submittedCount)} submitted, ${count(c.observedCount)} observed"><div class="bar-pair"><div class="bar submitted" style="height:${(c.submittedCount||0)/(Math.ceil(max/10)*10)*100}%"><span>${count(c.submittedCount)}</span></div><div class="bar observed ${unresolved(c)?'mismatch':''}" style="height:${(c.observedCount||0)/(Math.ceil(max/10)*10)*100}%"><span>${count(c.observedCount)}</span></div></div><span class="bar-label">${escape(c.room)}</span></div>`).join('')}</div></div>`;
}
function scene(c) {
  if(c.media?.source!=='sample')return `<div class="uploaded-scene">${icon(c.media?.type?.startsWith('video')?'video':'image')}<strong>${escape(c.media?.name||'Uploaded classroom media')}</strong><span>Private browser upload</span><button class="button secondary small" data-action="detail" data-id="${c.id}">Open media ${icon('arrow')}</button></div>`;
  const n=c.observedCount??c.sampleCount??18;
  return `<div class="monitor-scene"><div class="monitor-toolbar"><span>${icon('camera')} ${escape(c.room)} <span class="scene-divider">/</span> FRONT VIEW</span><span class="sample-label">ILLUSTRATIVE SAMPLE</span></div><div class="classroom-stage"><div class="classroom-board"><span>LEARN. PRACTISE. GROW.</span><i></i></div><div class="classroom-seats">${Array.from({length:30},(_,i)=>`<div class="seat ${i<n?'occupied':''}">${i<n?`<span class="person-mark"><i></i></span>`:''}<span class="desk"></span></div>`).join('')}</div><div class="stage-corner tl"></div><div class="stage-corner tr"></div><div class="stage-corner bl"></div><div class="stage-corner br"></div></div><div class="scene-caption"><span><i></i> ${n} sample occupants</span><span>Demo illustration · not camera footage</span></div></div>`;
}
function classTable(items, includeInstitution=state.ui.role==='admin') {
  if(!items.length)return `<div class="empty-state">${icon('classes')}<strong>No classes to show</strong><p>Create a class or adjust your filters.</p></div>`;
  return `<div class="table-wrap"><table class="data-table"><thead><tr><th>CLASS / SESSION</th>${includeInstitution?'<th>INSTITUTION</th>':''}<th>TRAINER</th><th>SUBMITTED</th><th>OBSERVED</th><th>STATUS</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>${items.map(c=>`<tr><td><button class="table-link" data-action="detail" data-id="${c.id}"><span class="cell-title">${escape(c.name)}</span><span class="cell-subtitle">${escape(c.room)} <span>·</span> ${escape(c.startTime)}–${escape(c.endTime)}</span></button></td>${includeInstitution?`<td><span class="cell-title">${escape(inst(c.institutionId)?.name)}</span><span class="cell-subtitle">${escape(inst(c.institutionId)?.city)}</span></td>`:''}<td><span class="trainer-cell"><span class="avatar avatar-tiny">${escape(initials(trainer(c.trainerId)?.name))}</span>${escape(trainer(c.trainerId)?.name||'Unassigned')}</span></td><td><strong class="count-value">${count(c.submittedCount)}</strong></td><td><strong class="count-value ${unresolved(c)?'text-amber':'text-teal'}">${count(c.observedCount)}</strong>${c.analysis?.source==='sample'?'<span class="cell-subtitle">Sample result</span>':''}</td><td>${badge(c)}</td><td><button class="icon-button" data-action="${state.ui.role==='trainer'&&c.status==='pending'?'attendance':'detail'}" data-id="${c.id}" aria-label="Open ${escape(c.name)}">${icon('arrow')}</button></td></tr>`).join('')}</tbody></table></div>`;
}
function classPage() {
  const role=state.ui.role, attendance=state.ui.page==='attendance';
  return heading('CLASSROOM WORKSPACE',attendance?'Your attendance, all in one place.':'A space for every session.',attendance?'Submit a count. Get a visual check. Keep a clear record.':'Manage sessions, classroom media, and attendance verification.',role==='head'?`<button class="button primary" data-action="new-class">${icon('plus')} Create class</button>`:'')+contextControls()+`<div class="toolbar"><label class="search-field">${icon('search')}<input id="class-search" placeholder="Search classes, rooms or trainers…" aria-label="Search classes" value="${escape(search)}" /></label><select id="status-filter" class="filter-select" aria-label="Filter verification status"><option value="all">All statuses</option><option value="pending">Not submitted</option><option value="verified">Count aligned</option><option value="review">Needs review</option><option value="unverified">Unable to verify</option><option value="reviewed">Reviewed</option></select><span class="results-count" id="results-count"></span></div><div id="class-results">${classResults()}</div>`;
}
function filteredClasses() { return classesFor().filter(c=>(statusFilter==='all'||c.status===statusFilter) && `${c.name} ${c.room} ${trainer(c.trainerId)?.name||''}`.toLowerCase().includes(search.toLowerCase())); }
function classResults() {
  const items=filteredClasses();
  if(!items.length)return `<div class="card empty-state">${icon('classes')}<strong>No classes found</strong><p>Try another search or create your first classroom session.</p>${state.ui.role==='head'?'<button class="button primary" data-action="new-class">Create class</button>':''}</div>`;
  if(state.ui.page==='attendance') return `<section class="card">${classTable(items)}</section><div class="info-note mt-16">${icon('info')} Your submission is saved before verification. The camera estimate never changes your original count.</div>`;
  return `<div class="class-grid">${items.map(c=>`<article class="class-card"><div class="class-cover ${c.media?.source==='sample'?'sample-cover':'upload-cover'}"><span class="cover-grid"></span><span class="course-icon">${icon(c.course?.toLowerCase().includes('digital')?'classes':c.course?.toLowerCase().includes('electric')?'scan':'building')}</span><span class="cover-label">${escape(c.course||'Skills training')}</span><span class="badge neutral cover-badge">${icon(c.media?.source==='sample'?'camera':c.media?.type?.startsWith('video')?'video':'image')} ${c.media?.source==='sample'?'Sample media':'Uploaded media'}</span></div><div class="class-card-body"><div class="class-card-top">${badge(c)}<span class="cell-subtitle">${escape(c.date===today()?'Today':c.date)}</span></div><h2>${escape(c.name)}</h2><div class="class-meta"><span>${icon('classes')} ${escape(c.room)}</span><span>${icon('clock')} ${escape(c.startTime)}–${escape(c.endTime)}</span></div><div class="class-trainer"><span class="avatar avatar-tiny">${escape(initials(trainer(c.trainerId)?.name))}</span><span>${escape(trainer(c.trainerId)?.name||'Unassigned')}</span></div><div class="class-counts"><div><span>SUBMITTED</span><strong>${count(c.submittedCount)}</strong></div><div><span>OBSERVED</span><strong class="${unresolved(c)?'text-amber':'text-teal'}">${count(c.observedCount)}</strong></div><div><span>CAPACITY</span><strong>${c.capacity}</strong></div></div><div class="class-actions"><button class="button ${state.ui.role==='trainer'&&c.status==='pending'?'primary':'secondary'} full-width" data-action="${state.ui.role==='trainer'&&c.status==='pending'?'attendance':'detail'}" data-id="${c.id}">${state.ui.role==='trainer'&&c.status==='pending'?'Mark attendance':'View classroom'} ${icon('arrow')}</button></div></div></article>`).join('')}</div>`;
}
function institutionPage() {
  return heading('CONNECTED INSTITUTIONS','A network you can see clearly.','Create institutions, appoint their heads, and follow their daily progress.',`<button class="button primary" data-action="new-institution">${icon('plus')} Add institution</button>`)+`<div class="institution-grid">${state.institutions.map(i=>{const c=state.classes.filter(c=>c.institutionId===i.id), done=c.filter(c=>c.status==='verified').length;return `<article class="institution-card"><div class="institution-card-top"><span class="institution-icon" style="--institution-color:${escape(i.color||'#147d68')}">${icon('building')}</span><span class="badge neutral">${escape(i.code)}</span></div><h2>${escape(i.name)}</h2><p class="text-muted">${escape(i.city)}</p><div class="institution-meta"><span>Institution head</span><strong>${escape(i.head)}</strong><span>${escape(i.email)}</span></div><div class="institution-metrics"><div><strong>${c.length}</strong><span>Classes</span></div><div><strong>${state.trainers.filter(t=>t.institutionId===i.id).length}</strong><span>Trainers</span></div><div><strong class="text-amber">${c.filter(unresolved).length}</strong><span>Need review</span></div></div><div class="progress-caption"><span>Counts aligned</span><strong>${done}/${c.length}</strong></div><div class="progress-track"><div class="progress-fill" style="width:${c.length?done/c.length*100:0}%"></div></div><button class="button secondary full-width mt-24" data-action="open-institution" data-id="${i.id}">Open institution workspace ${icon('arrow')}</button></article>`;}).join('')}</div>`;
}
function trainerPage() {
  const trainers=state.trainers.filter(t=>t.institutionId===state.ui.institutionId);
  return heading('YOUR TRAINING TEAM','Good learning starts with people.','Add trainers and assign them when you create a class.',`<button class="button primary" data-action="new-trainer">${icon('plus')} Add trainer</button>`)+contextControls()+`<section class="card"><div class="table-wrap"><table class="data-table"><thead><tr><th>TRAINER</th><th>SPECIALTY</th><th>ASSIGNED CLASSES</th><th>SUBMISSIONS</th><th>WORKSPACE</th></tr></thead><tbody>${trainers.map(t=>{const c=state.classes.filter(c=>c.trainerId===t.id);return `<tr><td><div class="institution-cell"><span class="avatar">${escape(initials(t.name))}</span><div><span class="cell-title">${escape(t.name)}</span><span class="cell-subtitle">${escape(t.email)}</span></div></div></td><td>${escape(t.specialty)}</td><td><strong>${c.length}</strong></td><td>${c.filter(c=>c.submittedCount!=null).length} / ${c.length}</td><td><button class="button ghost small" data-action="open-trainer" data-id="${t.id}">Preview trainer view ${icon('arrow')}</button></td></tr>`;}).join('')||'<tr><td colspan="5"><div class="empty-state">Add your first trainer to start assigning classes.</div></td></tr>'}</tbody></table></div></section>`;
}
function reviewsPage() {
  const items=classesFor().filter(unresolved), reviewed=classesFor().filter(c=>c.review?.decision==='accepted');
  return heading('HUMAN REVIEW','A little context makes a difference.','Inspect discrepancies, understand limitations, and record a fair decision.')+contextControls()+`<div class="alert-banner">${icon('info')}<div><strong>Every flag is a request to review.</strong><span>A count mismatch does not establish absence or misconduct. Keep context alongside every decision.</span></div></div><section class="card mt-24"><div class="card-header"><div><h2 class="card-title">Open reviews <span class="inline-count">${items.length}</span></h2><p class="card-subtitle">Attendance gaps and observations that need more context</p></div></div>${items.length?classTable(items):'<div class="empty-state">'+icon('shield')+'<strong>Nothing waiting for review</strong><p>New discrepancies will appear here.</p></div>'}</section>${reviewed.length?`<section class="card mt-24"><div class="card-header"><h2 class="card-title">Reviewed with context</h2></div>${classTable(reviewed)}</section>`:''}`;
}
function reportPage() {
  const items=classesFor().filter(c=>c.date===reportDate), groups=state.institutions.filter(i=>items.some(c=>c.institutionId===i.id));
  return heading('DAILY INSIGHTS','The whole day, accounted for.','A clear institution-by-institution record of submissions and visual checks.',`<button class="button primary" data-action="export">${icon('download')} Download CSV</button>`)+contextControls()+`<div class="report-controls"><label class="report-date">${icon('calendar')}<input id="report-date" type="date" value="${reportDate}" aria-label="Report date" /></label><span class="badge neutral">${items.length} classes · ${groups.length} institutions</span><span class="text-muted">Live summary · updates as submissions arrive</span></div><div class="stats-grid">${stat('Classes scheduled',items.length,`${groups.length} institutions represented`,'classes')}${stat('Trainer submitted',sum(items,'submittedCount'),`${items.filter(c=>c.submittedCount!=null).length} submissions received`,'people','blue')}${stat('Visually observed',sum(items,'observedCount'),`${items.filter(c=>c.observedCount!=null).length} classes analyzed`,'scan')}${stat('Open reviews',items.filter(unresolved).length,`${items.filter(c=>c.status==='pending').length} classes awaiting attendance`,'alert','amber')}</div>${groups.map(i=>{const list=items.filter(c=>c.institutionId===i.id);return `<section class="card mt-24"><div class="card-header"><div class="institution-cell"><span class="institution-icon">${icon('building')}</span><div><h2 class="card-title">${escape(i.name)}</h2><p class="card-subtitle">${escape(i.city)} · ${list.length} classes · ${escape(i.head)}</p></div></div><span class="badge ${list.some(unresolved)?'review':'verified'}">${list.filter(unresolved).length} open reviews</span></div>${classTable(list,false)}</section>`;}).join('')||'<div class="card empty-state">'+icon('calendar')+'<strong>No classes on this date</strong><p>Choose another day to view its report.</p></div>'}<div class="info-note mt-16">${icon('info')}Totals represent attendance instances across classes, not unique trainees. Sample results are labelled in the report.</div>`;
}

function showModal(title, subtitle, body, footer='', cls='') {
  $('#modal-root').innerHTML=`<dialog class="modal ${cls}" aria-labelledby="modal-title"><div class="modal-header"><div><h2 id="modal-title">${title}</h2><p>${subtitle}</p></div><button class="icon-button modal-close" data-action="close-modal" aria-label="Close dialog">${icon('close')}</button></div><div class="modal-body">${body}</div>${footer?`<div class="modal-footer">${footer}</div>`:''}</dialog>`;
  const dialog=$('dialog'); dialog.showModal(); dialog.addEventListener('cancel',e=>{e.preventDefault();closeModal();}); dialog.addEventListener('click',e=>{if(e.target===dialog){const b=dialog.getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)closeModal();}});
}
async function closeModal() {
  if(uploading) { toast('Please wait for the upload to finish.','error');return; }
  if(draftMedia) { await deleteMedia(draftMedia.id).catch(()=>{});draftMedia=null; }
  if(draftURL){URL.revokeObjectURL(draftURL);draftURL=null;}
  if(detailURL){URL.revokeObjectURL(detailURL);detailURL=null;}
  activeModal=null; $('dialog')?.close(); $('#modal-root').innerHTML='';
}
const inputField=(label,name,placeholder='',type='text',extra='')=>`<div class="form-field"><label class="form-label" for="field-${name}">${label}</label><input class="form-input" id="field-${name}" name="${name}" type="${type}" placeholder="${placeholder}" ${extra} required /></div>`;
function newClass() {
  if(state.ui.role!=='head')return;
  const trainers=state.trainers.filter(t=>t.institutionId===state.ui.institutionId);
  if(!trainers.length){toast('Add a trainer before creating a class.','error');navigate('trainers');return;}
  activeModal={type:'create-class'};draftMedia=null;
  showModal('Create a classroom session','Give your trainer a class, a schedule, and a view of the room.',`<form id="class-form"><div class="form-grid">${inputField('Class name','name','e.g. Electrical Workshop — Batch B','text','maxlength="80"')}${inputField('Training course','course','e.g. Electrical installation','text','maxlength="80"')}${inputField('Room','room','e.g. Workshop 02','text','maxlength="35"')}<div class="form-field"><label class="form-label" for="field-trainerId">Assigned trainer</label><select id="field-trainerId" name="trainerId" class="form-select" required>${trainers.map(t=>`<option value="${t.id}">${escape(t.name)}</option>`).join('')}</select></div>${inputField('Session date','date','','date',`value="${today()}"`)}${inputField('Room capacity','capacity','30','number','min="1" max="500" value="30"')}${inputField('Starts at','startTime','','time','value="09:00"')}${inputField('Ends at','endTime','','time','value="11:00"')}</div><div class="form-field mt-16"><label class="form-label" for="field-inventory">Approved equipment <span class="optional">optional</span></label><input id="field-inventory" name="inventory" class="form-input" placeholder="e.g. 20 chairs, 4 workbenches" maxlength="200" /><span class="form-help">Recorded for this prototype. Automated equipment checks are a later phase.</span></div><div class="form-field mt-24"><label class="form-label" for="media-input">Classroom photo or video <span class="required-dot">*</span></label><label class="upload-zone" id="upload-zone" for="media-input"><span class="upload-icon">${icon('upload')}</span><strong>Drop your classroom media here</strong><span>or <b>browse files</b> to upload</span><small>JPG, PNG, WebP, MP4 or WebM · up to 75 MB</small><input id="media-input" type="file" accept="image/jpeg,image/png,image/webp,video/mp4,video/webm" class="sr-only" /></label><div id="upload-result"></div><span class="form-help">Saved in your browser. A clear, wide view works best. Uploads run real person detection when attendance is submitted.</span></div><div class="error-text" id="form-error" role="alert"></div></form>`,`<button class="button secondary" data-action="close-modal">Cancel</button><button class="button primary" id="create-class-button" type="submit" form="class-form" disabled>${icon('plus')} Create class</button>`);
}
async function handleUpload(file) {
  if(!file||activeModal?.type!=='create-class'||uploading)return;
  if(!['image/jpeg','image/png','image/webp','video/mp4','video/webm'].includes(file.type)){toast('Choose a JPG, PNG, WebP, MP4, or WebM file.','error');return;}
  if(file.size>75*1024*1024){toast('This file is too large. Please keep uploads under 75 MB.','error');return;}
  if(!file.size){toast('This file is empty. Choose another file.','error');return;}
  uploading=true;$('#create-class-button').disabled=true;
  $('#upload-result').innerHTML=`<div class="upload-progress"><span class="spinner"></span> Saving your media locally…</div>`;
  try {
    const previous=draftMedia, previousURL=draftURL;
    draftMedia=await saveMedia(file);draftURL=URL.createObjectURL(file);
    if(previous)await deleteMedia(previous.id).catch(()=>{});
    if(previousURL)URL.revokeObjectURL(previousURL);
    $('#upload-result').innerHTML=`<div class="upload-preview">${file.type.startsWith('image')?`<img src="${draftURL}" alt="Classroom upload preview"/>`:`<video src="${draftURL}" muted controls preload="metadata" playsinline></video>`}<div class="upload-meta"><span class="badge verified">${icon('check')} Upload ready</span><strong>${escape(file.name)}</strong><span>${file.size<1024*1024?`${Math.ceil(file.size/1024)} KB`:`${(file.size/1024/1024).toFixed(1)} MB`} · Stored in this browser</span><button type="button" class="button ghost small" data-action="change-upload">Choose another file</button></div></div>`;
    $('#upload-zone').classList.add('hidden');$('#create-class-button').disabled=false;
  } catch(e) { $('#upload-zone').classList.remove('hidden');$('#upload-result').innerHTML='<p class="error-text">Could not store this upload. Your browser storage may be full. Try a smaller file.</p>'; $('#create-class-button').disabled=!draftMedia; }
  finally {uploading=false;}
}
function newInstitution() {
  if(state.ui.role!=='admin')return;activeModal={type:'institution'};
  showModal('Add an institution','Create its workspace and appoint an institution head.',`<form id="institution-form"><div class="form-grid">${inputField('Institution name','name','e.g. Jaipur Skills Academy','text','maxlength="90"')}${inputField('Institution code','code','e.g. JSA-004','text','maxlength="20"')}${inputField('City','city','e.g. Jaipur','text','maxlength="60"')}${inputField('Capacity','capacity','120','number','min="1" max="10000"')}${inputField('Institution head','head','Full name','text','maxlength="80"')}${inputField('Head email','email','head@institution.edu','email','maxlength="120"')}</div><div class="info-note mt-16">${icon('info')}This creates a demo workspace. No invitations or emails are sent.</div><div class="error-text" id="form-error" role="alert"></div></form>`,`<button class="button secondary" data-action="close-modal">Cancel</button><button class="button primary" type="submit" form="institution-form">Create institution ${icon('arrow')}</button>`);
}
function newTrainer() {
  if(state.ui.role!=='head')return;activeModal={type:'trainer'};
  showModal('Add a trainer',`Build the team at ${escape(inst()?.name)}.`,`<form id="trainer-form"><div class="form-grid">${inputField('Full name','name','e.g. Priya Nair','text','maxlength="80"')}${inputField('Email address','email','trainer@institution.edu','email','maxlength="120"')}${inputField('Specialty','specialty','e.g. Digital skills','text','maxlength="80"')}</div><div class="info-note mt-16">${icon('info')}The trainer appears in the demo role switcher. No invitation is sent.</div><div class="error-text" id="form-error" role="alert"></div></form>`,`<button class="button secondary" data-action="close-modal">Cancel</button><button class="button primary" type="submit" form="trainer-form">Add trainer ${icon('plus')}</button>`);
}
function attendanceModal(id) {
  const c=state.classes.find(c=>c.id===id);if(!c||state.ui.role!=='trainer'||c.trainerId!==state.ui.trainerId)return;
  if(jobs.has(id)){detailModal(id);return;}
  activeModal={type:'attendance',id};
  showModal('Mark attendance',`${escape(c.name)} · ${escape(c.room)} · ${escape(c.startTime)}–${escape(c.endTime)}`,`<form id="attendance-form" data-id="${id}"><div class="attendance-intro"><span class="attendance-large-icon">${icon('people')}</span><div><strong>How many trainees are present?</strong><p>Submit the count you recorded for this session.</p></div></div><div class="attendance-count-field"><label class="form-label" for="attendance-count">Trainer submitted count</label><input id="attendance-count" name="submittedCount" class="form-input" type="number" min="0" max="${c.capacity}" value="${c.submittedCount??''}" placeholder="0" required /><span class="form-help">Room capacity: ${c.capacity}. The model counts visible people, which can include staff.</span></div><div class="form-field mt-16"><label class="form-label" for="attendance-note">Session note <span class="optional">optional</span></label><textarea id="attendance-note" name="note" class="form-textarea" placeholder="Add context, such as a trainer standing inside the seating area…" maxlength="500">${escape(c.trainerNote||'')}</textarea></div><div class="info-note mt-16">${icon('info')}<span>${c.media?.source==='sample'?'This sample class replays a labelled demo result. Create a class with your own photo or video to run real person detection.':'Your media is analyzed in this browser. The first run downloads a detection model and needs internet. This experimental check estimates visible people.'}</span></div><div class="error-text" id="form-error" role="alert"></div></form>`,`<button class="button secondary" data-action="close-modal">Cancel</button><button class="button primary" type="submit" form="attendance-form">${icon('scan')} Submit & ${c.media?.source==='sample'?'run sample check':'analyze'}</button>`);
}
async function detailModal(id) {
  const c=state.classes.find(c=>c.id===id);if(!c||!classesFor().some(c=>c.id===id))return;
  if(detailURL){URL.revokeObjectURL(detailURL);detailURL=null;}
  activeModal={type:'detail',id};
  const gap=c.submittedCount!=null&&c.observedCount!=null?c.submittedCount-c.observedCount:null, job=jobs.get(id);
  const body=`<div class="detail-topline">${badge(c)}<span class="text-muted">${escape(c.date)} · ${escape(c.startTime)}–${escape(c.endTime)}</span></div><div class="detail-counts"><div class="count-box"><span>TRAINER SUBMITTED</span><strong>${count(c.submittedCount)}</strong></div><div class="count-box"><span>VISUALLY OBSERVED</span><strong class="text-teal">${count(c.observedCount)}</strong></div><div class="count-box"><span>DIFFERENCE</span><strong class="${gap?'text-amber':''}">${gap==null?'—':gap>0?`+${gap}`:gap}</strong></div></div>${job?`<div id="analysis-progress">${progressMarkup(job.progress)}</div>`:''}<div class="media-viewer" id="detail-media">${c.media?.source==='sample'?scene(c):'<div class="empty-state compact"><span class="spinner"></span><p>Opening your classroom media…</p></div>'}</div><div class="detail-meta"><span>${icon('people')} ${escape(trainer(c.trainerId)?.name)}</span><span>${icon('classes')} ${escape(c.room)} · ${c.capacity} places</span><span>${icon('building')} ${escape(inst(c.institutionId)?.city)}</span></div>${c.analysis?`<div class="info-note mt-16">${icon('info')}<div><strong>${c.analysis.source==='sample'?'Labelled sample result':c.analysis.source==='browser-model'?'Experimental visual estimate':'Verification unavailable'}</strong><span>${c.analysis.source==='sample'?'This seeded result demonstrates the workflow. It was not calculated from the classroom illustration.':escape(c.analysis.method||'Review the camera view and retry if needed.')}${c.analysis.confidence?` Confidence: ${escape(c.analysis.confidence)}.`:''}</span>${c.analysis.qualityFlags?.length?`<ul>${c.analysis.qualityFlags.map(f=>`<li>${escape(f)}</li>`).join('')}</ul>`:''}</div></div>`:''}${c.trainerNote?`<div class="review-record"><span class="eyebrow">TRAINER NOTE</span><p>${escape(c.trainerNote)}</p></div>`:''}${c.review?`<div class="review-record"><span class="eyebrow">${c.review.decision==='accepted'?'REVIEW ACCEPTED':'FOLLOW-UP REQUESTED'}</span><p>${escape(c.review.note)}</p><small>${escape(c.review.reviewer||'Institution head')} · ${timeLabel(c.review.reviewedAt)}</small></div>`:''}<div class="inventory-section"><h3>Approved classroom inventory</h3><div class="inventory-chips">${(c.expectedEquipment||[]).map(e=>`<span>${escape(e.name)} <b>${e.quantity}</b></span>`).join('')||'<span>No inventory recorded</span>'}</div><p class="form-help">Declared inventory only. Automated equipment verification is not included in this prototype.</p></div>${state.ui.role!=='trainer'&&unresolved(c)?`<form id="review-form" data-id="${id}" class="review-form"><label class="form-label" for="review-note">Add review context</label><textarea id="review-note" name="note" class="form-textarea" required minlength="5" maxlength="1000" placeholder="Explain the discrepancy or describe the follow-up needed…"></textarea><div class="review-actions"><button class="button secondary" type="submit" name="decision" value="followup">Request follow-up</button><button class="button primary" type="submit" name="decision" value="accepted">${icon('check')} Accept with note</button></div><div class="error-text" id="form-error" role="alert"></div></form>`:''}`;
  const footer=`<button class="button secondary" data-action="close-modal">Close</button>${state.ui.role==='trainer'&&!job?`<button class="button primary" data-action="attendance" data-id="${id}">${icon('people')} ${c.submittedCount==null?'Mark attendance':'Update submission'}</button>`:''}${state.ui.role!=='trainer'&&c.submittedCount!=null&&!job?`<button class="button secondary" data-action="reanalyze" data-id="${id}">${icon('scan')} ${c.media?.source==='sample'?'Replay sample check':'Run check again'}</button>`:''}`;
  showModal(escape(c.name),`${escape(inst(c.institutionId)?.name)} · Classroom detail`,body,footer,'modal-wide');
  if(c.media?.source!=='sample'){
    const blob=await getMedia(c.media?.id).catch(()=>null);
    if(activeModal?.type!=='detail'||activeModal.id!==id)return;
    if(!blob){$('#detail-media').innerHTML='<div class="empty-state compact">'+icon('alert')+'<strong>Media is not available in this browser</strong><p>Uploads are local to the browser where the class was created.</p></div>';return;}
    detailURL=URL.createObjectURL(blob);
    const boxData=c.analysis?.boxes||[];
    const boxes=c.analysis?.width&&c.analysis?.height?boxData.map(b=>`<span class="detection-box" style="left:${b.bbox[0]/c.analysis.width*100}%;top:${b.bbox[1]/c.analysis.height*100}%;width:${b.bbox[2]/c.analysis.width*100}%;height:${b.bbox[3]/c.analysis.height*100}%"><span>person ${Math.round(b.score*100)}%</span></span>`).join(''):'';
    $('#detail-media').innerHTML=`<div class="detection-media">${blob.type.startsWith('video')?`<video id="detail-video" src="${detailURL}" controls playsinline preload="metadata"></video>`:`<img src="${detailURL}" alt="Uploaded classroom with detected person boxes"/>${boxes?`<div class="media-overlay">${boxes}</div>`:''}`}</div><div class="media-caption">${icon('shield')} ${escape(c.media.name)} · local to this browser</div>`;
  }
}
function progressMarkup(p={stage:'loading',progress:0,message:'Preparing verification…'}) { return `<div class="analysis-panel"><div class="analysis-panel-title"><span class="spinner"></span><strong>${escape(p.message)}</strong><span>${Math.round(p.progress)}%</span></div><div class="progress-track"><div class="progress-fill" style="width:${p.progress}%"></div></div><div class="analysis-steps">${[['loading','Prepare'],['sampling','Read media'],['detecting','Count people'],['complete','Compare']].map(([stage,label],i)=>`<span class="analysis-step ${stage===p.stage?'active':''}">${i+1}. ${label}</span>`).join('')}</div><p class="form-help">You can close this panel. Keep this browser tab open until the check finishes.</p></div>`; }
async function runAnalysis(id) {
  const c=state.classes.find(c=>c.id===id);if(!c||jobs.has(id)||c.submittedCount==null)return;
  if(c.analysis||c.review){c.analysisHistory ||= []; c.analysisHistory.push({analysis:c.analysis,review:c.review,observedCount:c.observedCount,savedAt:new Date().toISOString()});}
  c.status='processing';c.review=null;c.analysis=null;c.observedCount=null;
  const job={progress:{stage:'loading',progress:0,message:'Preparing verification…'}};jobs.set(id,job);persist();render();detailModal(id);
  const onProgress=p=>{job.progress=p;const el=$('#analysis-progress');if(el&&activeModal?.id===id)el.innerHTML=progressMarkup(p);};
  try {
    let result;
    if(c.media?.source==='sample'){
      for(const [stage,progress,message] of [['loading',15,'Preparing labelled demo'],['sampling',40,'Replaying the sample observation'],['detecting',75,'Comparing the sample headcount']]){onProgress({stage,progress,message});await new Promise(r=>setTimeout(r,650));}
      result={source:'sample',count:c.sampleCount,confidence:'high',qualityFlags:[],model:'Labelled sample dataset',method:'A seeded observation used to demonstrate this workflow.'};
    }else{
      const blob=await getMedia(c.media?.id);if(!blob)throw new Error('The uploaded media is unavailable in this browser. Create a class with a new upload.');
      result=await analyzeMedia(blob,{onProgress});
    }
    c.analysis=result;c.observedCount=result.count;c.analyzedAt=new Date().toISOString();
    c.status=result.confidence==='low'?'unverified':c.submittedCount===result.count?'verified':'review';
    log('attendance',`${c.name}: ${c.submittedCount} submitted, ${result.count} ${result.source==='sample'?'sample':'visually observed'} occupants.`,id,c.institutionId);
    toast(c.status==='verified'?'Verification complete. The counts align.':c.status==='unverified'?'The count needs a clearer view. Added to the review queue.':'Count discrepancy found. Added to the institution review queue.');
  } catch(e){c.status='unverified';c.analysis={source:'failed',qualityFlags:[e.message||'Analysis could not finish. Please retry.'],model:'COCO-SSD'};log('review',`${c.name}: verification could not complete.`,id,c.institutionId);toast('Unable to verify this media. Your attendance submission is saved.','error');}
  finally{jobs.delete(id);persist();render();if(activeModal?.type==='detail'&&activeModal.id===id)detailModal(id);}
}
function guide() {
  activeModal={type:'guide'};
  showModal('Meet your new classroom workspace','A working prototype of the attendance verification journey.',`<div class="guide-steps"><div><span>01</span><div><strong>Start as an institution head</strong><p>Add a trainer, then create a class with a classroom photo or video.</p></div></div><div><span>02</span><div><strong>Switch to the assigned trainer</strong><p>Mark attendance. Uploaded media runs a real browser model; seeded classes replay labelled sample results.</p></div></div><div><span>03</span><div><strong>Review and see the wider picture</strong><p>Return to the institution head to review gaps, then switch to administrator for daily institution reports.</p></div></div></div><div class="info-note mt-24">${icon('info')}<div><strong>Prototype boundaries</strong><p>Roles are demo views, not secure sign-in. Records and media are stored only in this browser. Internet is needed to download the detection model. Visual counts cannot identify individual trainees or people outside the camera view. Equipment verification and CCTV integration are future steps.</p></div></div>`,`<button class="button ghost danger" data-action="reset">Reset demo data</button><button class="button primary" data-action="close-modal">Explore workspace ${icon('arrow')}</button>`);
}
function exportReport() {
  const day=state.ui.page==='reports'?reportDate:today(), items=classesFor().filter(c=>c.date===day);
  if(!items.length){toast('No classes to export for this date.','error');return;}
  const rows=[['Date','Institution','City','Class','Room','Trainer','Start','End','Submitted count','Observed count','Difference','Status','Analysis source','Review decision','Review note'],...items.map(c=>[c.date,inst(c.institutionId)?.name,inst(c.institutionId)?.city,c.name,c.room,trainer(c.trainerId)?.name,c.startTime,c.endTime,c.submittedCount??'',c.observedCount??'',c.submittedCount!=null&&c.observedCount!=null?c.submittedCount-c.observedCount:'',c.status,c.analysis?.source||'',c.review?.decision||'',c.review?.note||''])];
  const csv=rows.map(row=>row.map(v=>{let s=String(v??'');if(/^[=+\-@\t\r]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';}).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`centrewatch-daily-${day}.csv`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);toast('Report export started. Check your browser downloads.');
}

document.addEventListener('click',async e=>{
  const el=e.target.closest('[data-page],[data-action]');if(!el)return;
  if(el.dataset.page){e.preventDefault();navigate(el.dataset.page);return;}
  const id=el.dataset.id;
  switch(el.dataset.action){
    case 'open-nav':document.body.classList.add('nav-open');break;
    case 'close-nav':document.body.classList.remove('nav-open');break;
    case 'guide':guide();break;
    case 'new-class':newClass();break;
    case 'new-institution':newInstitution();break;
    case 'new-trainer':newTrainer();break;
    case 'attendance':attendanceModal(id);break;
    case 'detail':detailModal(id);break;
    case 'close-modal':closeModal();break;
    case 'export':exportReport();break;
    case 'notifications':if(state.ui.role==='trainer'){navigate('attendance');toast('Your verification updates appear beside each class.');}else navigate('reviews');break;
    case 'change-upload':$('#media-input')?.click();break;
    case 'open-institution':state.ui.institutionId=id;state.ui.role='head';state.ui.trainerId=state.trainers.find(t=>t.institutionId===id)?.id||'';navigate('overview');break;
    case 'open-trainer':state.ui.trainerId=id;state.ui.role='trainer';navigate('overview');break;
    case 'reanalyze':runAnalysis(id);break;
    case 'reset':if(jobs.size){toast('Wait for active checks to finish before resetting.','error');break;}if(confirm('Reset all prototype records and remove uploaded media from this browser?')){await clearMedia();state=resetState();await closeModal();render();toast('Demo workspace reset.');}break;
  }
});
document.addEventListener('change',e=>{
  const el=e.target;
  if(el.id==='role-select'){state.ui.role=el.value;if(!state.ui.trainerId)state.ui.trainerId=state.trainers.find(t=>t.institutionId===state.ui.institutionId)?.id||'';navigate('overview');}
  if(el.id==='institution-select'){state.ui.institutionId=el.value;state.ui.trainerId=state.trainers.find(t=>t.institutionId===el.value)?.id||'';persist();render();}
  if(el.id==='trainer-select'){state.ui.trainerId=el.value;persist();render();}
  if(el.id==='status-filter'){statusFilter=el.value;$('#class-results').innerHTML=classResults();}
  if(el.id==='report-date'){reportDate=el.value||today();render();}
  if(el.id==='media-input')handleUpload(el.files[0]);
});
document.addEventListener('input',e=>{if(e.target.id==='class-search'){search=e.target.value;$('#class-results').innerHTML=classResults();}});
document.addEventListener('dragover',e=>{if(e.target.closest('#upload-zone')){e.preventDefault();$('#upload-zone').classList.add('dragging');}});
document.addEventListener('dragleave',e=>{if(e.target.closest('#upload-zone'))$('#upload-zone').classList.remove('dragging');});
document.addEventListener('drop',e=>{if(e.target.closest('#upload-zone')){e.preventDefault();$('#upload-zone').classList.remove('dragging');handleUpload(e.dataTransfer.files[0]);}});
document.addEventListener('submit',async e=>{
  if(!['class-form','attendance-form','institution-form','trainer-form','review-form'].includes(e.target.id))return;e.preventDefault();
  const form=e.target, data=Object.fromEntries(new FormData(form)); for(const k of Object.keys(data))if(typeof data[k]==='string')data[k]=data[k].trim();
  const error=m=>{$('#form-error').textContent=m;};
  if(form.id==='class-form'){
    if(state.ui.role!=='head')return;
    if(!draftMedia||uploading){error('Upload classroom media before creating the class.');return;}
    if(!data.name||!data.course||!data.room){error('Please complete the class name, course, and room.');return;}
    if(data.endTime<=data.startTime){error('The end time must be after the start time.');return;}
    const overlap=state.classes.some(c=>c.date===data.date&&c.institutionId===state.ui.institutionId&&(c.room.toLowerCase()===data.room.toLowerCase()||c.trainerId===data.trainerId)&&c.startTime<data.endTime&&c.endTime>data.startTime);
    if(overlap){error('This room or trainer already has an overlapping session. Choose another time.');return;}
    const equipment=(data.inventory||'').split(',').map(s=>s.trim()).filter(Boolean).map(s=>{const m=s.match(/^(\d+)\s+(.+)$/);return {name:m?m[2]:s,quantity:m?Number(m[1]):1};});
    const c={id:uid('class'),institutionId:state.ui.institutionId,trainerId:data.trainerId,name:data.name,course:data.course,room:data.room,date:data.date,startTime:data.startTime,endTime:data.endTime,capacity:Number(data.capacity),expectedEquipment:equipment,media:{...draftMedia,source:'upload'},submittedCount:null,submittedAt:null,observedCount:null,status:'pending',analysis:null,review:null};
    state.classes.unshift(c);draftMedia=null;log('class',`${c.name} was created and assigned to ${trainer(c.trainerId)?.name}.`,c.id);persist();await closeModal();navigate('classes');toast('Class created. Its trainer can now submit attendance.');
  }
  if(form.id==='institution-form'){
    if(state.ui.role!=='admin')return;
    if(!data.name||!data.code||!data.city||!data.head){error('Complete every institution field.');return;}
    if(state.institutions.some(i=>i.code.toLowerCase()===data.code.toLowerCase())){error('That institution code is already in use.');return;}
    const i={...data,id:uid('inst'),capacity:Number(data.capacity),color:'#147d68'};state.institutions.push(i);log('institution',`${i.name} joined the workspace.`,null,i.id);persist();await closeModal();navigate('institutions');toast('Institution created. Open its workspace to add trainers.');
  }
  if(form.id==='trainer-form'){
    if(state.ui.role!=='head')return;
    if(!data.name||!data.specialty){error('Enter the trainer name and specialty.');return;}
    if(state.trainers.some(t=>t.institutionId===state.ui.institutionId&&t.email.toLowerCase()===data.email.toLowerCase())){error('This trainer email already exists in the institution.');return;}
    state.trainers.push({...data,id:uid('tr'),institutionId:state.ui.institutionId});log('system',`${data.name} was added to the training team.`);persist();await closeModal();render();toast('Trainer added. You can assign them when creating a class.');
  }
  if(form.id==='attendance-form'){
    const c=state.classes.find(c=>c.id===form.dataset.id);if(!c||state.ui.role!=='trainer'||c.trainerId!==state.ui.trainerId)return;
    const n=Number(data.submittedCount);if(!Number.isInteger(n)||n<0||n>c.capacity){error(`Enter a whole number from 0 to ${c.capacity}.`);return;}
    c.submissionHistory ||= []; if(c.submittedCount!=null)c.submissionHistory.push({count:c.submittedCount,submittedAt:c.submittedAt,analysis:c.analysis,review:c.review});
    c.submittedCount=n;c.submittedAt=new Date().toISOString();c.trainerNote=data.note;log('attendance',`${trainer(c.trainerId)?.name} submitted ${n} attendees for ${c.name}.`,c.id,c.institutionId);persist();await closeModal();runAnalysis(c.id);
  }
  if(form.id==='review-form'){
    const c=state.classes.find(c=>c.id===form.dataset.id);if(!c||state.ui.role==='trainer')return;
    if(data.note.length<5){error('Add a short explanation before recording the review.');return;}
    const decision=e.submitter?.value||'followup';if(c.review){c.reviewHistory ||= [];c.reviewHistory.push(c.review);}c.review={decision,note:data.note,reviewedAt:new Date().toISOString(),reviewer:persona().name};c.status=decision==='accepted'?'reviewed':'review';log('review',`${c.name}: ${decision==='accepted'?'review accepted with context':'follow-up requested'}.`,c.id,c.institutionId);persist();render();detailModal(c.id);toast(decision==='accepted'?'Review recorded. The original counts are preserved.':'Follow-up recorded for the class.');
  }
});
window.addEventListener('beforeunload',e=>{if(jobs.size||uploading){e.preventDefault();e.returnValue='';}});
render();
