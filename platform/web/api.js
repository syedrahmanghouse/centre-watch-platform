let csrf = '';
let generation = 0;
const pending = new Set();
export function setCsrf(value) { csrf = value || ''; }
export function invalidateRequests() {
  generation++;
  csrf = '';
  for (const controller of pending) controller.abort();
  pending.clear();
}
export async function api(path, {method='GET', body, ...options} = {}) {
  const requestGeneration = generation;
  const controller = new AbortController();
  pending.add(controller);
  const headers = {...options.headers};
  if (method !== 'GET') headers['X-CSRF-Token'] = csrf;
  if (body && !(body instanceof FormData)) { headers['Content-Type']='application/json'; body=JSON.stringify(body); }
  let response;
  try { response = await fetch(`/api${path}`, {...options, method, body, headers, credentials:'same-origin', cache:'no-store', signal:controller.signal}); }
  catch { throw new Error(requestGeneration !== generation ? 'The session changed. Please retry in your current workspace.' : 'Cannot reach the monitoring server. Your saved records are safe. Reconnect and try again.'); }
  finally { pending.delete(controller); }
  const data = await response.json().catch(()=>({}));
  if (requestGeneration !== generation) throw new Error('The session changed. Please retry in your current workspace.');
  if (!response.ok) {
    const message = Array.isArray(data.detail) ? data.detail.map(x=>`${x.loc?.slice(1).join('.') || 'Input'}: ${x.msg}`).join('; ') : data.detail;
    const error = new Error(message || `Request failed (${response.status}).`); error.status=response.status; throw error;
  }
  if (data.csrfToken) setCsrf(data.csrfToken);
  return data;
}
