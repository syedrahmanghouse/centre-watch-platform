"""Passwords use scrypt; session secrets never enter persistent storage."""
import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import urlsplit
from fastapi import HTTPException, Request, Response

COOKIE = 'cw_session'


def password_hash(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return f'scrypt${salt.hex()}${digest.hex()}'


def verify_password(password, stored):
    try:
        _, salt, digest = stored.split('$')
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1, dklen=32)
        return hmac.compare_digest(candidate.hex(), digest)
    except (ValueError, TypeError):
        return False


def digest_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


def user_view(user):
    return {'id': user['id'], 'name': user['name'], 'email': user['email'], 'role': user['role'], 'institutionId': user['institution_id']}


def issue_session(db, user, response, request):
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with db.transaction():
        db.execute('DELETE FROM sessions WHERE expires<?', (time.time(),))
        db.execute('INSERT INTO sessions VALUES(?,?,?,?)', (digest_token(token), user['id'], csrf, time.time() + 43200))
    secure = request.url.scheme == 'https' or os.environ.get('CW_SECURE_COOKIES') == '1'
    response.set_cookie(COOKIE, token, max_age=43200, httponly=True, secure=secure, samesite='strict', path='/')
    return {'user': user_view(user), 'csrfToken': csrf}


def check_origin(request):
    origin = request.headers.get('origin')
    if origin:
        expected = os.environ.get('CW_PUBLIC_ORIGIN') or str(request.base_url).rstrip('/')
        parsed, target = urlsplit(origin), urlsplit(expected)
        if (parsed.scheme, parsed.netloc) != (target.scheme, target.netloc):
            raise HTTPException(403, 'Cross-origin requests are not permitted.')
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, 'Cross-site requests are not permitted.')


def current_user(request: Request):
    db = request.app.state.db
    token = request.cookies.get(COOKIE, '')
    session = db.one('SELECT * FROM sessions WHERE token_hash=? AND expires>?', (digest_token(token), time.time()))
    if not session:
        raise HTTPException(401, 'Please sign in.')
    user = db.one('SELECT * FROM users WHERE id=?', (session['user_id'],))
    if not user:
        raise HTTPException(401, 'Session no longer valid.')
    request.state.session = session
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        check_origin(request)
        if not hmac.compare_digest(request.headers.get('x-csrf-token', ''), session['csrf']):
            raise HTTPException(403, 'Missing or invalid request token. Refresh and try again.')
    return user


def require_role(user, *roles):
    if user['role'] not in roles:
        raise HTTPException(403, 'Your role cannot perform this action.')


def scoped(user, institution_id):
    return user['role'] == 'admin' or user['institution_id'] == institution_id


def class_access(user, cls):
    return scoped(user, cls['institutionId']) and (user['role'] != 'trainer' or cls['trainerId'] == user['id'])
