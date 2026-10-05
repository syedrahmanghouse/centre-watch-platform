"""CentreWatch on-site API: scoped accounts, durable records and real inference."""
import csv
import io
import json
import mimetypes
import math
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from cryptography.fernet import Fernet
from fastapi import FastAPI, Depends, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .db import Database, uid, now
from .auth import COOKIE, current_user, check_origin, scoped, class_access, require_role, password_hash, verify_password, issue_session, digest_token, user_view
from .schemas import Account, Login, Institution, Trainer, ClassPayload, Attendance, Review, Settings, Device, PasswordChange, PasswordReset, IncomingAssignment
from .jobs import JobManager, apply_observation

BASE=Path(__file__).resolve().parents[1]
SUFFIXES={'.mp4','.mov','.avi','.mkv','.webm','.jpg','.jpeg','.png'}
MIME={'.mp4':'video/mp4','.mov':'video/quicktime','.avi':'video/x-msvideo','.mkv':'video/x-matroska','.webm':'video/webm','.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png'}


def create_app(data_dir=None, incoming_dir=None, start_worker=True):
    db=Database(data_dir or os.environ.get('CW_DATA_DIR',BASE/'runtime'))
    incoming=Path(incoming_dir or os.environ.get('CW_INCOMING_DIR',BASE.parent/'data'/'incoming')).resolve()
    incoming.mkdir(parents=True,exist_ok=True)
    uploads=db.root/'uploads';uploads.mkdir(exist_ok=True)
    secret=db.root/'camera.key'
    if not secret.exists():
        fd=os.open(secret,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as fh: fh.write(Fernet.generate_key())
    cipher=Fernet(secret.read_bytes())

    def resolve_source(cls):
        media=db.get('media',cls['media']['id'])
        if not media: raise ValueError('Source no longer available.')
        if media['source']=='rtsp': return cipher.decrypt(media['encryptedUrl'].encode()).decode()
        file=Path(media['path']).resolve()
        if not file.is_relative_to(incoming if media['source']=='incoming' else uploads): raise ValueError('Source outside allowed directory.')
        if not file.is_file(): raise ValueError('Source file unavailable.')
        return str(file)

    manager=JobManager(db,resolve_source,incoming)
    @asynccontextmanager
    async def lifespan(app):
        if start_worker: manager.start()
        yield
        manager.stop()
    app=FastAPI(title='CentreWatch',version='2.0.0',lifespan=lifespan)
    app.state.db=db;app.state.jobs=manager;app.state.incoming=incoming
    failures=defaultdict(deque)

    @app.middleware('http')
    async def security_headers(request,call_next):
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['X-Frame-Options']='DENY'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Permissions-Policy']='camera=(), microphone=(), geolocation=()'
        if request.url.path.startswith('/api'): response.headers['Cache-Control']='no-store'
        return response

    @app.exception_handler(sqlite3.IntegrityError)
    async def duplicate_error(request,exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409,content={'detail':'This email or record already exists.'})

    def get_class(class_id,user):
        cls=db.get('classes',class_id)
        if not cls or not class_access(user,cls): raise HTTPException(404,'Class not found.')
        return cls

    def media_view(media):
        return {key:media.get(key) for key in ('id','name','type','size','source')}

    def institution_for(user,requested=None):
        institution_id=requested if user['role']=='admin' else user['institution_id']
        if not institution_id or not db.get('institutions',institution_id): raise HTTPException(400,'Choose a valid institution.')
        return institution_id

    def insert_user(account,role,institution_id=None,specialty=''):
        try: validated=Account(name=account['name'],email=account['email'],password=account['password'])
        except ValueError: raise HTTPException(422,'Use a valid name, email address, and password of at least 10 characters.')
        user={'id':uid('user'),'email':validated.email,'name':validated.name,'role':role,'institution_id':institution_id,'password_hash':password_hash(validated.password),'specialty':specialty}
        db.execute('INSERT INTO users VALUES(:id,:email,:name,:role,:institution_id,:password_hash,:specialty)',user)
        return user

    def public_job(job):
        return {k:v for k,v in job.items() if k not in ('manifest',)}

    def source_for(payload,user,institution_id):
        if payload.mediaId:
            media=db.get('media',payload.mediaId)
            if not media or (media.get('institutionId')!=institution_id and not (user['role']=='admin' and media.get('institutionId') is None)): raise HTTPException(404,'Uploaded media not found for this institution.')
            if media.get('institutionId') is None: media['institutionId']=institution_id;db.put('media',media)
            return media_view(media)
        if payload.incomingFile:
            filename=payload.incomingFile
            if Path(filename).name!=filename or Path(filename).suffix.lower() not in SUFFIXES: raise HTTPException(400,'Choose an allowed filename from incoming footage.')
            path=(incoming/filename).resolve()
            if path.parent!=incoming or not path.is_file() or (incoming/filename).is_symlink(): raise HTTPException(404,'Incoming footage not found.')
            assignment=db.get('incomingAssignments',filename)
            if not assignment or assignment['institutionId']!=institution_id: raise HTTPException(403,'An administrator must assign this incoming file to the institution first.')
            media={'id':uid('media'),'name':filename,'type':MIME[path.suffix.lower()],'size':path.stat().st_size,'path':str(path),'institutionId':institution_id,'source':'incoming','createdAt':now()}
        else:
            parsed=urlsplit(payload.rtspUrl or '')
            if parsed.scheme not in ('rtsp','rtsps') or not parsed.hostname: raise HTTPException(400,'Camera URL must be a valid RTSP or RTSPS stream.')
            media={'id':uid('media'),'name':payload.cameraName,'type':'video/rtsp','size':0,'institutionId':institution_id,'source':'rtsp','encryptedUrl':cipher.encrypt(payload.rtspUrl.encode()).decode(),'createdAt':now()}
        db.put('media',media)
        return media_view(media)

    @app.get('/api/health')
    def health(): return {'status':'ok','version':'2.0.0','storage':'sqlite','inference':'local'}

    @app.get('/api/capabilities')
    def capabilities(user=Depends(current_user)):
        try:
            from .vision import model_status
            model=model_status()
            model.pop('modelPath',None)
        except Exception: model={'ready':False,'detail':'Detection runtime is not available.'}
        return {'model':model,'rtsp':True,'ptz':False,'offlineInference':True,'maxUploadBytes':536870912,'supportedMedia':sorted(SUFFIXES),'edgeSync':True,'privacy':'Aggregate counts; no face identification or audio analysis.','timezone':datetime.now().astimezone().tzname()}

    @app.post('/api/setup')
    def setup(payload:Account,request:Request,response:Response):
        check_origin(request)
        with db.transaction():
            if db.one("SELECT id FROM users WHERE role='admin'"): raise HTTPException(409,'Setup is already complete.')
            user=insert_user(payload.model_dump(),'admin')
            db.put('settings',{'id':'global','intervalMinutes':30,'retentionDays':30,'uploadRetentionHours':24,'tolerance':2})
            db.audit(user['id'],None,None,'setup','Administrator account created.')
        return issue_session(db,user,response,request)

    @app.post('/api/login')
    def login(payload:Login,request:Request,response:Response):
        check_origin(request)
        ip=request.client.host if request.client else 'unknown'
        attempts=failures[ip]
        while attempts and attempts[0]<time.time()-900: attempts.popleft()
        if len(attempts)>=15: raise HTTPException(429,'Too many sign-in attempts. Try again in 15 minutes.')
        user=db.one('SELECT * FROM users WHERE email=?',(payload.email.lower(),))
        if not user or not verify_password(payload.password,user['password_hash']):
            attempts.append(time.time());raise HTTPException(401,'Email or password is incorrect.')
        attempts.clear()
        return issue_session(db,user,response,request)

    @app.post('/api/account/password')
    def change_password(payload:PasswordChange,request:Request,response:Response,user=Depends(current_user)):
        attempts=failures['password:'+user['id']]
        while attempts and attempts[0]<time.time()-900: attempts.popleft()
        if len(attempts)>=10: raise HTTPException(429,'Too many password attempts. Try again in 15 minutes.')
        if not verify_password(payload.currentPassword,user['password_hash']):
            attempts.append(time.time());raise HTTPException(400,'Current password is incorrect.')
        with db.transaction():
            db.execute('UPDATE users SET password_hash=? WHERE id=?',(password_hash(payload.password),user['id']))
            db.execute('DELETE FROM sessions WHERE user_id=?',(user['id'],))
            db.audit(user['id'],user['institution_id'],None,'account','Changed account password and revoked previous sessions.')
        attempts.clear()
        return issue_session(db,user,response,request)

    @app.post('/api/users/{user_id}/password')
    def reset_password(user_id:str,payload:PasswordReset,user=Depends(current_user)):
        require_role(user,'admin','head')
        target=db.one('SELECT * FROM users WHERE id=?',(user_id,))
        if not target or target['role']=='admin' or (user['role']=='head' and (target['role']!='trainer' or target['institution_id']!=user['institution_id'])): raise HTTPException(404,'Account not available for password recovery.')
        with db.transaction():
            db.execute('UPDATE users SET password_hash=? WHERE id=?',(password_hash(payload.password),user_id))
            db.execute('DELETE FROM sessions WHERE user_id=?',(user_id,))
            db.audit(user['id'],target['institution_id'],None,'account','Reset password for '+target['name']+'; previous sessions revoked.',{'targetUserId':user_id})
        return {'ok':True}

    @app.post('/api/logout')
    def logout(request:Request,response:Response,user=Depends(current_user)):
        with db.transaction(): db.execute('DELETE FROM sessions WHERE token_hash=?',(request.state.session['token_hash'],))
        response.delete_cookie(COOKIE,path='/')
        return {'ok':True}

    @app.get('/api/bootstrap')
    def bootstrap(request:Request):
        if not db.one("SELECT id FROM users WHERE role='admin'"): return {'setupRequired':True,'user':None}
        user=current_user(request)
        classes=[c for c in db.list('classes') if class_access(user,c)]
        class_ids={c['id'] for c in classes}
        trainers=[dict(user_view(u),specialty=u['specialty']) for u in db.all("SELECT * FROM users WHERE role='trainer'") if scoped(user,u['institution_id']) and (user['role']!='trainer' or u['id']==user['id'])]
        return {'setupRequired':False,'user':user_view(user),'csrfToken':request.state.session['csrf'],'institutions':[i for i in db.list('institutions') if scoped(user,i['id'])],'trainers':trainers,'classes':classes,'activity':[a for a in db.activities() if (user['role']=='admin' or a['institutionId']==user['institution_id']) and (user['role']!='trainer' or a['classId'] in class_ids)],'jobs':[public_job(j) for j in db.list('jobs') if j['classId'] in class_ids][:100],'settings':db.get('settings','global') or {},'devices':[{k:v for k,v in d.items() if k!='keyHash'} for d in db.list('devices') if scoped(user,d['institutionId'])] if user['role']!='trainer' else [],'evaluationReports':[public_job(e) for e in db.list('evaluations')] if user['role']=='admin' else []}

    @app.post('/api/institutions',status_code=201)
    def create_institution(payload:Institution,user=Depends(current_user)):
        require_role(user,'admin')
        with db.transaction():
            if any(i['code'].lower()==payload.code.lower() for i in db.list('institutions')): raise HTTPException(409,'Institution code already exists.')
            inst={'id':uid('inst'),**payload.model_dump(exclude={'password'}),'createdAt':now()}
            head=insert_user({'name':payload.head,'email':payload.email,'password':payload.password},'head',inst['id'])
            inst['headId']=head['id'];inst['email']=head['email']
            db.put('institutions',inst);db.audit(user['id'],inst['id'],None,'institution',f"Created {inst['name']} and its institution head.")
        return inst

    @app.post('/api/trainers',status_code=201)
    def create_trainer(payload:Trainer,user=Depends(current_user)):
        require_role(user,'head','admin');institution_id=institution_for(user,payload.institutionId)
        with db.transaction():
            trainer=insert_user(payload.model_dump(),'trainer',institution_id,payload.specialty)
            db.audit(user['id'],institution_id,None,'trainer',f"Created trainer account for {trainer['name']}.")
        return dict(user_view(trainer),specialty=payload.specialty)

    @app.get('/api/incoming')
    def incoming_files(user=Depends(current_user)):
        require_role(user,'head','admin')
        files=[]
        for path in sorted(incoming.iterdir()):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in SUFFIXES: continue
            assignment=db.get('incomingAssignments',path.name)
            institution_id=assignment['institutionId'] if assignment else None
            if user['role']=='head' and institution_id!=user['institution_id']: continue
            files.append({'name':path.name,'size':path.stat().st_size,'type':MIME[path.suffix.lower()],'institutionId':institution_id})
        return {'files':files}

    @app.post('/api/incoming/assign')
    def assign_incoming(payload:IncomingAssignment,user=Depends(current_user)):
        require_role(user,'admin')
        filename=payload.filename;path=(incoming/filename).resolve()
        if Path(filename).name!=filename or path.parent!=incoming or not path.is_file() or (incoming/filename).is_symlink() or path.suffix.lower() not in SUFFIXES: raise HTTPException(404,'Incoming footage not found.')
        if not db.get('institutions',payload.institutionId): raise HTTPException(404,'Institution not found.')
        with db.transaction():
            if any(m.get('source')=='incoming' and m.get('name')==filename and m.get('institutionId')!=payload.institutionId for m in db.list('media')): raise HTTPException(409,'This footage is already attached to another institution. Remove those media registrations before reassigning it.')
            assignment={'id':filename,'institutionId':payload.institutionId,'assignedAt':now(),'assignedBy':user['id']}
            db.put('incomingAssignments',assignment)
            db.audit(user['id'],payload.institutionId,None,'media','Assigned incoming footage '+filename+' to this institution.')
        return assignment

    @app.post('/api/media',status_code=201)
    async def upload(file:UploadFile=File(...),user=Depends(current_user)):
        require_role(user,'head','admin')
        suffix=Path(file.filename or '').suffix.lower()
        if suffix not in SUFFIXES: raise HTTPException(400,'Upload MP4, MOV, AVI, MKV, WebM, JPG, or PNG.')
        media_id=uid('media');path=uploads/(media_id+suffix);size=0
        try:
            with path.open('wb') as target:
                while chunk:=await file.read(1024*1024):
                    size+=len(chunk)
                    if size>536870912: raise HTTPException(413,'File exceeds 512 MB. Put larger footage in the incoming folder.')
                    target.write(chunk)
            if not size: raise HTTPException(400,'File is empty.')
            # Probe actual container/image content, not client-supplied MIME or extension.
            try: import cv2
            except ImportError: raise HTTPException(503,'Install the video runtime before uploading footage.')
            if suffix in ('.jpg','.jpeg','.png'):
                valid=cv2.imread(str(path)) is not None
            else:
                capture=cv2.VideoCapture(str(path));valid,frame=capture.read();capture.release()
            if not valid: raise HTTPException(400,'This file could not be decoded as supported footage.')
            media={'id':media_id,'name':Path(file.filename).name,'size':size,'type':MIME[suffix],'source':'upload','institutionId':user['institution_id'],'path':str(path),'createdAt':now()}
            with db.transaction(): db.put('media',media)
            return media_view(media)
        except Exception:
            path.unlink(missing_ok=True);raise
        finally: await file.close()

    def get_media(media_id,user):
        media=db.get('media',media_id)
        if not media or not scoped(user,media.get('institutionId')): raise HTTPException(404,'Media not found.')
        if user['role']=='trainer' and not any(c['media']['id']==media_id and class_access(user,c) for c in db.list('classes')): raise HTTPException(404,'Media not found.')
        return media

    @app.get('/api/media/{media_id}')
    def download_media(media_id:str,user=Depends(current_user)):
        media=get_media(media_id,user)
        if media['source']=='rtsp': raise HTTPException(400,'Camera credentials and streams are available only to the local analysis service.')
        path=Path(media['path']).resolve()
        if not path.is_relative_to(incoming if media['source']=='incoming' else uploads) or not path.is_file(): raise HTTPException(404,'Media file unavailable.')
        return FileResponse(path,media_type=media['type'])

    @app.delete('/api/media/{media_id}')
    def delete_media(media_id:str,user=Depends(current_user)):
        require_role(user,'admin','head');media=get_media(media_id,user)
        if any(c['media']['id']==media_id for c in db.list('classes')): raise HTTPException(409,'Assign another source to this class before removing its media.')
        if media['source']=='upload':
            path=Path(media['path']).resolve()
            if path.is_relative_to(uploads): path.unlink(missing_ok=True)
        with db.transaction():
            db.execute('DELETE FROM records WHERE kind=? AND id=?',('media',media_id))
            db.audit(user['id'],media.get('institutionId'),None,'media','Removed media registration and any owned uploaded file.')
        return {'ok':True}

    @app.post('/api/classes',status_code=201)
    def create_class(payload:ClassPayload,user=Depends(current_user)):
        require_role(user,'head','admin');institution_id=institution_for(user,payload.institutionId)
        trainer=db.one("SELECT * FROM users WHERE id=? AND role='trainer' AND institution_id=?",(payload.trainerId,institution_id))
        if not trainer: raise HTTPException(400,'Assign a trainer from this institution.')
        with db.transaction():
            media=source_for(payload,user,institution_id)
            if payload.monitoringEnabled and media['source']!='rtsp': raise HTTPException(400,'Scheduled monitoring requires a live camera. Uploaded footage is analyzed on submission or request.')
            cls={'id':uid('class'),**payload.model_dump(exclude={'mediaId','incomingFile','rtspUrl','cameraName'}),'institutionId':institution_id,'media':media,'submittedCount':None,'observedCount':None,'status':'pending','analysis':None,'review':None,'revision':0,'createdAt':now()}
            db.put('classes',cls);db.audit(user['id'],institution_id,cls['id'],'class',f"Created {cls['name']} in {cls['room']}.")
        return cls

    @app.patch('/api/classes/{class_id}')
    def update_class(class_id:str,payload:dict,user=Depends(current_user)):
        require_role(user,'head','admin');cls=get_class(class_id,user)
        allowed={'name','course','room','date','startTime','endTime','capacity','trainerId','expectedEquipment','mediaId','incomingFile','rtspUrl','cameraName','monitoringEnabled','intervalMinutes','tolerance','traineeZone','coverageConfirmed'}
        if set(payload)-allowed: raise HTTPException(422,'Unsupported class setting.')
        merged={k:v for k,v in cls.items() if k in ClassPayload.model_fields}
        merged['mediaId']=cls['media']['id']
        if any(k in payload for k in ('mediaId','incomingFile','rtspUrl')):
            for key in ('mediaId','incomingFile','rtspUrl'): merged.pop(key,None)
        merged.update(payload)
        if 'coverageConfirmed' in payload and 'expectedEquipment' not in payload:
            merged['expectedEquipment']=[dict(item,observable=bool(payload['coverageConfirmed'])) for item in cls.get('expectedEquipment',[])]
        try: validated=ClassPayload(**merged)
        except ValueError as exc: raise HTTPException(422,str(exc))
        if not db.one("SELECT id FROM users WHERE id=? AND role='trainer' AND institution_id=?",(validated.trainerId,cls['institutionId'])): raise HTTPException(400,'Assign a trainer from this institution.')
        with db.transaction():
            media=source_for(validated,user,cls['institutionId'])
            if validated.monitoringEnabled and media['source']!='rtsp': raise HTTPException(400,'Scheduled monitoring requires a live camera.')
            cls.update(validated.model_dump(exclude={'mediaId','incomingFile','rtspUrl','cameraName'}));cls['media']=media
            cls.update(revision=cls['revision']+1,analysis=None,observedCount=None,status='pending',review=None,attendanceStatus='pending',infrastructureStatus='pending',analyzedAt=None,sourceExpired=bool((db.get('media',media['id']) or {}).get('expiredAt')))
            db.put('classes',cls);db.audit(user['id'],cls['institutionId'],class_id,'class','Updated class configuration; previous observations remain in audit history.')
        return cls

    @app.post('/api/classes/{class_id}/attendance')
    def attendance(class_id:str,payload:Attendance,user=Depends(current_user)):
        require_role(user,'trainer');cls=get_class(class_id,user)
        if payload.submittedCount>cls['capacity']: raise HTTPException(422,'Submitted count exceeds class capacity.')
        with db.transaction():
            old=cls.get('submittedCount');cls.update(submittedCount=payload.submittedCount,submittedAt=now(),submissionNote=payload.note,revision=cls.get('revision',0)+1,review=None,analysis=None,observedCount=None,analyzedAt=None,attendanceStatus='pending',infrastructureStatus='pending')
            db.put('classes',cls);db.audit(user['id'],cls['institutionId'],class_id,'attendance',f"Attendance submitted: {payload.submittedCount}.",{'before':old,'after':payload.submittedCount,'note':payload.note})
            job=manager.enqueue(cls,user['id'],'attendance submission')
        return {'class':db.get('classes',class_id),'job':public_job(job)}

    @app.post('/api/classes/{class_id}/analyze')
    def analyze(class_id:str,user=Depends(current_user)):
        cls=get_class(class_id,user)
        job=manager.enqueue(cls,user['id'])
        return {'class':db.get('classes',class_id),'job':public_job(job)}

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id:str,user=Depends(current_user)):
        job=db.get('jobs',job_id)
        if not job: raise HTTPException(404,'Job not found.')
        get_class(job['classId'],user)
        return public_job(job)

    @app.get('/api/jobs/{job_id}/evidence/{filename}')
    def evidence(job_id:str,filename:str,user=Depends(current_user)):
        job=db.get('jobs',job_id)
        if not job: raise HTTPException(404,'Evidence not found.')
        get_class(job['classId'],user)
        base=(db.root/'evidence'/job_id).resolve();path=(base/filename).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.suffix.lower() not in ('.jpg','.jpeg','.png'): raise HTTPException(404,'Evidence unavailable or expired.')
        return FileResponse(path)

    @app.post('/api/classes/{class_id}/review')
    def review(class_id:str,payload:Review,user=Depends(current_user)):
        require_role(user,'head','admin');cls=get_class(class_id,user)
        if not cls.get('analysis'): raise HTTPException(409,'Wait for an observation before reviewing.')
        with db.transaction():
            cls['review']={'decision':payload.decision,'note':payload.note,'reviewerId':user['id'],'reviewerName':user['name'],'time':now()}
            db.put('classes',cls);db.audit(user['id'],cls['institutionId'],class_id,'review',f"{user['name']} recorded {payload.decision}: {payload.note}",cls['review'])
        return cls

    @app.get('/api/classes/{class_id}/observations')
    def observations(class_id:str,user=Depends(current_user)):
        get_class(class_id,user)
        return {'observations':[o for o in db.list('observations') if o['classId']==class_id]}

    def report_data(user,date):
        require_role(user,'head','admin')
        if date:
            try: datetime.strptime(date,'%Y-%m-%d')
            except ValueError: raise HTTPException(422,'Use date YYYY-MM-DD.')
        classes=[c for c in db.list('classes') if scoped(user,c['institutionId']) and (not date or c['date']==date)]
        institutions={i['id']:i['name'] for i in db.list('institutions')}
        return {'date':date,'generatedAt':now(),'totals':{'classes':len(classes),'submitted':sum(c['submittedCount'] is not None for c in classes),'verified':sum(c['status']=='verified' for c in classes),'review':sum(c['status']=='review' for c in classes),'unverified':sum(c['status']=='unverified' for c in classes)},'classes':[dict(c,institutionName=institutions.get(c['institutionId'],'')) for c in classes]}

    @app.get('/api/reports')
    def reports(date:str|None=None,user=Depends(current_user)): return report_data(user,date)

    @app.get('/api/reports.csv')
    def reports_csv(date:str|None=None,user=Depends(current_user)):
        report=report_data(user,date);stream=io.StringIO();writer=csv.writer(stream)
        writer.writerow(['Institution','Class','Room','Date','Submitted','Observed','Difference','Status','Review','Review note'])
        def safe(v):
            value='' if v is None else str(v)
            return "'"+value if value.startswith(('=','+','-','@','\t','\r')) else value
        for c in report['classes']:
            difference=c['observedCount']-c['submittedCount'] if c.get('observedCount') is not None and c.get('submittedCount') is not None else None
            review=c.get('review') or {}
            writer.writerow([safe(v) for v in [c['institutionName'],c['name'],c['room'],c['date'],c.get('submittedCount'),c.get('observedCount'),difference,c['status'],review.get('decision'),review.get('note')]])
        return Response(stream.getvalue(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="centrewatch-{date or "all"}.csv"'})

    @app.patch('/api/settings')
    def settings(payload:Settings,user=Depends(current_user)):
        require_role(user,'admin')
        with db.transaction():
            result=db.put('settings',{'id':'global',**payload.model_dump()})
            db.audit(user['id'],None,None,'settings','Updated monitoring and evidence retention settings.',result)
        return result

    @app.post('/api/evaluations',status_code=202)
    def create_evaluation(payload:dict,user=Depends(current_user)):
        require_role(user,'admin')
        manifest=payload.get('manifest')
        if not isinstance(manifest,dict) or len(json.dumps(manifest))>1048576: raise HTTPException(422,'Supply a labelled dataset manifest (maximum 1 MB).')
        record={'id':uid('evaluation'),'name':str(payload.get('name','Dataset assessment'))[:100],'manifest':manifest,'status':'queued','createdAt':now(),'actorId':user['id']}
        with db.transaction(): db.put('evaluations',record)
        return public_job(record)

    @app.get('/api/evaluations')
    def evaluations(user=Depends(current_user)):
        require_role(user,'admin');return {'evaluations':[public_job(e) for e in db.list('evaluations')]}

    @app.get('/api/evaluations/example')
    def evaluation_template(user=Depends(current_user)):
        require_role(user,'admin')
        path=BASE.parent/'data'/'annotations'/'manifest.example.json'
        if not path.is_file(): raise HTTPException(404,'Example manifest is unavailable.')
        return FileResponse(path,media_type='application/json',filename='manifest.example.json')

    @app.get('/api/evaluations/{evaluation_id}')
    def evaluation(evaluation_id:str,user=Depends(current_user)):
        require_role(user,'admin');record=db.get('evaluations',evaluation_id)
        if not record: raise HTTPException(404,'Evaluation not found.')
        return public_job(record)

    @app.get('/api/evaluations/{evaluation_id}/artifacts/{filename}')
    def evaluation_artifact(evaluation_id:str,filename:str,user=Depends(current_user)):
        require_role(user,'admin')
        if not db.get('evaluations',evaluation_id) or filename not in ('report.json','predictions.json','predictions.csv','manifest.json'): raise HTTPException(404,'Artifact not found.')
        path=db.root/'evaluations'/evaluation_id/filename
        if not path.is_file(): raise HTTPException(404,'Artifact not ready.')
        return FileResponse(path,filename=filename)

    @app.get('/api/devices')
    def list_devices(user=Depends(current_user)):
        require_role(user,'head','admin')
        return {'devices':[{k:v for k,v in d.items() if k!='keyHash'} for d in db.list('devices') if scoped(user,d['institutionId'])]}

    @app.post('/api/devices',status_code=201)
    def register_device(payload:Device,user=Depends(current_user)):
        require_role(user,'head','admin');institution_id=institution_for(user,payload.institutionId)
        key=secrets.token_urlsafe(40)
        record={'id':uid('device'),'name':payload.name,'institutionId':institution_id,'keyHash':digest_token(key),'createdAt':now(),'lastSeenAt':None,'revoked':False}
        with db.transaction():
            db.put('devices',record);db.audit(user['id'],institution_id,None,'device','Registered edge device '+payload.name)
        return {k:v for k,v in dict(record,key=key).items() if k!='keyHash'}

    @app.delete('/api/devices/{device_id}')
    def revoke_device(device_id:str,user=Depends(current_user)):
        require_role(user,'head','admin');device=db.get('devices',device_id)
        if not device or not scoped(user,device['institutionId']): raise HTTPException(404,'Device not found.')
        with db.transaction(): device['revoked']=True;db.put('devices',device)
        return {'ok':True}

    def edge_device(request):
        key=request.headers.get('x-device-key','')
        if not key: raise HTTPException(401,'Device key required.')
        hashed=digest_token(key)
        device=next((d for d in db.list('devices') if not d.get('revoked') and secrets.compare_digest(d['keyHash'],hashed)),None)
        if not device: raise HTTPException(401,'Device key invalid or revoked.')
        return device

    @app.get('/api/edge/config')
    def edge_config(request:Request):
        device=edge_device(request)
        return {'deviceId':device['id'],'institutionId':device['institutionId'],'classes':[c for c in db.list('classes') if c['institutionId']==device['institutionId']],'settings':db.get('settings','global')}

    @app.post('/api/edge/events')
    async def edge_event(request:Request):
        device=edge_device(request)
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>262144: raise HTTPException(413,'Metadata event exceeds 256 KiB.')
        try:
            event=json.loads(body)
            if not isinstance(event,dict) or set(event)!={'event_id','device_id','occurred_at','event_type','payload'}: raise ValueError('Unsupported event fields.')
            if event['event_type'] not in ('analysis','observation'): raise ValueError('Unsupported event type.')
            if not isinstance(event['payload'],dict) or set(event['payload'])!={'classId','analysis'}: raise ValueError('Unsupported payload fields.')
            event_id=event['event_id'];payload=event['payload'];class_id=payload['classId'];captured=event['occurred_at']
            captured_time=datetime.fromisoformat(captured.replace('Z','+00:00'))
            if captured_time.tzinfo is None: raise ValueError('Timestamp requires timezone.')
            if captured_time.timestamp()>time.time()+300: raise ValueError('Timestamp is in the future.')
            if not isinstance(event_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}',event_id): raise ValueError('Invalid event ID.')
            if event.get('device_id')!=device['id']: raise ValueError('Device ID does not match key.')
            analysis=payload['analysis']
            allowed={'count','confidence','qualityFlags','equipment','model','method','frames','elapsedSeconds','status','countRange','sampledAt','people','detail','uncertainty','frameCount','durationSeconds','usableFrames','confidenceMeaning','source','sourceType','modelVersion','processedAt','observationWindow'}
            if not isinstance(analysis,dict) or set(analysis)-allowed: raise ValueError('Send count metadata only, without media or evidence.')
            count=analysis.get('count')
            if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count<0 or count>10000): raise ValueError('Invalid count.')
            if not isinstance(analysis.get('equipment',[]),list) or len(analysis.get('equipment',[]))>100: raise ValueError('Invalid equipment metadata.')
            if not isinstance(analysis.get('qualityFlags',[]),list) or any(not isinstance(f,str) or len(f)>200 for f in analysis.get('qualityFlags',[])): raise ValueError('Invalid quality flags.')
            def bound_metadata(value,depth=0):
                if depth>5: raise ValueError('Metadata nesting too deep.')
                if isinstance(value,float) and not math.isfinite(value): raise ValueError('Metadata numbers must be finite.')
                if isinstance(value,str) and len(value)>1000: raise ValueError('Metadata string exceeds limit.')
                if isinstance(value,dict):
                    if any(str(k).lower() in ('image','video','base64','data','url','path','evidence','embedding','face','audio') for k in value): raise ValueError('Media and identity metadata are prohibited.')
                    for v in value.values(): bound_metadata(v,depth+1)
                elif isinstance(value,list):
                    if len(value)>300: raise ValueError('Metadata list exceeds limit.')
                    for v in value: bound_metadata(v,depth+1)
            bound_metadata(analysis)
        except (ValueError,KeyError,TypeError) as exc: raise HTTPException(422,'Invalid edge event: '+str(exc)[:200])
        cls=db.get('classes',class_id)
        if not cls or cls['institutionId']!=device['institutionId']: raise HTTPException(404,'Class not assigned to this device institution.')
        compound=device['id']+':'+event_id
        with db.transaction():
            existing=db.one('SELECT data FROM edge_events WHERE event_id=?',(compound,))
            if existing:
                if json.loads(existing['data'])!=event: raise HTTPException(409,'Event ID already used with different content.')
                return {'accepted':True,'duplicate':True,'eventId':event_id}
            db.execute('INSERT INTO edge_events VALUES(?,?,?,?)',(compound,device['id'],now(),json.dumps(event)))
            latest=datetime.fromisoformat(cls['analyzedAt']) if cls.get('analyzedAt') else None
            if latest is None or captured_time>=latest:
                apply_observation(db,cls,analysis,captured_at=captured)
            else:
                db.put('observations',{'id':uid('observation'),'classId':class_id,'institutionId':cls['institutionId'],'analysis':analysis,'observedCount':count,'capturedAt':captured,'source':'edge','historical':True})
            device['lastSeenAt']=now();db.put('devices',device)
            db.audit(None,device['institutionId'],class_id,'edge','Received offline observation.',{'deviceId':device['id'],'eventId':event_id,'capturedAt':captured})
        return {'accepted':True,'duplicate':False,'eventId':event_id}

    web=BASE/'web'
    if web.exists(): app.mount('/',StaticFiles(directory=web,html=True),name='web')
    return app

app=create_app()
