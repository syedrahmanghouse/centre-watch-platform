"""Durable analysis queue; a failed model never becomes a successful observation."""
import json
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from .db import uid, now


def apply_observation(db, cls, result, actor=None, captured_at=None):
    count = result.get('count')
    if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count < 0):
        raise ValueError('Analyzer returned an invalid headcount.')
    if result.get('confidence') == 'low' and count is not None:
        result = dict(result, candidateCount=count, count=None)
        count = None
    flags = result.get('qualityFlags', [])
    uncertain = count is None or result.get('status') in ('unverified','insufficient_evidence','error') or result.get('confidence') == 'low'
    gap = None if count is None or cls.get('submittedCount') is None else abs(count - cls['submittedCount'])
    equipment_issues = any(item.get('status') in ('gap','missing','shortfall','review','insufficient','possible_missing') for item in result.get('equipment',[]) if isinstance(item,dict))
    cls['attendanceStatus'] = 'unverified' if uncertain else ('review' if gap is not None and gap > cls.get('tolerance',2) else ('verified' if gap is not None else 'pending'))
    cls['infrastructureStatus'] = 'review' if equipment_issues else ('unverified' if any(i.get('status') == 'unknown' for i in result.get('equipment',[]) if isinstance(i,dict)) else ('verified' if result.get('equipment') else 'not_configured'))
    cls['analysis'] = result
    cls['observedCount'] = count
    cls['analyzedAt'] = captured_at or now()
    cls['status'] = ('review' if equipment_issues or cls['attendanceStatus']=='review' else ('unverified' if uncertain or cls['infrastructureStatus']=='unverified' else ('verified' if gap is not None else 'analyzed')))
    cls['review'] = None
    cls['nextCheckAt'] = (datetime.now().astimezone() + timedelta(minutes=cls.get('intervalMinutes',30)*random.uniform(.8,1.2))).isoformat()
    db.put('classes',cls)
    observation = {'id':uid('observation'),'classId':cls['id'],'institutionId':cls['institutionId'],'submittedCount':cls.get('submittedCount'),'observedCount':count,'status':cls['status'],'analysis':result,'capturedAt':cls['analyzedAt']}
    db.put('observations',observation)
    message = f"{cls['name']}: {cls.get('submittedCount','—')} submitted, {count if count is not None else 'unable to verify'} observed ({cls['status']})."
    db.audit(actor,cls['institutionId'],cls['id'],'analysis',message,{'observationId':observation['id'],'qualityFlags':flags})
    return cls


class JobManager:
    def __init__(self, db, resolve_source, incoming_dir):
        self.db,self.resolve_source,self.incoming_dir = db,resolve_source,Path(incoming_dir)
        self.stop_event = threading.Event()
        self.thread = None
        with db.transaction():
            for kind in ('jobs','evaluations'):
                for job in db.list(kind):
                    if job['status'] == 'running':
                        job.update(status='queued',message='Resuming after service restart.')
                        db.put(kind,job)

    def start(self):
        self.thread = threading.Thread(target=self.loop,daemon=True,name='centrewatch-analysis')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)

    def enqueue(self, cls, actor, reason='requested'):
        with self.db.transaction():
            active = next((j for j in self.db.list('jobs') if j['classId']==cls['id'] and j['status'] in ('queued','running')),None)
            if active:
                cls['status']='analyzing';self.db.put('classes',cls)
                return active
            job = {'id':uid('job'),'classId':cls['id'],'institutionId':cls['institutionId'],'actorId':actor,'status':'queued','progress':0,'createdAt':now(),'reason':reason,'revision':cls.get('revision',0),'submittedCount':cls.get('submittedCount'),'message':'Waiting for analysis.'}
            self.db.put('jobs',job)
            cls['nextCheckAt']=(datetime.now().astimezone()+timedelta(minutes=cls.get('intervalMinutes',30)*random.uniform(.8,1.2))).isoformat()
            cls['status']='analyzing'
            self.db.put('classes',cls)
            return job

    def update(self,kind,job,**changes):
        with self.db.transaction():
            job.update(changes)
            self.db.put(kind,job)

    def run_analysis(self,job):
        from . import vision
        db=self.db
        cls=db.get('classes',job['classId'])
        if cls is None:
            self.update('jobs',job,status='failed',finishedAt=now(),message='The class record is no longer available.')
            return
        self.update('jobs',job,status='running',startedAt=now(),message='Reading source and running local detection.')
        try:
            source=self.resolve_source(cls)
            output_dir=db.root/'evidence'/job['id']
            output_dir.mkdir(parents=True,exist_ok=True)
            def progress(value, message=None):
                if isinstance(value,dict):
                    message=value.get('message',message)
                    value=value.get('progress',min(.95,float(value.get('framesProcessed',0))/30))
                value=max(0,min(1,float(value)))
                self.update('jobs',job,progress=value,message=message or 'Analyzing sampled frames.')
            result=vision.analyze_source(source,cls.get('expectedEquipment',[]),options={'output_dir':str(output_dir),'evidence_mode':'annotated','class_id':cls['id'],'media_type':cls['media']['type'],'trainee_zone':cls.get('traineeZone'),'coverage_confirmed':cls.get('coverageConfirmed',False),'max_frames':30,'sample_seconds':60,'submitted_count':job.get('submittedCount'),'tolerance':cls.get('tolerance',2)},progress=progress)
            # Evidence paths stay internal; browser obtains files through a scoped route.
            if isinstance(result.get('evidence'),list):
                for evidence in result['evidence']:
                    if isinstance(evidence,dict):
                        path=evidence.pop('path',None)
                        if path:
                            file=Path(path)
                            if file.resolve().is_relative_to(output_dir.resolve()): evidence['url']=f"/api/jobs/{job['id']}/evidence/{file.name}"
            with db.transaction():
                current=db.get('classes',cls['id'])
                if current.get('revision',0) == job['revision']:
                    apply_observation(db,current,result,job['actorId'])
                else:
                    result['superseded']=True
                    # A resubmission while detection was running always gets a fresh check.
                    self.update('jobs',job,status='completed',progress=1,result=result,finishedAt=now(),message='Earlier submission analyzed; new submission queued.')
                    self.enqueue(current,job['actorId'],'updated submission')
                    return
                self.update('jobs',job,status='completed',progress=1,result=result,finishedAt=now(),message='Analysis complete.')
        except Exception as exc:
            # Do not leak camera URLs or credentials through driver exception strings.
            message='Analysis could not complete. Verify the source, model availability, and camera connection.'
            if exc.__class__.__name__ in ('ModelUnavailableError','ModelNotReadyError','VisionUnavailable','PeopleModelUnavailable'): message='The local detection model is not installed. Prepare the model before retrying.'
            self.update('jobs',job,status='failed',finishedAt=now(),message=message,errorType=type(exc).__name__)
            with db.transaction():
                current=db.get('classes',cls['id'])
                if current.get('revision',0)==job['revision']:
                    failure={'count':None,'confidence':'low','status':'unverified','qualityFlags':['analysis_failed'],'detail':message,'equipment':[{'name':item['name'],'expected':item['quantity'],'observed':None,'status':'unknown','operability':'unverified'} for item in current.get('expectedEquipment',[])]}
                    apply_observation(db,current,failure,job['actorId'])
                else:
                    self.enqueue(current,job['actorId'],'updated submission after failed prior check')
                db.audit(job['actorId'],cls['institutionId'],cls['id'],'error',message,{'jobId':job['id'],'errorType':type(exc).__name__})

    def run_evaluation(self,job):
        from .evaluation import run_evaluation
        self.update('evaluations',job,status='running',startedAt=now())
        try:
            output=self.db.root/'evaluations'/job['id']
            report=run_evaluation(job['manifest'],incoming_dir=self.incoming_dir,output_dir=output)
            self.update('evaluations',job,status='completed',finishedAt=now(),report=report)
        except Exception as exc:
            self.update('evaluations',job,status='failed',finishedAt=now(),message=str(exc)[:500])

    def schedule(self):
        current=datetime.now().astimezone()
        for cls in self.db.list('classes'):
            if not cls.get('monitoringEnabled') or cls['media']['source']!='rtsp' or cls.get('submittedCount') is None: continue
            start=datetime.fromisoformat(f"{cls['date']}T{cls['startTime']}").astimezone()
            end=datetime.fromisoformat(f"{cls['date']}T{cls['endTime']}").astimezone()
            due=datetime.fromisoformat(cls['nextCheckAt']) if cls.get('nextCheckAt') else start
            if start<=current<=end and current>=due: self.enqueue(cls,None,'scheduled check')

    def cleanup(self):
        settings=self.db.get('settings','global') or {}
        threshold=time.time()-settings.get('retentionDays',30)*86400
        # Retention deletes only owned upload/evidence files, never incoming originals.
        upload_threshold=time.time()-settings.get('uploadRetentionHours',24)*3600
        for media in self.db.list('media'):
            if media.get('source')!='upload' or media.get('expiredAt'): continue
            file=Path(media['path']).resolve()
            if file.is_relative_to((self.db.root/'uploads').resolve()) and file.is_file() and file.stat().st_mtime<upload_threshold:
                file.unlink()
                with self.db.transaction():
                    media['expiredAt']=now();self.db.put('media',media)
                    for cls in self.db.list('classes'):
                        if cls['media']['id']==media['id']:
                            cls['sourceExpired']=True;self.db.put('classes',cls)
                    self.db.audit(None,media.get('institutionId'),None,'retention','Expired uploaded source under the local retention policy.',{'mediaId':media['id']})
        base=self.db.root/'evidence'
        if base.exists():
            for file in base.glob('*/*'):
                if file.is_file() and file.stat().st_mtime<threshold: file.unlink()

    def loop(self):
        last_maintenance=0
        while not self.stop_event.is_set():
            try:
                if time.monotonic()-last_maintenance>30:
                    self.schedule();self.cleanup();last_maintenance=time.monotonic()
                job=next((j for j in reversed(self.db.list('jobs')) if j['status']=='queued'),None)
                if job: self.run_analysis(job);continue
                evaluation=next((j for j in reversed(self.db.list('evaluations')) if j['status']=='queued'),None)
                if evaluation: self.run_evaluation(evaluation);continue
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Queue maintenance failed')
            self.stop_event.wait(1)
