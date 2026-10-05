"""Real API/RBAC/queue tests; model behavior stubbed only inside isolated tests."""
import os
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server.app import create_app
from server.db import uid, now

@pytest.fixture
def env(tmp_path):
    incoming=tmp_path/'incoming';incoming.mkdir();(incoming/'room.mp4').write_bytes(b'fixture video for queued-job tests')
    (incoming/'room-a.mp4').write_bytes(b'fixture video a')
    (incoming/'room-b.mp4').write_bytes(b'fixture video b')
    app=create_app(tmp_path/'runtime',incoming,start_worker=False)
    client=TestClient(app)
    return app,client


def session(client,email,password='a secure password 123'):
    response=client.post('/api/login',json={'email':email,'password':password})
    assert response.status_code==200,response.text
    client.headers['X-CSRF-Token']=response.json()['csrfToken']
    return response.json()['user']


def setup(client):
    result=client.post('/api/setup',json={'email':'admin@example.test','name':'Admin','password':'a secure password 123'})
    assert result.status_code==200,result.text
    client.headers['X-CSRF-Token']=result.json()['csrfToken']
    return result


def institution(client,suffix='a'):
    response=client.post('/api/institutions',json={'name':'Skill centre '+suffix,'code':'CENTRE-'+suffix,'city':'Chennai','head':'Head '+suffix,'email':f'head{suffix}@example.test','password':'a secure password 123'})
    assert response.status_code==201,response.text
    return response.json()


def fixture_class(client,suffix='a'):
    inst=institution(client,suffix)
    assigned=client.post('/api/incoming/assign',json={'filename':f'room-{suffix}.mp4','institutionId':inst['id']})
    assert assigned.status_code==200,assigned.text
    session(client,f'head{suffix}@example.test')
    response=client.post('/api/trainers',json={'name':'Trainer '+suffix,'email':f'trainer{suffix}@example.test','password':'a secure password 123','specialty':'Electrical'})
    assert response.status_code==201,response.text
    trainer=response.json()
    response=client.post('/api/classes',json={'name':'Electrical basics','room':'Room 1','course':'Electrician','date':'2026-10-04','startTime':'09:00','endTime':'10:00','capacity':30,'trainerId':trainer['id'],'incomingFile':f'room-{suffix}.mp4','expectedEquipment':[{'name':'chair','quantity':25,'observable':True}]})
    assert response.status_code==201,response.text
    return inst,trainer,response.json()


def test_setup_secure_sessions_and_csrf(env):
    app,client=env
    assert client.get('/api/bootstrap').json()['setupRequired'] is True
    result=setup(client)
    cookie=result.headers['set-cookie']
    assert 'HttpOnly' in cookie and 'SameSite=strict' in cookie
    assert client.get('/api/bootstrap').json()['user']['role']=='admin'
    assert client.post('/api/setup',json={'email':'admin2@example.test','name':'Admin','password':'a secure password 123'}).status_code==409
    client.headers.pop('X-CSRF-Token')
    assert client.post('/api/logout').status_code==403
    assert not app.state.db.one('SELECT password_hash FROM users')['password_hash'].endswith('a secure password 123')


def test_cross_origin_setup_rejected(env):
    _,client=env
    response=client.post('/api/setup',headers={'Origin':'https://attacker.example'},json={'email':'admin@example.test','name':'Admin','password':'a secure password 123'})
    assert response.status_code==403


def test_role_and_tenant_isolation(env):
    app,client=env;setup(client)
    inst_a,trainer_a,class_a=fixture_class(client)
    session(client,'admin@example.test')
    inst_b,trainer_b,class_b=fixture_class(client,'b')
    assert client.post(f"/api/classes/{class_a['id']}/analyze").status_code==404
    data=client.get('/api/bootstrap').json()
    assert [c['id'] for c in data['classes']]==[class_b['id']]
    assert client.post('/api/institutions',json={'name':'Bad centre','code':'BAD','city':'City','head':'Head','email':'bad@example.test','password':'a secure password 123'}).status_code==403
    session(client,'trainera@example.test')
    assert client.post(f"/api/classes/{class_b['id']}/attendance",json={'submittedCount':20}).status_code==404
    assert client.get('/api/reports').status_code==403
    assert client.get('/api/incoming').status_code==403
    assert client.post(f"/api/classes/{class_a['id']}/review",json={'decision':'accepted','note':'Unallowed review'}).status_code==403


def test_attendance_persists_and_queue_compares_real_result(env,monkeypatch):
    app,client=env;setup(client);_,_,cls=fixture_class(client)
    session(client,'trainera@example.test')
    result=client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':25,'note':'Session start'})
    assert result.status_code==200,result.text
    job=result.json()['job']
    assert job['status']=='queued'
    assert app.state.db.get('classes',cls['id'])['submittedCount']==25
    from server import vision
    monkeypatch.setattr(vision,'analyze_source',lambda *args,**kwargs:{'count':20,'confidence':'high','qualityFlags':[],'frames':[],'equipment':[{'name':'chair','expected':25,'observed':21,'status':'gap'}]})
    app.state.jobs.run_analysis(job)
    observed=client.get('/api/bootstrap').json()['classes'][0]
    assert observed['observedCount']==20 and observed['status']=='review'
    assert observed['attendanceStatus']=='review' and observed['infrastructureStatus']=='review'
    assert len(client.get(f"/api/classes/{cls['id']}/observations").json()['observations'])==1
    session(client,'heada@example.test')
    assert client.post(f"/api/classes/{cls['id']}/review",json={'decision':'followup','note':'Inspect seating and reported attendance.'}).status_code==200
    assert app.state.db.get('classes',cls['id'])['observedCount']==20
    assert len(app.state.db.activities())>=6


def test_camera_snapshots_require_class_access_and_disable_caching(env):
    app,client=env;setup(client)
    _,_,cls=fixture_class(client)
    job=app.state.jobs.enqueue(cls,None,'snapshot access test')
    directory=app.state.db.root/'evidence'/job['id']
    directory.mkdir(parents=True)
    content=b'private snapshot fixture'
    (directory/'observation-000.jpg').write_bytes(content)
    url=f"/api/jobs/{job['id']}/evidence/observation-000.jpg"
    response=client.get(url)
    assert response.status_code==200 and response.content==content
    assert response.headers['cache-control']=='no-store'
    session(client,'trainera@example.test')
    assert client.get(url).status_code==200
    session(client,'heada@example.test')
    response=client.post('/api/trainers',json={'name':'Unassigned trainer','email':'unassigned@example.test','password':'a secure password 123','specialty':'Electrical'})
    assert response.status_code==201
    session(client,'unassigned@example.test')
    assert client.get(url).status_code==404
    session(client,'admin@example.test')
    assert client.get(url).status_code==200
    fixture_class(client,'b')
    assert client.get(url).status_code==404
    assert client.post('/api/logout').status_code==200
    assert client.get(url).status_code==401


def test_analysis_failure_never_reports_verified(env,monkeypatch):
    app,client=env;setup(client);_,_,cls=fixture_class(client)
    session(client,'trainera@example.test')
    job=client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':25}).json()['job']
    from server import vision
    # Simulate a prior verified scan being followed by a failed periodic check.
    with app.state.db.transaction():
        previous=app.state.db.get('classes',cls['id'])
        previous.update(attendanceStatus='verified',infrastructureStatus='verified',review={'decision':'accepted','note':'Previous scan'},observedCount=25)
        app.state.db.put('classes',previous)
    def fail(*args,**kwargs): raise RuntimeError('rtsp://username:secret@host private camera URL')
    monkeypatch.setattr(vision,'analyze_source',fail)
    app.state.jobs.run_analysis(job)
    actual=app.state.db.get('classes',cls['id'])
    assert actual['status']=='unverified' and actual['observedCount'] is None
    assert actual['attendanceStatus']=='unverified' and actual['infrastructureStatus']=='unverified'
    assert actual['review'] is None and actual['analyzedAt']
    assert app.state.db.list('observations')[0]['observedCount'] is None
    public=client.get(f"/api/jobs/{job['id']}").text
    assert 'secret' not in public and 'username' not in public


def test_queue_resubmission_does_not_overwrite_current_attendance(env,monkeypatch):
    app,client=env;setup(client);_,_,cls=fixture_class(client);session(client,'trainera@example.test')
    job=client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':25}).json()['job']
    client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':27})
    from server import vision
    monkeypatch.setattr(vision,'analyze_source',lambda *args,**kwargs:{'count':25,'confidence':'high','qualityFlags':[],'equipment':[]})
    app.state.jobs.run_analysis(job)
    actual=app.state.db.get('classes',cls['id'])
    assert actual['submittedCount']==27 and actual['status']=='analyzing'
    queued=[j for j in app.state.db.list('jobs') if j['status']=='queued']
    assert len(queued)==1 and queued[0]['submittedCount']==27


def test_incoming_path_and_assignment_validation(env):
    app,client=env;setup(client);_,trainer,cls=fixture_class(client)
    assert client.patch(f"/api/classes/{cls['id']}",json={'incomingFile':'../room.mp4'}).status_code==400
    assert client.patch(f"/api/classes/{cls['id']}",json={'trainerId':'missing'}).status_code==400
    assert client.patch(f"/api/classes/{cls['id']}",json={'monitoringEnabled':True}).status_code==400
    session(client,'trainera@example.test')
    assert client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':31}).status_code==422
    assert client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':-1}).status_code==422


def test_edge_sync_deduplication_and_tenant_scope(env):
    app,client=env;setup(client);inst,_,cls=fixture_class(client)
    device=client.post('/api/devices',json={'name':'Edge PC'}).json()
    headers={'X-Device-Key':device['key']}
    event={'event_id':'sample-event-001','device_id':device['id'],'occurred_at':now(),'event_type':'observation','payload':{'classId':cls['id'],'analysis':{'count':20,'confidence':'high','qualityFlags':[],'equipment':[]}}}
    first=client.post('/api/edge/events',headers=headers,json=event)
    assert first.status_code==200,first.text
    assert first.json()['duplicate'] is False
    assert client.post('/api/edge/events',headers=headers,json=event).json()['duplicate'] is True
    event['payload']['analysis']['count']=30
    assert client.post('/api/edge/events',headers=headers,json=event).status_code==409
    event['event_id']='sample-event-002';event['payload']['analysis']['evidence']=['raw']
    assert client.post('/api/edge/events',headers=headers,json=event).status_code==422
    event['payload']['analysis'].pop('evidence');event['payload']['classId']='foreign-class'
    assert client.post('/api/edge/events',headers=headers,json=event).status_code==404
    assert 'keyHash' not in client.get('/api/bootstrap').text


def test_retention_only_deletes_owned_uploads(env):
    app,client=env;setup(client)
    db=app.state.db
    owned=db.root/'uploads'/'old.mp4';owned.write_bytes(b'old')
    incoming=app.state.incoming/'room.mp4'
    old=time.time()-90000;os.utime(owned,(old,old));os.utime(incoming,(old,old))
    with db.transaction():
        db.put('media',{'id':'old-upload','source':'upload','path':str(owned),'institutionId':None})
        db.put('media',{'id':'old-incoming','source':'incoming','path':str(incoming),'institutionId':None})
    app.state.jobs.cleanup()
    assert not owned.exists() and incoming.exists()
    assert db.get('media','old-upload')['expiredAt']


def test_reports_csv_escapes_formula_cells(env):
    app,client=env;setup(client);_,_,cls=fixture_class(client)
    assert client.patch(f"/api/classes/{cls['id']}",json={'name':'=SUM(1,2)'}).status_code==200
    response=client.get('/api/reports.csv?date=2026-10-04')
    assert response.status_code==200 and "'=SUM(1,2)" in response.text
    assert client.get('/api/reports?date=invalid').status_code==422

def test_password_change_revokes_previous_sessions(env):
    app,client=env;setup(client)
    old_token=client.cookies.get('cw_session')
    response=client.post('/api/account/password',json={'currentPassword':'wrong password','password':'replacement password 123'})
    assert response.status_code==400
    response=client.post('/api/account/password',json={'currentPassword':'a secure password 123','password':'replacement password 123'})
    assert response.status_code==200,response.text
    assert client.cookies.get('cw_session')!=old_token
    stale=TestClient(app);stale.cookies.set('cw_session',old_token)
    assert stale.get('/api/bootstrap').status_code==401
    assert client.post('/api/login',json={'email':'admin@example.test','password':'a secure password 123'}).status_code==401
    assert client.post('/api/login',json={'email':'admin@example.test','password':'replacement password 123'}).status_code==200


def test_head_password_recovery_is_scoped(env):
    app,client=env;setup(client);_,trainer_a,_=fixture_class(client)
    session(client,'admin@example.test');_,trainer_b,_=fixture_class(client,'b')
    assert client.post(f"/api/users/{trainer_a['id']}/password",json={'password':'new teacher password 123'}).status_code==404
    assert client.post(f"/api/users/{trainer_b['id']}/password",json={'password':'new teacher password 123'}).status_code==200
    assert session(client,'trainerb@example.test','new teacher password 123')['role']=='trainer'


def test_incoming_assignment_prevents_cross_tenant_media_access(env):
    app,client=env;setup(client);inst_a,_,cls=fixture_class(client)
    files=client.get('/api/incoming').json()['files']
    assert [file['name'] for file in files]==['room-a.mp4']
    assert client.patch(f"/api/classes/{cls['id']}",json={'incomingFile':'room-b.mp4'}).status_code==403
    assert client.post('/api/incoming/assign',json={'filename':'room-b.mp4','institutionId':inst_a['id']}).status_code==403
    session(client,'admin@example.test');inst_b=institution(client,'b')
    assert client.post('/api/incoming/assign',json={'filename':'room-a.mp4','institutionId':inst_b['id']}).status_code==409
    assert client.post('/api/incoming/assign',json={'filename':'../room-a.mp4','institutionId':inst_b['id']}).status_code==404


def test_restart_recovers_in_progress_jobs_without_losing_records(env):
    app,client=env;setup(client);_,_,cls=fixture_class(client);session(client,'trainera@example.test')
    job=client.post(f"/api/classes/{cls['id']}/attendance",json={'submittedCount':25}).json()['job']
    with app.state.db.transaction():
        job['status']='running';app.state.db.put('jobs',job)
    restarted=create_app(app.state.db.root,app.state.incoming,start_worker=False)
    assert restarted.state.db.get('jobs',job['id'])['status']=='queued'
    assert restarted.state.db.get('classes',cls['id'])['submittedCount']==25


def test_upload_decodes_content_and_limits_raw_media_to_assigned_trainer(env):
    cv2=pytest.importorskip('cv2');np=pytest.importorskip('numpy')
    app,client=env;setup(client);_,_,cls=fixture_class(client)
    _,encoded=cv2.imencode('.png',np.full((100,100,3),100,dtype=np.uint8))
    response=client.post('/api/media',files={'file':('classroom.png',encoded.tobytes(),'image/png')})
    assert response.status_code==201,response.text
    media=response.json()
    assert client.patch(f"/api/classes/{cls['id']}",json={'mediaId':media['id']}).status_code==200
    assert client.post('/api/media',files={'file':('fake.png',b'<html>not an image</html>','image/png')}).status_code==400
    session(client,'trainera@example.test')
    assert client.get(f"/api/media/{media['id']}").status_code==200
    session(client,'admin@example.test');fixture_class(client,'b');session(client,'trainerb@example.test')
    assert client.get(f"/api/media/{media['id']}").status_code==404
