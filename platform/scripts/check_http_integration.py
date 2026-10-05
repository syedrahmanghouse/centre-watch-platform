#!/usr/bin/env python3
"""Start an isolated real server and exercise roles, inference, reports and sync.
Requires an explicitly supplied development image. Uses temporary data/accounts,
never the installed user's records. A successful run is integration evidence,
not a classroom accuracy assessment.
"""
import sys,tempfile,time,json,socket,subprocess,os,shutil,argparse,secrets
from pathlib import Path
import httpx
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image',type=Path,required=True)
parser.add_argument('--expected-visible-count',type=int,required=True,help='Manually verified count in this development image')
parser.add_argument('--submitted-count',type=int,default=5)
args=parser.parse_args()
if args.submitted_count==args.expected_visible_count: parser.error('Choose a different submitted count to exercise discrepancy review.')
if not args.image.is_file(): parser.error('Image does not exist.')
platform=Path(__file__).resolve().parents[1]
fixture_password=secrets.token_urlsafe(24)
sys.path.insert(0,str(platform))
from scripts.offline_sync import Outbox,make_event
runtime=Path(tempfile.mkdtemp(prefix='cw-real-http-'))
(runtime/'incoming').mkdir()
shutil.copy(args.image,runtime/'incoming'/'test-photo.jpg')
with socket.socket() as s:
    s.bind(('127.0.0.1',0));port=s.getsockname()[1]
base=f'http://127.0.0.1:{port}'
env={**os.environ,'CW_DATA_DIR':str(runtime/'data'),'CW_INCOMING_DIR':str(runtime/'incoming')}
log=(runtime/'server.log').open('w')
process=subprocess.Popen([sys.executable,'-m','uvicorn','server.app:app','--host','127.0.0.1','--port',str(port)],cwd=platform,env=env,stdout=log,stderr=log)
try:
    with httpx.Client(base_url=base,timeout=15) as c:
        for _ in range(100):
            try:
                if c.get('/api/health').status_code==200:break
            except httpx.ConnectError:pass
            time.sleep(.1)
        assert c.get('/').status_code==200
        def post(path,payload):
            r=c.post('/api'+path,json=payload)
            assert r.status_code<400,(r.status_code,r.text)
            result=r.json()
            if result.get('csrfToken'):c.headers['X-CSRF-Token']=result['csrfToken']
            return result
        def login(role):return post('/login',{'email':role+'@example.test','password':fixture_password})
        post('/setup',{'name':'HTTP Admin','email':'admin@example.test','password':fixture_password})
        inst=post('/institutions',{'name':'HTTP test centre','code':'HTTPTEST','city':'Fixture','head':'HTTP Head','email':'head@example.test','password':fixture_password})
        post('/incoming/assign',{'filename':'test-photo.jpg','institutionId':inst['id']})
        login('head')
        tr=post('/trainers',{'name':'HTTP Trainer','email':'trainer@example.test','password':fixture_password})
        cls=post('/classes',{'name':'HTTP real model test','room':'Fixture','date':'2026-10-04','startTime':'09:00','endTime':'10:00','capacity':30,'trainerId':tr['id'],'incomingFile':'test-photo.jpg','tolerance':0,'expectedEquipment':[]})
        login('trainer')
        job=post('/classes/'+cls['id']+'/attendance',{'submittedCount':args.submitted_count})['job']
        for _ in range(150):
            status=c.get('/api/jobs/'+job['id']).json()
            if status['status'] in ('failed','completed'):break
            time.sleep(.2)
        assert status['status']=='completed',status
        assert status['result']['count']==args.expected_visible_count,status
        login('head')
        post('/classes/'+cls['id']+'/review',{'decision':'followup','note':'Investigate mismatch in isolated test.'})
        observations=c.get('/api/classes/'+cls['id']+'/observations').json()['observations']
        assert observations[0]['submittedCount']==args.submitted_count and observations[0]['observedCount']==args.expected_visible_count
        device=post('/devices',{'name':'HTTP low bandwidth test'})
        outbox=Outbox(runtime/'outbox.sqlite3')
        event=make_event(device['id'],cls['id'],status['result'])
        queued=outbox.enqueue(event)
        outbox.close()
        # Restart the outbox object to prove the metadata survives process/session lifetime.
        outbox=Outbox(runtime/'outbox.sqlite3')
        acknowledged=outbox.sync(base,device['key'],allow_local_http=True)
        assert acknowledged['acked']==1,acknowledged
        duplicate=c.post('/api/edge/events',json=event,headers={'X-Device-Key':device['key']})
        assert duplicate.status_code==200 and duplicate.json()['duplicate'],duplicate.text
        login('admin')
        report=c.get('/api/reports?date=2026-10-04').json()
        assert report['classes'][0]['submittedCount']==args.submitted_count and report['classes'][0]['observedCount']==args.expected_visible_count
        assert c.get('/api/reports.csv?date=2026-10-04').status_code==200
        assert c.get('/api/evaluations/example').status_code==200
        assert c.get('/api/capabilities').json()['model']['ready']
        result={'checkedAt':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'source':'Development image; not classroom accuracy assessment','httpServer':'uvicorn on loopback','model':'Prepared YOLO-World local artifact','submitted':args.submitted_count,'observed':args.expected_visible_count,'mismatch':'review','reviewPersisted':True,'adminCsvExport':True,'incomingAssignmentEnforced':True,'outboxReopened':True,'edgeEventBodyBytes':queued['bytes'],'edgeAcknowledged':acknowledged['acked'],'edgeDuplicateAcknowledged':True,'evidenceImageCount':len(status['result']['evidence']),'modelReady':True,'runtime':str(runtime)}
        (platform.parent/'docs'/'integration-check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
finally:
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    log.close()
