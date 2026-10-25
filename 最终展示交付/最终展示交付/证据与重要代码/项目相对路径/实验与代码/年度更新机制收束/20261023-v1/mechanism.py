"""Small deterministic maintenance policy; no final evaluation labels enter decisions."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
def audit_candidate(candidate,incumbent):
 reasons=[]
 for group,old in incumbent.items():
  new=candidate[group]
  if old['positives']<CONFIG['min_monitor_positives']:reasons.append(group+':insufficient_positive_support');continue
  if new['recall10']<old['recall10']-CONFIG['recall_tolerance']-1e-12:reasons.append(group+':recall_drop')
  if new['ap']<old['ap']-CONFIG['ap_tolerance']-1e-12:reasons.append(group+':ap_drop')
 return {'accept':not reasons,'reasons':reasons}
def policy_state(date,support,completed):
 events=[];force=False
 for p in CONFIG['policies']:
  if p['known_date']>date or p['effective_date']>date:continue
  done=completed.get(p['id'],0)
  if done>=p['successful_full_updates']:status='COMPLETED'
  elif support.get(p['id'],0)<p['minimum_recorded_positive_rows']:status='POLICY_WAIT'
  else:status='POLICY_FULL';force=True
  events.append(dict(id=p['id'],status=status,completed=done,required=p['successful_full_updates'],support=support.get(p['id'],0)))
 return {'prefer_full':force,'events':events}
def consume_policy(state,completed,accepted_full):
 out=dict(completed)
 if accepted_full:
  for p in state['events']:
   if p['status']=='POLICY_FULL':out[p['id']]=out.get(p['id'],0)+1
 return out

def cache_key(encoder_hash,preprocessing_version,data_version):
 return hashlib.sha256('|'.join([encoder_hash,preprocessing_version,data_version]).encode()).hexdigest()
def publish_ready(checks):return all(checks.get(k) is True for k in CONFIG['release_checks_required'])

def rollback_target(versions,current_version):
 """Resolve a saved parent model; the caller explicitly loads it instead of overwriting weights."""
 by_id={v['version']:v for v in versions}
 current=by_id[current_version];parent_id=current.get('parent')
 if parent_id is None:raise ValueError('initial version has no parent')
 parent=by_id[parent_id];path=Path(parent['model_path'])
 if not path.is_file():raise FileNotFoundError(path)
 actual=hashlib.sha256(path.read_bytes()).hexdigest()
 if actual!=parent['model_sha256']:raise ValueError('parent model hash mismatch')
 return dict(version=parent_id,model_path=str(path),model_sha256=actual,encoder_hash=parent['encoder_hash'])

class RepresentationCache:
 """Version-keyed row cache. fetch_and_encode receives only unavailable current rows."""
 def __init__(self):self.version=None;self.rows={}
 def get(self,keys,encoder_hash,preprocessing_version,data_version,fetch_and_encode):
  version=cache_key(encoder_hash,preprocessing_version,data_version)
  invalidated=self.version is not None and self.version!=version
  if self.version!=version:self.rows={};self.version=version
  missing=list(dict.fromkeys(k for k in keys if k not in self.rows))
  if missing:
   encoded=fetch_and_encode(missing)
   if set(encoded)!=set(missing):raise ValueError('encoder result keys differ from requested keys')
   self.rows.update(encoded)
  return [self.rows[k] for k in keys],dict(newly_encoded=len(missing),reused=len(keys)-len(missing),invalidated=invalidated,version=version)
