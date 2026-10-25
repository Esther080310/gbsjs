import copy,unittest
from mechanism import audit_candidate,policy_state,consume_policy,cache_key,publish_ready,CONFIG
class MechanismTests(unittest.TestCase):
 def setUp(self):self.old={'latest':dict(positives=50,recall10=.3,ap=.4),'anchor':dict(positives=60,recall10=.25,ap=.35)}
 def test_small_change_allowed(self):
  c=copy.deepcopy(self.old);c['latest']['recall10']-=.009;self.assertTrue(audit_candidate(c,self.old)['accept'])
 def test_large_drop_rejected(self):
  c=copy.deepcopy(self.old);c['latest']['recall10']-=.011;self.assertFalse(audit_candidate(c,self.old)['accept'])
 def test_ap_alone_can_reject(self):
  c=copy.deepcopy(self.old);c['anchor']['ap']-=.011;self.assertFalse(audit_candidate(c,self.old)['accept'])
 def test_insufficient_labels_do_not_pass(self):
  self.old['anchor']['positives']=29;self.assertFalse(audit_candidate(self.old,self.old)['accept'])
 def test_policy_not_visible_early(self):self.assertEqual(policy_state('2019-01-01',{'securities_law_2020':80},{})['events'],[])
 def test_policy_wait(self):self.assertFalse(policy_state('2021-04-30',{'securities_law_2020':0},{})['prefer_full'])
 def test_policy_two_successes_not_two_attempts(self):
  k='securities_law_2020';state=policy_state('2023-04-30',{k:50},{});self.assertTrue(state['prefer_full']);done=consume_policy(state,{},False);self.assertEqual(done,{})
  done=consume_policy(state,done,True);self.assertEqual(done[k],1);state=policy_state('2024-04-30',{k:80},done);done=consume_policy(state,done,True);self.assertEqual(done[k],2);self.assertFalse(policy_state('2025-04-30',{k:80},done)['prefer_full'])
 def test_encoder_or_data_version_invalidates_cache(self):
  self.assertNotEqual(cache_key('a','s','d'),cache_key('b','s','d'));self.assertNotEqual(cache_key('a','s','d'),cache_key('a','s','d2'))
 def test_publication_requires_all_checks(self):
  self.assertFalse(publish_ready({}));self.assertTrue(publish_ready({k:True for k in CONFIG['release_checks_required']}))
if __name__=='__main__':unittest.main()
