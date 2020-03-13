diff --git a/baselines/gail/trpo_mpi.py b/baselines/gail/trpo_mpi.py
index 615a4326a7..9b26da6991 100644
--- a/baselines/gail/trpo_mpi.py
+++ b/baselines/gail/trpo_mpi.py
@@ -249,7 +249,7 @@ def fisher_vector_product(p):
             # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
             ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
             vpredbefore = seg["vpred"]  # predicted value function before udpate
-            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
+            atarg = (atarg - atarg.mean()) / (atarg.std() or 1.)  # standardized advantage function estimate
 
             if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy
