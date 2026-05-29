"""Steelman the method: is there a PREFIX-ADAPTIVE TEMPERATURE story even though
candidate-conditioning fails? For each position find the optimal global-temperature
beta*_t = argmin_beta KL(q_g || softmax(beta*ell)). Report:
  - how much beta*_t varies across positions (if ~constant, prefix-adaptive T is dead);
  - in-sample upper bound: per-position-optimal-T vs single global best-T (how much a
    perfectly prefix-adaptive temperature could beat a constant one);
  - does beta*_t correlate with the filter prefix latent (proxy: mean_v theta[v])?
CPU-only; reads records.jsonl.
"""
import json, sys
import numpy as np
from scipy import stats
from decoding_decoding.full_sequence.metrics import softmax, kl_divergence

path, T = sys.argv[1], float(sys.argv[2])
recs = [json.loads(l) for l in open(path) if l.strip()]
Tk = str(T); recs = [r for r in recs if Tk in r["c_by_T"]]

betastar, theta_mean, klpt, kl_global, kl_perpos = [], [], [], [], []
betas = np.linspace(0.05, 8.0, 300)
# global best-T over ALL positions: pick the single beta minimizing mean KL
def kl_at(beta):
    tot=0.0
    for r in recs:
        ell=np.array(r["ell"]); c=np.array(r["c_by_T"][Tk]); g=np.array(r["log_p"])/T
        tot+=kl_divergence(softmax(g+c), softmax(beta*ell))
    return tot/len(recs)
global_best_beta = betas[np.argmin([kl_at(b) for b in betas])]

for r in recs:
    ell=np.array(r["ell"]); c=np.array(r["c_by_T"][Tk]); g=np.array(r["log_p"])/T
    q_g=softmax(g+c)
    klpt.append(kl_divergence(q_g, softmax(g)))
    kls=[kl_divergence(q_g, softmax(b*ell)) for b in betas]
    j=int(np.argmin(kls)); betastar.append(betas[j]); kl_perpos.append(kls[j])
    kl_global.append(kl_divergence(q_g, softmax(global_best_beta*ell)))
    theta_mean.append(float(np.mean(r["theta"])))

D=np.mean(klpt)
bs=np.array(betastar)
out=[]
out.append(f"T={T} n={len(recs)} D_total={D:.4f} global_best_beta={global_best_beta:.3f}")
out.append(f"beta*_t per-position: mean={bs.mean():.3f} std={bs.std():.3f} min={bs.min():.3f} max={bs.max():.3f}")
out.append(f"frac recovered by GLOBAL best-T          = {1-np.mean(kl_global)/D:+.3f}")
out.append(f"frac recovered by PER-POSITION best-T    = {1-np.mean(kl_perpos)/D:+.3f}  (in-sample upper bound on prefix-adaptive temperature)")
out.append(f"  => beyond-global-T headroom from adaptive T = {(np.mean(kl_global)-np.mean(kl_perpos))/D:+.3f} of D")
if np.std(bs)>1e-9 and np.std(theta_mean)>1e-9:
    rho=stats.spearmanr(bs, theta_mean).correlation
    out.append(f"corr(beta*_t, filter prefix latent proxy) Spearman={rho:+.3f}")
open("/tmp/steelman_out.txt","a").write("\n".join(out)+"\n\n")
print("\n".join(out))
