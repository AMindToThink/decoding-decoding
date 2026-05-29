"""Scrutinize the NO-GO: is the ~0 beyond-temperature recovery an artifact of
cross-position pooling? Compute PER-POSITION ceilings (generous, in-sample) that a
function of the candidate logit/latent could reach, plus rank diagnostics.

Reads a records.jsonl produced by the runner. CPU-only.
"""
import json, sys
import numpy as np
from scipy import stats
from decoding_decoding.full_sequence.metrics import softmax, kl_divergence

path = sys.argv[1]
T = float(sys.argv[2])
recs = [json.loads(l) for l in open(path) if l.strip()]
Tk = str(T)
recs = [r for r in recs if Tk in r["c_by_T"]]

def best_T_kl(g_pt, ell, c):
    q_g = softmax(g_pt + c); best = 1e9
    for beta in np.linspace(0.05, 8.0, 200):
        best = min(best, kl_divergence(q_g, softmax(beta*ell)))
    return best

incr_quad, incr_free, spearman, r2_quad = [], [], [], []
klpt_all, klbt_all = [], []
for r in recs:
    ell = np.array(r["ell"]); c = np.array(r["c_by_T"][Tk])
    g_pt = np.array(r["log_p"])/T
    s = ell - r["E_q"]
    q_g = softmax(g_pt + c)
    klpt = kl_divergence(q_g, softmax(g_pt)); klpt_all.append(klpt)
    klbt = best_T_kl(g_pt, ell, c); klbt_all.append(klbt)
    # per-position quadratic-in-logit (in-sample, generous upper bound)
    A = np.vstack([np.ones_like(s), s, s**2]).T
    coef,*_ = np.linalg.lstsq(A, c, rcond=None)
    pred = A@coef
    ss_res = np.var(c - pred); ss_tot = np.var(c)
    r2_quad.append(1 - ss_res/ss_tot if ss_tot>1e-12 else 0.0)
    klq = kl_divergence(q_g, softmax(g_pt + pred))
    incr_quad.append(klbt - klq)  # >0 => quadratic beats best-T
    # per-position FREE function of logit: isotonic-ish via 8-bin means (in-sample)
    order = np.argsort(s); cb = c.copy()
    nb=8; edges=np.quantile(s, np.linspace(0,1,nb+1)); edges=np.unique(edges)
    if len(edges)>=2:
        bi=np.clip(np.digitize(s, edges[1:-1]),0,len(edges)-2)
        predf=np.array([c[bi==b].mean() if (bi==b).any() else 0.0 for b in range(len(edges)-1)])[bi]
    else:
        predf=np.full_like(c, c.mean())
    klf=kl_divergence(q_g, softmax(g_pt+predf))
    incr_free.append(klbt-klf)
    if np.std(c)>1e-9 and np.std(s)>1e-9:
        spearman.append(stats.spearmanr(c, s).correlation)

D=np.mean(klpt_all)
out=[]
out.append(f"T={T} n={len(recs)} D_total={D:.4f} mean_best_T_kl={np.mean(klbt_all):.4f} fracBestT={1-np.mean(klbt_all)/D:.3f}")
out.append(f"per_pos_quadratic_logit: mean_incr_over_bestT={np.mean(incr_quad):.5f} as_frac_of_D={np.mean(incr_quad)/D:+.3f}  (R2_quad mean={np.nanmean(r2_quad):.3f})")
out.append(f"per_pos_free8bin_logit:  mean_incr_over_bestT={np.mean(incr_free):.5f} as_frac_of_D={np.mean(incr_free)/D:+.3f}  (in-sample upper bound)")
out.append(f"per_pos_spearman(c, logit): mean={np.nanmean(spearman):+.3f}  median={np.nanmedian(spearman):+.3f}  frac|rho|>0.3={np.mean(np.abs(spearman)>0.3):.2f}")
open("/tmp/scrutiny_out.txt","a").write("\n".join(out)+"\n\n")
print("\n".join(out))
