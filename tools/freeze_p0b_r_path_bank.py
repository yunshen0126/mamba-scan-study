import os,json,hashlib,math,sys,time,platform
from pathlib import Path
import numpy as np, torch
SRC=Path(os.environ['P0B_CONFIG_SOURCE']); ROOT=Path('.'); L=ROOT/'P0B_L_PATH_BANK_FROZEN.json'; O=ROOT/'P0B_R_PATH_BANK_FROZEN.json'; R=ROOT/'REPORT_B3_R_PATH_BANK_FREEZE.md'
E=['31388e81da18d0ef8929390adeaa1b115ca4120c78f81d8fbcafbf168aed3ffe','93a41e67f539b469a8c2855bc577805d4dc6a7ffcb8c648b11097c9d58ffbec7','3e79d2f8c941f7c54f11eaee21332265d9d064a9fb9971169fa18a6295d3cc8c']
H=lambda b:hashlib.sha256(b).hexdigest(); HF=lambda p:H(p.read_bytes()); HO=lambda a:H(np.asarray(a,dtype=np.int64).tobytes())
def pi(o): x=np.empty(len(o),dtype=np.int64);x[o]=np.arange(len(o));return x
def edges(n):
 x=np.arange(n*n).reshape(n,n);return [(x[:,:-1].ravel(),x[:,1:].ravel()),(x[:,1:].ravel(),x[:,:-1].ravel()),(x[:-1,:].ravel(),x[1:,:].ravel()),(x[1:,:].ravel(),x[:-1,:].ravel())]
def cd(ps,n):
 z=[]
 for u,v in edges(n):
  q=np.full(len(u),np.inf)
  for p in ps:
   d=p[v]-p[u];ok=d>0;q[ok]=np.minimum(q[ok],d[ok])
  z.append([float(np.mean(q<=t*(n*n-1))) for t in (.01,.05,.1,.2)])
 return z
def auc(a):
 trapezoid=getattr(np,"trapezoid",None)
 if trapezoid is None: trapezoid=getattr(np,"trapz",None)
 if trapezoid is None: raise RuntimeError("NumPy provides neither trapezoid nor trapz in the approved mair environment")
 return float(trapezoid([0]+a,[0,.01,.05,.1,.2])/.2)
def report_only():
 t=time.perf_counter(); p=O; before=HF(p); assert before=='2f7b8a6fd3cfbbae9897b4ef4dc9dcfd1bf7744619d5818ceaca7604d565aee3'; data=json.loads(p.read_text()); lines=['# B3 R Path Bank Freeze','',f'command: `conda run -n mair python tools/freeze_p0b_r_path_bank.py --report-only`',f'python: {sys.executable}',f'numpy: {np.__version__} ({np.__file__})',f'wall_time_seconds: {time.perf_counter()-t:.6f}',f'frozen_json_sha256_before: `{before}`','', 'All metrics below are diagnostics only: no redraw, filtering, seed change, or frozen-path mutation occurred.','','| n | set | path | seed | order SHA | inverse SHA | legal | replay | isolation | mean | p50 | p90 | p95 | max | d_x | d_y | AxisBias |','|---:|:--|:--|---:|:--|:--|:--|:--|:--|---:|---:|---:|---:|---:|---:|---:|---:|']
 for gr in data['grids']:
  n=gr['n']; x=np.arange(n*n).reshape(n,n); hu,hv=x[:,:-1].ravel(),x[:,1:].ravel(); vu,vv=x[:-1,:].ravel(),x[1:,:].ravel()
  for s,ss in gr['sets'].items():
   ps=[]
   for name,v in ss['paths'].items():
    o=np.array(v['order']); q=pi(o); d=np.abs(q[np.r_[hu,vu]]-q[np.r_[hv,vv]]); dx=float(np.mean(np.abs(q[hu]-q[hv])));dy=float(np.mean(np.abs(q[vu]-q[vv])));ps.append(q);lines.append(f'| {n} | {s} | {name} | {v["seed"]} | `{v["order_sha256"]}` | `{v["inverse_order_sha256"]}` | PASS | PASS | PASS | {np.mean(d):.6f} | {np.percentile(d,50):.6f} | {np.percentile(d,90):.6f} | {np.percentile(d,95):.6f} | {np.max(d)} | {dx:.6f} | {dy:.6f} | {math.log(dx/dy):.6f} |')
   z=cd(ps,n);a=[auc(k) for k in z];lines += ['',f'## n={n} {s}',f'collection C_dir RIGHT/LEFT/DOWN/UP: {z}',f'AUC_dir: {a}; AUC_macro: {sum(a)/4:.12f}']
 lines += ['',f'frozen_json_sha256_after: `{HF(p)}`']; assert HF(p)==before; R.write_text('\n'.join(lines)+'\n')
def main():
 assert HF(SRC)==E[0] and HF(L)==E[1] and HF(ROOT/'docs/P0A_B0_FORMAL_DEFINITIONS.md')==E[2]
 l=json.loads(L.read_text()); grids=[]; lines=['# B3 R Path Bank Freeze','']
 for n in (8,32):
  g=[np.arange(n*n),np.arange(n*n)[::-1],np.arange(n*n).reshape(n,n).T.ravel(),np.arange(n*n).reshape(n,n).T.ravel()[::-1]]; lp=[pi(np.array(l['grids'][0 if n==8 else 1]['paths'][f'L{i}']['order'])) for i in range(1,5)]
  a=[auc(x) for x in cd([pi(x) for x in g],n)]; b=[auc(x) for x in cd([pi(g[0]),pi(g[2])],n)]; c=[auc(x) for x in cd(lp,n)]; exp=(.85,.425,.7991071428571429) if n==8 else (.975,.4875,.9645413306451613);assert all(math.isclose(x,exp[0],abs_tol=1e-12) for x in a) and math.isclose(sum(b)/4,exp[1],abs_tol=1e-12) and all(math.isclose(x,exp[2],abs_tol=1e-12) for x in c)
  sets={};seen=[]
  for s in (1,2,3):
   q={}
   for i in (1,2,3,4):
    seed=17071+1000*s+i; gen=torch.Generator(device='cpu');gen.manual_seed(seed);o=torch.randperm(n*n,generator=gen,device='cpu',dtype=torch.int64).numpy(); gen=torch.Generator(device='cpu');gen.manual_seed(seed);assert np.array_equal(o,torch.randperm(n*n,generator=gen,device='cpu',dtype=torch.int64).numpy()) and np.array_equal(np.sort(o),np.arange(n*n));p=pi(o);assert np.array_equal(p[o],np.arange(n*n)) and not any(np.array_equal(o,x) for x in seen+g+[x for y in l['grids'][0 if n==8 else 1]['paths'].values() for x in [np.array(y['order'])]]);seen.append(o);q[f'R{s}_{i}']={'set_id':f'S{s}','path_id':f'R{s}_{i}','seed':seed,'order':o.tolist(),'order_sha256':HO(o),'inverse_order_sha256':HO(p)}
   ps=[pi(np.array(x['order'])) for x in q.values()];d=cd(ps,n);sets[f'S{s}']={'paths':q,'c_dir':d,'auc_dir':[auc(x) for x in d],'auc_macro':sum(auc(x) for x in d)/4}
  grids.append({'n':n,'N':n*n,'sets':sets});lines.append(f'n={n}: 12/12 legality/replay/uniqueness/G-L isolation PASS; G4/G13/LMTO AUC macros={sum(a)/4:.12f}/{sum(b)/4:.12f}/{sum(c)/4:.12f}')
 payload={'schema_version':'1.0','status':'FROZEN_FOR_P0B_FEASIBILITY','freeze_date':'2026-07-20','freeze_scope':'P0-B feasibility pilot','decision_record':'P0B_PREREG_FREEZE_R_PATHS.md','generator_name':'torch_cpu_randperm_presampled','generator_spec':'new CPU Generator for each (n,s,i); seed=17071+1000*s+i','torch_version':torch.__version__,'numpy_version':np.__version__,'source_config_original_path':str(SRC),'source_config_original_sha256':HF(SRC),'repo_copy_before_edit_sha256':HF(SRC),'repo_config_final_sha256':HF(ROOT/'docs/P0B_CONFIG_TABLE.md'),'training_mapping':{'RND_Ss':'seed 0/1/2/3 -> R^s_1/R^s_2/R^s_3/R^s_4, repeated on four channels','RND_Ds':'fixed R^s_1..R^s_4, existing Latin square channel rotation'},'grids':grids};O.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n');lines+=['',f'frozen_json_sha256: {HF(O)}','24/24 PASS; no redraw or selection.'];R.write_text('\n'.join(lines)+'\n');print(HF(O))
if '--report-only' in sys.argv: report_only()
else: main()
