"""Visualize v3 results directly from measured JSON; no synthetic plotted values."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
rows=json.loads((p/'results/v3_elliptic.json').read_text())['rows']
labels=[r['case'] for r in rows]
uniform=[next(c['evaluation_metrics']['relative_l2'] for c in r['candidates'] if c['kind']=='uniform' and c['parameter']==1.5) for r in rows]
selected=[r['candidates'][r['selected_index']]['evaluation_metrics']['relative_l2'] for r in rows]
fig,ax=plt.subplots(figsize=(8,4.6))
x=list(range(len(rows)))
ax.bar([t-.18 for t in x],uniform,.36,label='uniform 1.5')
ax.bar([t+.18 for t in x],selected,.36,label='guarded v3 selection')
ax.set_xticks(x,labels)
ax.set_yscale('log');ax.set_ylabel('relative L2 error')
ax.set_title('65-coefficient elliptic pilot: independent-energy radius selection')
ax.legend();fig.tight_layout()
out=p/'results/v3_elliptic_accuracy.png';fig.savefig(out,dpi=160)
plt.close(fig);print(out)
