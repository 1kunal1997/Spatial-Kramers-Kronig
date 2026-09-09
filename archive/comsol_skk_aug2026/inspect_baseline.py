"""Inspect sKK_PML_baseline.mph: verify geometry numbering and all selections."""
import os, sys
import numpy as np
import mph

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'Comsol Files', 'sKK_PML_baseline.mph')

print('starting COMSOL client...', flush=True)
client = mph.start()
print('loading', MODEL, flush=True)
model = client.load(MODEL)
jm = model.java
geom = jm.component('comp1').geom('geom1')

nd = int(geom.getNDomains())
nb = int(geom.getNBoundaries())
print(f'\ngeometry: {nd} domains, {nb} boundaries')

# vertex coords + boundary->vertex adjacency -> boundary midpoints
vc = np.array(geom.getVertexCoord())          # (sdim, nv)
adj_raw = geom.getAdj(1, 0)                   # ragged: per-boundary vertex indices
adj = [list(np.array(r).ravel()) for r in adj_raw]
print(f'vertexCoord shape {vc.shape}; getAdj returned {len(adj)} rows (row 0 empty, 1-based)')
print('\nvertices (um):')
for i in range(vc.shape[1]):
    print(f'  v{i+1}: x={vc[0,i]*1e6:+.4f}  y={vc[1,i]*1e6:+.4f}')

print('\nboundary midpoints (x, y) in um:')
mids = {}
for b in range(1, nb + 1):
    vs = [int(v) - 1 for v in adj[b]]          # adj rows and vertex ids are 1-based
    xy = vc[:, vs].mean(axis=1)
    mids[b] = xy
    print(f'  bnd {b}: x={xy[0]*1e6:+.4f}  y={xy[1]*1e6:+.4f}   verts={[v+1 for v in vs]}')

ud = np.array(geom.getUpDown())
print('\nboundary up/down domains:')
for b in range(nb):
    print(f'  bnd {b+1}: down={ud[0][b]}  up={ud[1][b]}')

def sel(node, label):
    try:
        s = list(np.array(node.selection().entities()).ravel())
    except Exception:
        try:
            s = list(np.array(node.selection().inputEntities()).ravel())
        except Exception as e:
            s = f'<{e}>'
    print(f'  {label:34s} {s}')

print('\nselections:')
sel(jm.component('comp1').coordSystem('pml1'), 'PML domains')
sel(jm.component('comp1').material('mat1'), 'material background domains')
sel(jm.component('comp1').material('mat2'), 'material terminator domains')
sel(jm.component('comp1').physics('ewfd').feature('port1'), 'port1 boundaries')
sel(jm.component('comp1').physics('ewfd').feature('pc1'), 'pc1 (Floquet) boundaries')

print('\nparameters:')
for k, v in model.parameters().items():
    print(f'  {k:8s} = {v}')

print('\nkey physics props:')
for prop, name in [('components', 'components'), ]:
    print(f'  {name} = {jm.component("comp1").physics("ewfd").prop("components").getString("components")}')
ef = jm.component('comp1').physics('ewfd').prop('EquationForm')
for p in ['form', 'freq_src', 'freq']:
    print(f'  EquationForm.{p} = {ef.getString(p)}')
p1 = jm.component('comp1').physics('ewfd').feature('port1')
for p in ['PortType', 'alpha_inc', 'PortSlit']:
    try: print(f'  port1.{p} = {p1.getString(p)}')
    except Exception as e: print(f'  port1.{p} -> {e}')
pml = jm.component('comp1').coordSystem('pml1')
for p in ['stretchingType', 'PMLfactor', 'PMLgamma', 'typicalWavelength', 'wavelengthSourceType']:
    try: print(f'  pml1.{p} = {pml.getString(p)}')
    except Exception as e: print(f'  pml1.{p} -> {e}')

client.clear()
print('\ndone.')
