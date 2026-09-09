"""Reproduce Kunal's four sanity checks on sKK_PML_baseline.mph from Python."""
import os
import numpy as np
import mph

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'Comsol Files', 'sKK_PML_baseline.mph')
OUT = os.path.dirname(os.path.abspath(__file__))

client = mph.start()
model = client.load(MODEL)

studies = model.studies()
print('studies:', studies)
STUDY = studies[0]
print(f'solving theta sweep 0..89 deg via {STUDY!r} ...', flush=True)
model.solve(STUDY)

def ev(expr):
    return np.asarray(model.evaluate(expr)).ravel()

R    = ev('ewfd.Rtotal')
A    = ev('ewfd.Atotal')
RT   = ev('ewfd.RTtotal')
try:
    Lsca = ev('ewfd.Lsca')
except Exception as e:
    Lsca = np.full_like(R, np.nan); print('Lsca unavailable:', e)
th   = np.linspace(0, 89, len(R))

np.savez(os.path.join(OUT, 'pml_baseline_sweep.npz'),
         theta=th, R=R, A=A, RT=RT, Lsca=Lsca)

print(f'\n{len(R)} angles solved\n')
print(f'{"theta":>6} {"R":>12} {"A":>12} {"R+A":>10} {"Lsca":>10}')
for t in [0, 10, 18, 30, 50, 60, 70, 80, 85, 89]:
    i = int(np.argmin(abs(th - t)))
    print(f'{th[i]:>6.0f} {R[i]:>12.4e} {A[i]:>12.6f} {R[i]+A[i]:>10.6f} {Lsca[i]:>10.2e}')

print('\n--- checks ---')
print(f'1. R(0)          = {R[0]:.4e}      (want < 1e-5)                 {"PASS" if R[0] < 1e-5 else "FAIL"}')
mono = np.all(np.diff(R[20:]) > 0)
print(f'2. rises to grazing, R(89) = {R[-1]:.4f}, monotone above 20 deg   {"PASS" if mono else "note: not monotone"}')
err = np.max(np.abs(R + A - 1.0))
print(f'3. max|R+A-1|    = {err:.3e}                                     {"PASS" if err < 1e-6 else "FAIL"}')
print(f'4. max R         = {R.max():.4f}                                  {"PASS" if R.max() <= 1.0 else "FAIL"}')

# fit R = exp(C cos theta) on the absorption-limited branch
m = (th >= 30) & (th <= 89)
C = np.polyfit(np.cos(np.radians(th[m])), np.log(R[m]), 1)
print(f'\ncos-theta fit on 30-89 deg:  ln R = {C[0]:.3f} * cos(theta) + {C[1]:.3f}')
print(f'{"theta":>6} {"R measured":>12} {"R fit":>12} {"ratio":>8}')
for t in [30, 50, 60, 70, 80, 89]:
    i = int(np.argmin(abs(th - t)))
    fit = np.exp(C[0]*np.cos(np.radians(th[i])) + C[1])
    print(f'{th[i]:>6.0f} {R[i]:>12.4e} {fit:>12.4e} {R[i]/fit:>8.2f}')

client.clear()
print('\nsaved -> comsol_skk/pml_baseline_sweep.npz')
