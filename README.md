\# ballistics-sim



Projectile flight simulation in Python (NumPy, Matplotlib).

Goal: go from an ideal throw in a vacuum to a realistic atmosphere, where air

density and gravity change with altitude.



> Work in progress. This is a learning project.



\## Status



\- \[x] No-drag trajectory (`ballistics/physics.py`), checked against formulas

\- \[ ] Plots for different launch angles

\- \[ ] Numerical integration (Euler, RK4, `solve\_ivp`)

\- \[ ] Air drag

\- \[ ] Altitude-dependent gravity and air density

\- \[ ] Animation and interface



\## Quick start



&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt

&#x20;   python -m pytest



\## Project layout



\- `ballistics/`: physics code (no plotting)

\- `tests/`: tests that compare the model with known formulas

