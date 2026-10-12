# Laura Chalk and the Experimental Proof of Schrödinger's Quantum Mechanics

The python script main.py computes the relative intensities of the Stark component's of the Hydrogen Balmer series using Schrödinger's  theory of Quantum mechanics.  The results are compared with 
Laura Chalk's experimental measurements measurements (chalk_observed_intensities.txt) and Kramer's theoretical results (kramers_intensities.txt) 

Usage: 
 
Set the line and polarization at the top of `main()` in `main.py`:
 
```python
n1, n2 = 2, 6                 # Balmer series: n1 = 2, n2 = 3..6 (H-alpha ... H-delta)
polarization = "parallel"     # or "transverse"
```
 
The python script stark.py computes the perturbed states in the presence of an Electric field and plots the rresulting perturbed energies, as well as the ovrlap of the perturbed eigenstates in terms of the unperturbed basis.
