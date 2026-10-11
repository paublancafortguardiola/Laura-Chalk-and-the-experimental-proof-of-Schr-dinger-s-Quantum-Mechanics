# Laura Chalk and the Experimental Evidence for Quantum Mechanics

The python script main.py computes the relative intensities of the Stark component's of the Hydrogen Balmer series using Schrödinger's  theory of Quantum mechanics.  The results are compared with 
Laura Chalk's experimental measurements measurements (chalk_observed_intensities.txt) and Kramer's theoretical results (kramers_intensities.txt) 

 
## Usage
 
 
Set the line and polarization at the top of `main()` in `main.py`:
 
```python
n1, n2 = 2, 6                 # Balmer series: n1 = 2, n2 = 3..6 (H-alpha ... H-delta)
polarization = "parallel"     # or "transverse"
```
 
The figure is saved as `stark_H<greek>_<polarization>.pdf`.
 
