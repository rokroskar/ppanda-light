## ppanda-renku

# PPANDA
Pulse Profiles of Accreting Neutron stars Deeply Analyzed

_C. Ferrigno, E. Ambrosi, A. D'A&igrave;, D. K. Maniadakis_

Papers:
- [Ferrigno et al. 2023](https://doi.org/10.1051/0004-6361/202347062)
- [D'A&igrave; et al. 2025](https://doi.org/10.1051/0004-6361/202451469)

[![launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/p/carlo.ferrigno/ppanda-light/sessions/01JS4MC44NM7AGSD9P2S7RV6N5/start)

This will launch the session in renkulab 2.0

The pulse profiles of accreting X-ray binaries as seen by NuSTAR, presented as a streamlit app.

The notebook that re-runs the product extraction and is the actual workflow is called `model_2023-05-25.ipynb`.

It relies on some pre-ccocked data inserted as LFS files. 

This is the second stage of the workflow, it relies on the presence of energy-phase matrices to compute the pulsed fraction and model its energy dependency. 

## Building
This is derived from a streamlit template based on the basic Python (3.10) template.

