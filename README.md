# Conditional Variational AutoEncoder-based on Stochastic Attacks
The current repository is associated with the article "Conditional Variational AutoEncoder-based on Stochastic Attacks" that is under a reviewing process.

This project has been developed in Python 3.6.9.

This project is composed of the following scripts and repositories:
- <b>training_phase.py</b>: provides the training process interface (<i>ie.</i> projection to the orthonormal monomial basis, construction of the cVAE-SA, hyperparameters' setting, application of the training process);
- <b>cvaest_architecture.py</b>: implements the cVAE-SA architecture (<i>ie.</i> the encoder and the decoder) as well as the ELBO, the reconstruction and the KL-divergence loss functions;
- <b>key_recovery_phase.py</b>: conducts the key recovery phase which follows the modus operandi detailed in Sec.3.4 in the paper;
- <b>log.py</b>: captures and saves the training process log;
- <b>plot_history.py</b>: plots the evolution of the loss functions (<i>ie.</i> the ELBO loss, the reconstruction loss and the KL-divergence loss);
- <b>"training_history"</b>: contains information related to the ELBO, the reconstruction and the KL-divergence loss functions,
- <b>"trained_models"</b>: containts the model trained with the "training_phase.py" script.
- <b>"models_in_paper"</b>: containts the models used in the article.

All the simulated traces used in this paper are fully available in the "Simulated_traces.zip" zip file. In addition, an additional script, namely "simulation_trace_exemple.py" is proposed to conduct additional simulations with other setup configurations. The use of this script uses the <a href='https://github.com/Ledger-Donjon/lascar'>Lascar library (Ledger-Donjon)</a> for computing the Signal-to-Noise Ratio (SNR).

## Raw data files hashes
The zip file SHA-256 hash value is:
<hr>

**Simulated_traces.zip:**
`5f9aceeabaaca2d4d9bb356f81fa47c795d25172bef5c9e10882fc93698c8869`

<hr>
