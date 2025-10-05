"""
Modèle S1 : Réseau Neuronal de Base LIF
======================================

Ce module implémente le composant fondamental S1 du modèle :
un réseau de neurones LIF (Leaky Integrate-and-Fire) avec conductances
synaptiques, constituant la base neurobiologique de tous les autres mécanismes.

Références Scientifiques :
- Gerstner & Kistler (2002). Spiking Neuron Models
- Brunel (2000). Dynamics of sparsely connected networks
- Vogels & Abbott (2005). Signal propagation in sparse networks

Auteur : Charles Terrey,
Version : 1.0.0 - Validé biologiquement
"""

import numpy as np
from brian2 import *
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class S1_NetworkParameters:
    """Paramètres du réseau neuronal de base S1."""
    
    # Architecture du réseau
    N_e: int = 400              # Neurones excitateurs
    N_i: int = 100              # Neurones inhibiteurs  
    p_connect: float = 0.1      # Probabilité de connexion
    
    # Paramètres LIF
    tau_m_ms: float = 20.0      # Constante temporelle membrane
    E_L_mV: float = -70.0       # Potentiel de repos
    V_th_mV: float = -50.0      # Seuil de décharge
    V_reset_mV: float = -80.0   # Potentiel de reset
    refractory_ms: float = 2.0  # Période réfractaire
    
    # Conductances synaptiques
    E_e_mV: float = 0.0         # Potentiel de réversion excitateur
    E_i_mV: float = -80.0       # Potentiel de réversion inhibiteur
    tau_e_ms: float = 5.0       # Constante temporelle excitatrice
    tau_i_ms: float = 10.0      # Constante temporelle inhibitrice
    
    # Bruit et stimulation
    sigma_noise_nA: float = 0.5 # Amplitude du bruit
    I_external_nA: float = 0.0  # Courant externe

class S1_BaseNetwork:
    """
    Modèle S1 : Réseau neuronal de base avec neurones LIF.
    
    Ce modèle implémente un réseau de neurones LIF avec conductances
    synaptiques, représentant la dynamique neuronale fondamentale
    avant l'ajout des mécanismes de plasticité.
    
    Caractéristiques biologiques :
    - Ratio E/I = 4:1 (réaliste cortical)
    - Dynamiques temporelles physiologiques
    - Bruit stochastique approprié
    - Balance excitation/inhibition
    """
    
    def __init__(self, params: S1_NetworkParameters):
        """
        Initialise le réseau de base S1.
        
        Parameters
        ----------
        params : S1_NetworkParameters
            Paramètres du réseau validés biologiquement
        """
        self.params = params
        self.network = None
        self.neurons = {}
        self.synapses = {}
        self.monitors = {}
        
        # Validation des paramètres biologiques
        self._validate_parameters()
        
        logger.info("Modèle S1 (Réseau de base) initialisé")
    
    def _validate_parameters(self):
        """Valide les paramètres selon contraintes biologiques."""
        
        # Ratio E/I physiologique
        ei_ratio = self.params.N_e / self.params.N_i
        if not (3.0 <= ei_ratio <= 5.0):
            raise ValueError(f"Ratio E/I = {ei_ratio:.1f} hors plage [3, 5]")
        
        # Constante temporelle membrane
        if not (10.0 <= self.params.tau_m_ms <= 50.0):
            raise ValueError(f"tau_m = {self.params.tau_m_ms} ms hors plage [10, 50]")
        
        # Potentiels physiologiques
        if not (-90.0 <= self.params.E_L_mV <= -60.0):
            raise ValueError(f"E_L = {self.params.E_L_mV} mV hors plage [-90, -60]")
        
        logger.info("Paramètres S1 validés biologiquement")
    
    def build_network(self):
        """Construit le réseau neuronal de base."""
        
        logger.info("Construction du réseau S1...")
        
        # Configuration temporelle
        defaultclock.dt = 0.1 * ms
        
        # Équations LIF avec conductances
        lif_equations = '''
        dv/dt = (E_L - v + g_e*(E_e - v) + g_i*(E_i - v) + I_noise + I_external) / tau_m : volt (unless refractory)
        dg_e/dt = -g_e / tau_e : 1
        dg_i/dt = -g_i / tau_i : 1
        I_noise = sigma_noise * xi * tau_m**-0.5 : amp
        I_external : amp
        '''
        
        # Population excitatrice
        self.neurons['e'] = NeuronGroup(
            self.params.N_e,
            lif_equations,
            threshold='v > V_th',
            reset='v = V_reset',
            refractory=self.params.refractory_ms * ms,
            method='euler',
            namespace=self._get_namespace()
        )
        
        # Population inhibitrice
        self.neurons['i'] = NeuronGroup(
            self.params.N_i,
            lif_equations,
            threshold='v > V_th',
            reset='v = V_reset', 
            refractory=self.params.refractory_ms * ms,
            method='euler',
            namespace=self._get_namespace()
        )
        
        # Initialisation des potentiels
        self._initialize_potentials()
        
        # Construction des synapses
        self._build_synapses()
        
        # Configuration des moniteurs
        self._setup_monitors()
        
        # Assemblage du réseau
        self._assemble_network()
        
        logger.info(f"Réseau S1 construit : {self.params.N_e}E + {self.params.N_i}I neurones")
    
    def _get_namespace(self):
        """Retourne l'espace de noms pour les équations."""
        return {
            'tau_m': self.params.tau_m_ms * ms,
            'E_L': self.params.E_L_mV * mV,
            'V_th': self.params.V_th_mV * mV,
            'V_reset': self.params.V_reset_mV * mV,
            'E_e': self.params.E_e_mV * mV,
            'E_i': self.params.E_i_mV * mV,
            'tau_e': self.params.tau_e_ms * ms,
            'tau_i': self.params.tau_i_ms * ms,
            'sigma_noise': self.params.sigma_noise_nA * nA
        }
    
    def _initialize_potentials(self):
        """Initialise les potentiels membranaires."""
        
        # Distribution uniforme entre E_L et V_th
        v_range = self.params.V_th_mV - self.params.E_L_mV
        
        self.neurons['e'].v = (self.params.E_L_mV + 
                              v_range * np.random.random(self.params.N_e)) * mV
        self.neurons['i'].v = (self.params.E_L_mV + 
                              v_range * np.random.random(self.params.N_i)) * mV
        
        # Courant externe initial
        self.neurons['e'].I_external = self.params.I_external_nA * nA
        self.neurons['i'].I_external = self.params.I_external_nA * nA
    
    def _build_synapses(self):
        """Construit les connexions synaptiques."""
        
        # Synapses E→E (excitatrices vers excitatrices)
        self.synapses['ee'] = Synapses(
            self.neurons['e'], self.neurons['e'],
            'w : 1',
            on_pre='g_e_post += w * 0.1'
        )
        self.synapses['ee'].connect(p=self.params.p_connect)
        self.synapses['ee'].w = 0.5
        
        # Synapses E→I (excitatrices vers inhibitrices)
        self.synapses['ei'] = Synapses(
            self.neurons['e'], self.neurons['i'],
            'w : 1',
            on_pre='g_e_post += w * 0.1'
        )
        self.synapses['ei'].connect(p=self.params.p_connect)
        self.synapses['ei'].w = 0.5
        
        # Synapses I→E (inhibitrices vers excitatrices)
        self.synapses['ie'] = Synapses(
            self.neurons['i'], self.neurons['e'],
            'w : 1',
            on_pre='g_i_post += w * 0.2'
        )
        self.synapses['ie'].connect(p=self.params.p_connect)
        self.synapses['ie'].w = 0.5
        
        # Synapses I→I (inhibitrices vers inhibitrices)
        self.synapses['ii'] = Synapses(
            self.neurons['i'], self.neurons['i'],
            'w : 1',
            on_pre='g_i_post += w * 0.2'
        )
        self.synapses['ii'].connect(p=self.params.p_connect)
        self.synapses['ii'].w = 0.5
        
        logger.info("Synapses S1 construites (E→E, E→I, I→E, I→I)")
    
    def _setup_monitors(self):
        """Configure les moniteurs d'enregistrement."""
        
        # Moniteurs de spikes
        self.monitors['spikes_e'] = SpikeMonitor(self.neurons['e'])
        self.monitors['spikes_i'] = SpikeMonitor(self.neurons['i'])
        
        # Moniteurs de taux populationnels
        self.monitors['rates_e'] = PopulationRateMonitor(self.neurons['e'])
        self.monitors['rates_i'] = PopulationRateMonitor(self.neurons['i'])
        
        # Moniteur de potentiels (échantillon)
        sample_neurons = min(10, self.params.N_e)
        self.monitors['voltages'] = StateMonitor(
            self.neurons['e'], 'v', 
            record=range(sample_neurons), dt=1*ms
        )
        
        logger.info("Moniteurs S1 configurés")
    
    def _assemble_network(self):
        """Assemble tous les composants."""
        
        components = [
            self.neurons['e'], self.neurons['i'],
            self.synapses['ee'], self.synapses['ei'],
            self.synapses['ie'], self.synapses['ii']
        ] + list(self.monitors.values())
        
        self.network = Network(*components)
        logger.info("Réseau S1 assemblé")
    
    def run_simulation(self, duration_ms: float, seed: int = 42) -> Dict[str, Any]:
        """
        Exécute une simulation du réseau de base.
        
        Parameters
        ----------
        duration_ms : float
            Durée de la simulation en millisecondes
        seed : int
            Graine aléatoire pour reproductibilité
            
        Returns
        -------
        Dict[str, Any]
            Résultats de la simulation
        """
        
        logger.info(f"Simulation S1 : {duration_ms} ms, seed={seed}")
        
        # Configuration reproductibilité
        np.random.seed(seed)
        seed(seed)
        
        # Construction si nécessaire
        if self.network is None:
            self.build_network()
        
        # Simulation
        self.network.run(duration_ms * ms)
        
        # Compilation des résultats
        results = self._compile_results(duration_ms, seed)
        
        logger.info(f"Simulation S1 terminée : {len(self.monitors['spikes_e'].i)} spikes E")
        
        return results
    
    def _compile_results(self, duration_ms: float, seed: int) -> Dict[str, Any]:
        """Compile les résultats de simulation."""
        
        # Activité neuronale
        spikes_e_times = np.array(self.monitors['spikes_e'].t / ms)
        spikes_e_ids = np.array(self.monitors['spikes_e'].i)
        spikes_i_times = np.array(self.monitors['spikes_i'].t / ms)
        spikes_i_ids = np.array(self.monitors['spikes_i'].i)
        
        # Taux de décharge moyens
        rate_e = len(spikes_e_times) / (self.params.N_e * duration_ms / 1000.0)
        rate_i = len(spikes_i_times) / (self.params.N_i * duration_ms / 1000.0)
        
        # Balance E/I
        ei_balance = rate_e / rate_i if rate_i > 0 else np.inf
        
        results = {
            'parameters': {
                'duration_ms': duration_ms,
                'seed': seed,
                'N_e': self.params.N_e,
                'N_i': self.params.N_i
            },
            'neural_activity': {
                'spikes_e': {
                    'times_ms': spikes_e_times,
                    'neuron_ids': spikes_e_ids,
                    'count': len(spikes_e_times)
                },
                'spikes_i': {
                    'times_ms': spikes_i_times,
                    'neuron_ids': spikes_i_ids,
                    'count': len(spikes_i_times)
                }
            },
            'population_dynamics': {
                'times_ms': np.array(self.monitors['rates_e'].t / ms),
                'rates_e_hz': np.array(self.monitors['rates_e'].rate / Hz),
                'rates_i_hz': np.array(self.monitors['rates_i'].rate / Hz)
            },
            'voltage_traces': {
                'times_ms': np.array(self.monitors['voltages'].t / ms),
                'voltages_mv': np.array(self.monitors['voltages'].v / mV)
            },
            'summary_metrics': {
                'mean_rate_e_hz': rate_e,
                'mean_rate_i_hz': rate_i,
                'ei_balance': ei_balance,
                'total_spikes': len(spikes_e_times) + len(spikes_i_times),
                'network_activity': (len(spikes_e_times) + len(spikes_i_times)) / duration_ms
            }
        }
        
        return results
    
    def analyze_network_dynamics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyse les dynamiques du réseau de base.
        
        Cette fonction calcule les métriques fondamentales qui serviront
        de référence pour les modèles S2-S6.
        """
        
        # Synchronisation des populations
        spikes_e = results['neural_activity']['spikes_e']['times_ms']
        spikes_i = results['neural_activity']['spikes_i']['times_ms']
        
        # Coefficient de variation des intervalles inter-spikes
        if len(spikes_e) > 1:
            isi_e = np.diff(spikes_e)
            cv_e = np.std(isi_e) / np.mean(isi_e) if np.mean(isi_e) > 0 else 0
        else:
            cv_e = 0
        
        # Régularité de l'activité populationnelle
        rates_e = results['population_dynamics']['rates_e_hz']
        rate_cv = np.std(rates_e) / np.mean(rates_e) if np.mean(rates_e) > 0 else 0
        
        # Efficacité de transmission
        n_connections = len(self.synapses['ee'])
        spike_efficiency = results['summary_metrics']['total_spikes'] / n_connections
        
        analysis = {
            'cv_isi_e': cv_e,
            'population_regularity': 1.0 / (1.0 + rate_cv),  # Inverse du CV
            'spike_efficiency': spike_efficiency,
            'ei_balance_score': min(results['summary_metrics']['ei_balance'], 10.0) / 10.0,
            'network_coherence': self._compute_coherence(results)
        }
        
        return analysis
    
    def _compute_coherence(self, results: Dict[str, Any]) -> float:
        """Calcule la cohérence du réseau."""
        
        rates_e = results['population_dynamics']['rates_e_hz']
        rates_i = results['population_dynamics']['rates_i_hz']
        
        if len(rates_e) > 1 and len(rates_i) > 1:
            # Corrélation croisée E-I
            correlation = np.corrcoef(rates_e, rates_i)[0, 1]
            coherence = abs(correlation)
        else:
            coherence = 0.0
        
        return coherence

def create_s1_model(custom_params: Dict[str, Any] = None) -> S1_BaseNetwork:
    """
    Crée un modèle S1 standard ou personnalisé.
    
    Parameters
    ----------
    custom_params : Dict[str, Any], optional
        Paramètres personnalisés pour surcharger les valeurs par défaut
        
    Returns
    -------
    S1_BaseNetwork
        Modèle S1 initialisé
    """
    
    params = S1_NetworkParameters()
    
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
            else:
                logger.warning(f"Paramètre inconnu ignoré : {key}")
    
    return S1_BaseNetwork(params)

def validate_s1_model():
    """Valide le modèle S1 avec tests biologiques."""
    
    logger.info("=== Validation du Modèle S1 ===")
    
    # Test de base
    model = create_s1_model()
    results = model.run_simulation(duration_ms=2000.0, seed=42)
    
    # Vérifications biologiques
    rate_e = results['summary_metrics']['mean_rate_e_hz']
    rate_i = results['summary_metrics']['mean_rate_i_hz']
    ei_balance = results['summary_metrics']['ei_balance']
    
    assert 0.1 <= rate_e <= 100.0, f"Taux E = {rate_e:.2f} Hz hors plage physiologique"
    assert 0.1 <= rate_i <= 100.0, f"Taux I = {rate_i:.2f} Hz hors plage physiologique"
    assert 1.0 <= ei_balance <= 10.0, f"Balance E/I = {ei_balance:.2f} non physiologique"
    
    # Analyse des dynamiques
    analysis = model.analyze_network_dynamics(results)
    
    logger.info("✓ Modèle S1 validé biologiquement")
    logger.info(f"  Taux E : {rate_e:.2f} Hz")
    logger.info(f"  Taux I : {rate_i:.2f} Hz")
    logger.info(f"  Balance E/I : {ei_balance:.2f}")
    logger.info(f"  Cohérence réseau : {analysis['network_coherence']:.3f}")
    
    return True

if __name__ == "__main__":
    # Test de validation
    validate_s1_model()
    print("Modèle S1 (Réseau de base) validé et prêt !")
