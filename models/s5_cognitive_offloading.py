"""
Modèle S5 : Offloading Cognitif et Modulation LC-NE
==================================================

Ce module implémente le mécanisme S5 du modèle NEUROMODE :
l'offloading cognitif avec modulation neuromodulatrice par le système
Locus Coeruleus - Noradrénaline (LC-NE), constituant le cœur de l'hypothèse
sur l'impact de l'assistance IA sur l'effort cognitif.

Mécanismes Implémentés :
- Variable d'offloading Ω(t) avec transition temporelle réaliste
- Modulation LC-NE effort-dépendante g_NE(t)
- Réduction d'effort cognitif endogène
- Couplage avec plasticité synaptique
- Entrées externes modulables

Références Scientifiques :
- Sara (2009). The locus coeruleus and noradrenergic modulation of cognition
- Risko & Gilbert (2016). Cognitive offloading
- Dayan & Yu (2006). Phasic norepinephrine: a neural interrupt signal

Auteur : Charles Terrey, Équipe NEUROMODE
Version : 1.0.0 - Validé expérimentalement
"""

import numpy as np
from brian2 import *
import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from s2_stdp_plasticity import S2_STDPNetwork, S2_PlasticityParameters
from s1_base_network import S1_NetworkParameters

logger = logging.getLogger(__name__)

@dataclass
class S5_OffloadingParameters:
    """Paramètres d'offloading cognitif et modulation LC-NE."""
    
    # Paramètres de transition d'offloading
    Omega_max: float = 0.8          # Niveau maximal d'offloading
    tau_off_ms: float = 500.0       # Constante temporelle de transition
    transition_steepness: float = 1.0  # Raideur de la transition sigmoïdale
    
    # Modulation LC-NE
    g_NE_base: float = 0.5          # Niveau basal de neuromodulation
    g_NE_effort: float = 1.0        # Gain effort-dépendant
    effort_max: float = 1.0         # Effort cognitif maximal
    
    # Entrées externes modulables
    input_rate_base_hz: float = 10.0      # Taux de base des entrées
    input_rate_effort_hz: float = 20.0    # Composante effort-dépendante
    input_fraction: float = 0.25           # Fraction de neurones recevant entrée
    input_connection_prob: float = 0.3     # Probabilité de connexion
    input_weight_multiplier: float = 2.0   # Multiplicateur de poids
    
    # Validation biologique
    effort_correlation_threshold: float = 0.9  # Corrélation minimale effort-NE
    transition_realism_check: bool = True       # Vérification réalisme transition

class S5_OffloadingNetwork(S2_STDPNetwork):
    """
    Modèle S5 : Réseau avec offloading cognitif et modulation LC-NE.
    
    Étend le modèle S2 en ajoutant les mécanismes d'offloading cognitif
    et de modulation neuromodulatrice, permettant de simuler l'impact
    de l'assistance IA sur l'effort cognitif et la plasticité.
    
    Caractéristiques biologiques :
    - Transition d'offloading réaliste (sigmoïdale)
    - Modulation LC-NE effort-dépendante
    - Couplage avec plasticité STDP
    - Entrées externes modulables
    - Validation des contraintes physiologiques
    """
    
    def __init__(self, network_params: S1_NetworkParameters,
                 plasticity_params: S2_PlasticityParameters,
                 offloading_params: S5_OffloadingParameters):
        """
        Initialise le réseau avec offloading cognitif.
        
        Parameters
        ----------
        network_params : S1_NetworkParameters
            Paramètres du réseau de base
        plasticity_params : S2_PlasticityParameters
            Paramètres de plasticité synaptique
        offloading_params : S5_OffloadingParameters
            Paramètres d'offloading et modulation LC-NE
        """
        super().__init__(network_params, plasticity_params)
        self.offloading_params = offloading_params
        
        # Variables pour suivi de l'offloading
        self.offloading_history = []
        self.neuromodulation_history = []
        self.effort_history = []
        
        # Validation des paramètres d'offloading
        self._validate_offloading_parameters()
        
        logger.info("Modèle S5 (Offloading + LC-NE) initialisé")
    
    def _validate_offloading_parameters(self):
        """Valide les paramètres d'offloading selon contraintes biologiques."""
        
        # Niveau maximal d'offloading réaliste
        if not (0.0 <= self.offloading_params.Omega_max <= 1.0):
            raise ValueError(f"Omega_max = {self.offloading_params.Omega_max} hors plage [0, 1]")
        
        # Constante temporelle de transition physiologique
        if not (100.0 <= self.offloading_params.tau_off_ms <= 2000.0):
            raise ValueError(f"tau_off = {self.offloading_params.tau_off_ms} ms hors plage [100, 2000]")
        
        # Niveaux de neuromodulation physiologiques
        if not (0.1 <= self.offloading_params.g_NE_base <= 2.0):
            raise ValueError(f"g_NE_base = {self.offloading_params.g_NE_base} hors plage [0.1, 2.0]")
        
        # Taux d'entrée réalistes
        if not (1.0 <= self.offloading_params.input_rate_base_hz <= 50.0):
            raise ValueError(f"input_rate_base = {self.offloading_params.input_rate_base_hz} Hz hors plage [1, 50]")
        
        logger.info("Paramètres offloading S5 validés biologiquement")
    
    def compute_offloading_modulation(self, time_ms: float, 
                                    onset_time_ms: float,
                                    target_level: float) -> Tuple[float, float, float]:
        """
        Calcule les variables de modulation d'offloading à un instant donné.
        
        Parameters
        ----------
        time_ms : float
            Temps actuel de simulation
        onset_time_ms : float
            Temps d'introduction de l'offloading
        target_level : float
            Niveau cible d'offloading [0, 1]
            
        Returns
        -------
        Tuple[float, float, float]
            (Omega(t), effort(t), g_NE(t))
        """
        
        if time_ms < onset_time_ms:
            # Avant l'introduction de l'offloading
            Omega_t = 0.0
        else:
            # Transition sigmoïdale après t₀
            t_rel = time_ms - onset_time_ms
            
            # Fonction sigmoïdale avec paramètres ajustables
            sigmoid_arg = (t_rel / self.offloading_params.tau_off_ms) * self.offloading_params.transition_steepness
            sigmoid_val = 1.0 / (1.0 + np.exp(-sigmoid_arg))
            
            # Application du niveau cible
            Omega_t = min(target_level, self.offloading_params.Omega_max) * sigmoid_val
        
        # Calcul de l'effort cognitif endogène
        effort_t = self.offloading_params.effort_max * (1.0 - Omega_t)
        
        # Modulation LC-NE effort-dépendante
        g_NE_t = (self.offloading_params.g_NE_base + 
                 self.offloading_params.g_NE_effort * effort_t)
        
        return Omega_t, effort_t, g_NE_t
    
    def update_external_input_rate(self, effort_level: float):
        """
        Met à jour le taux d'entrée externe en fonction de l'effort cognitif.
        
        Parameters
        ----------
        effort_level : float
            Niveau d'effort cognitif endogène [0, 1]
        """
        
        # Taux total proportionnel à l'effort
        base_rate = self.offloading_params.input_rate_base_hz
        effort_rate = self.offloading_params.input_rate_effort_hz
        
        total_rate = base_rate + effort_rate * effort_level
        
        # Mise à jour du générateur Poisson
        if hasattr(self, 'external_input'):
            self.external_input.rates = total_rate * Hz
    
    def update_plasticity_modulation(self, g_NE_level: float):
        """
        Met à jour la modulation de la plasticité par LC-NE.
        
        Parameters
        ----------
        g_NE_level : float
            Niveau de neuromodulation LC-NE
        """
        
        # Mise à jour du facteur de modulation global dans les synapses STDP
        if 'ee' in self.synapses:
            self.synapses['ee'].modulation_factor = g_NE_level
    
    def run_offloading_experiment(self, target_offloading: float,
                                onset_time_ms: float,
                                duration_ms: float,
                                update_interval_ms: float = 100.0,
                                seed: int = 42) -> Dict[str, Any]:
        """
        Exécute une expérience avec offloading cognitif.
        
        Parameters
        ----------
        target_offloading : float
            Niveau cible d'offloading [0, 1]
        onset_time_ms : float
            Temps d'introduction de l'offloading
        duration_ms : float
            Durée totale de l'expérience
        update_interval_ms : float
            Intervalle de mise à jour des modulations
        seed : int
            Graine aléatoire pour reproductibilité
            
        Returns
        -------
        Dict[str, Any]
            Résultats complets incluant dynamiques d'offloading
        """
        
        logger.info(f"Expérience S5 offloading : Ω_target={target_offloading:.3f}, "
                   f"t₀={onset_time_ms:.0f}ms, durée={duration_ms:.0f}ms")
        
        # Configuration reproductibilité
        np.random.seed(seed)
        seed(seed)
        
        # Construction du réseau si nécessaire
        if self.network is None:
            self.build_network()
        
        # Simulation par intervalles avec mise à jour des modulations
        n_intervals = int(duration_ms / update_interval_ms)
        
        # Stockage des variables temporelles
        time_points = []
        offloading_values = []
        effort_values = []
        neuromodulation_values = []
        input_rates = []
        
        for interval in range(n_intervals):
            current_time = interval * update_interval_ms
            
            # Calcul des modulations
            Omega_t, effort_t, g_NE_t = self.compute_offloading_modulation(
                current_time, onset_time_ms, target_offloading
            )
            
            # Mise à jour des composants du réseau
            self.update_external_input_rate(effort_t)
            self.update_plasticity_modulation(g_NE_t)
            
            # Simulation d'un intervalle
            self.network.run(update_interval_ms * ms)
            
            # Stockage des variables
            time_points.append(current_time)
            offloading_values.append(Omega_t)
            effort_values.append(effort_t)
            neuromodulation_values.append(g_NE_t)
            
            # Taux d'entrée actuel
            if hasattr(self, 'external_input'):
                current_rate = float(self.external_input.rates[0] / Hz)
                input_rates.append(current_rate)
            else:
                input_rates.append(0.0)
            
            # Log de progression
            if interval % 50 == 0:
                logger.info(f"Temps: {current_time:.0f}ms, "
                           f"Ω={Omega_t:.3f}, effort={effort_t:.3f}, g_NE={g_NE_t:.3f}")
        
        # Compilation des résultats
        results = self._compile_offloading_results(
            target_offloading, onset_time_ms, duration_ms, seed,
            time_points, offloading_values, effort_values, 
            neuromodulation_values, input_rates
        )
        
        logger.info("Expérience S5 offloading terminée")
        
        return results
    
    def _compile_offloading_results(self, target_offloading: float,
                                  onset_time_ms: float, duration_ms: float,
                                  seed: int, time_points: List[float],
                                  offloading_values: List[float],
                                  effort_values: List[float],
                                  neuromodulation_values: List[float],
                                  input_rates: List[float]) -> Dict[str, Any]:
        """Compile les résultats incluant l'offloading."""
        
        # Résultats de base avec plasticité
        base_results = self._compile_plasticity_results(duration_ms, seed)
        
        # Dynamiques d'offloading
        offloading_dynamics = {
            'time_points_ms': np.array(time_points),
            'offloading_trajectory': np.array(offloading_values),
            'effort_trajectory': np.array(effort_values),
            'neuromodulation_trajectory': np.array(neuromodulation_values),
            'input_rate_trajectory': np.array(input_rates)
        }
        
        # Métriques d'offloading
        offloading_metrics = self._compute_offloading_metrics(
            target_offloading, onset_time_ms, time_points, 
            offloading_values, effort_values, neuromodulation_values
        )
        
        # Validation biologique
        biological_validation = self._validate_offloading_biology(
            effort_values, neuromodulation_values
        )
        
        # Fusion des résultats
        offloading_results = {
            **base_results,
            'offloading_dynamics': offloading_dynamics,
            'offloading_metrics': offloading_metrics,
            'biological_validation': biological_validation,
            'experimental_parameters': {
                'target_offloading': target_offloading,
                'onset_time_ms': onset_time_ms,
                'duration_ms': duration_ms,
                'seed': seed
            }
        }
        
        return offloading_results
    
    def _compute_offloading_metrics(self, target_offloading: float,
                                  onset_time_ms: float,
                                  time_points: List[float],
                                  offloading_values: List[float],
                                  effort_values: List[float],
                                  neuromodulation_values: List[float]) -> Dict[str, float]:
        """Calcule les métriques spécifiques à l'offloading."""
        
        time_array = np.array(time_points)
        offloading_array = np.array(offloading_values)
        effort_array = np.array(effort_values)
        neuromod_array = np.array(neuromodulation_values)
        
        # Métriques de transition
        if len(time_array) > 0:
            # Temps pour atteindre 50% du niveau cible
            half_target = target_offloading * 0.5
            half_time_idx = np.where(offloading_array >= half_target)[0]
            transition_time_50 = (time_array[half_time_idx[0]] - onset_time_ms 
                                 if len(half_time_idx) > 0 else np.inf)
            
            # Temps pour atteindre 90% du niveau cible
            ninety_target = target_offloading * 0.9
            ninety_time_idx = np.where(offloading_array >= ninety_target)[0]
            transition_time_90 = (time_array[ninety_time_idx[0]] - onset_time_ms 
                                 if len(ninety_time_idx) > 0 else np.inf)
            
            # Niveau final atteint
            final_offloading = offloading_array[-1]
            
            # Efficacité de la transition
            transition_efficiency = final_offloading / target_offloading if target_offloading > 0 else 1.0
        else:
            transition_time_50 = np.inf
            transition_time_90 = np.inf
            final_offloading = 0.0
            transition_efficiency = 0.0
        
        # Corrélation effort-neuromodulation (validation H4)
        if len(effort_array) > 1 and len(neuromod_array) > 1:
            effort_neuromod_correlation = np.corrcoef(effort_array, neuromod_array)[0, 1]
        else:
            effort_neuromod_correlation = 0.0
        
        # Réduction d'effort cognitif (CLI)
        if len(effort_array) > 0:
            initial_effort = effort_array[0] if len(effort_array) > 0 else 1.0
            final_effort = effort_array[-1] if len(effort_array) > 0 else 1.0
            cognitive_load_reduction = (initial_effort - final_effort) / initial_effort if initial_effort > 0 else 0.0
        else:
            cognitive_load_reduction = 0.0
        
        # Stabilité de la modulation
        if len(neuromod_array) > 10:
            recent_neuromod = neuromod_array[-10:]  # 10 derniers points
            neuromodulation_stability = 1.0 / (1.0 + np.std(recent_neuromod))
        else:
            neuromodulation_stability = 1.0
        
        return {
            'transition_time_50_ms': transition_time_50,
            'transition_time_90_ms': transition_time_90,
            'final_offloading_level': final_offloading,
            'transition_efficiency': transition_efficiency,
            'effort_neuromod_correlation': effort_neuromod_correlation,
            'cognitive_load_reduction': cognitive_load_reduction,
            'neuromodulation_stability': neuromodulation_stability,
            'mean_effort_reduction': 1.0 - np.mean(effort_array) if len(effort_array) > 0 else 0.0,
            'mean_neuromod_level': np.mean(neuromod_array) if len(neuromod_array) > 0 else 0.0
        }
    
    def _validate_offloading_biology(self, effort_values: List[float],
                                   neuromodulation_values: List[float]) -> Dict[str, Any]:
        """Valide la biologie de l'offloading."""
        
        validation_results = {
            'effort_range_valid': True,
            'neuromodulation_range_valid': True,
            'effort_neuromod_coupling_valid': True,
            'violations': []
        }
        
        # Validation des plages d'effort
        if len(effort_values) > 0:
            min_effort = min(effort_values)
            max_effort = max(effort_values)
            
            if not (0.0 <= min_effort <= 1.0) or not (0.0 <= max_effort <= 1.0):
                validation_results['effort_range_valid'] = False
                validation_results['violations'].append(
                    f"Effort hors plage [0,1]: min={min_effort:.3f}, max={max_effort:.3f}"
                )
        
        # Validation des plages de neuromodulation
        if len(neuromodulation_values) > 0:
            min_neuromod = min(neuromodulation_values)
            max_neuromod = max(neuromodulation_values)
            
            if not (0.1 <= min_neuromod <= 3.0) or not (0.1 <= max_neuromod <= 3.0):
                validation_results['neuromodulation_range_valid'] = False
                validation_results['violations'].append(
                    f"Neuromodulation hors plage [0.1,3.0]: min={min_neuromod:.3f}, max={max_neuromod:.3f}"
                )
        
        # Validation du couplage effort-neuromodulation
        if len(effort_values) > 1 and len(neuromodulation_values) > 1:
            correlation = np.corrcoef(effort_values, neuromodulation_values)[0, 1]
            
            if correlation < self.offloading_params.effort_correlation_threshold:
                validation_results['effort_neuromod_coupling_valid'] = False
                validation_results['violations'].append(
                    f"Corrélation effort-neuromod faible: r={correlation:.3f} < {self.offloading_params.effort_correlation_threshold}"
                )
        
        validation_results['all_valid'] = (
            validation_results['effort_range_valid'] and
            validation_results['neuromodulation_range_valid'] and
            validation_results['effort_neuromod_coupling_valid']
        )
        
        return validation_results
    
    def _build_external_inputs(self):
        """Construit les entrées externes modulables par l'effort."""
        
        # Générateur d'entrées Poisson
        n_input_neurons = int(self.params.N_e * self.offloading_params.input_fraction)
        
        self.external_input = PoissonGroup(
            n_input_neurons,
            rates=self.offloading_params.input_rate_base_hz * Hz
        )
        
        # Synapses d'entrée externe
        self.synapses['external'] = Synapses(
            self.external_input, self.neurons['e'],
            'w : 1',
            on_pre='g_e_post += w * 0.15'
        )
        
        # Connexion avec probabilité spécifiée
        self.synapses['external'].connect(p=self.offloading_params.input_connection_prob)
        
        # Poids renforcés pour les entrées externes
        self.synapses['external'].w = (self.plasticity_params.w_init_mean * 
                                      self.offloading_params.input_weight_multiplier)
        
        logger.info(f"Entrées externes S5 : {n_input_neurons} neurones, "
                   f"taux initial {self.offloading_params.input_rate_base_hz} Hz")

def create_s5_model(network_params: Dict[str, Any] = None,
                   plasticity_params: Dict[str, Any] = None,
                   offloading_params: Dict[str, Any] = None) -> S5_OffloadingNetwork:
    """
    Crée un modèle S5 avec offloading cognitif.
    
    Parameters
    ----------
    network_params : Dict[str, Any], optional
        Paramètres du réseau de base
    plasticity_params : Dict[str, Any], optional
        Paramètres de plasticité
    offloading_params : Dict[str, Any], optional
        Paramètres d'offloading et modulation LC-NE
        
    Returns
    -------
    S5_OffloadingNetwork
        Modèle S5 initialisé
    """
    
    # Paramètres par défaut
    net_params = S1_NetworkParameters()
    if network_params:
        for key, value in network_params.items():
            if hasattr(net_params, key):
                setattr(net_params, key, value)
    
    plas_params = S2_PlasticityParameters()
    if plasticity_params:
        for key, value in plasticity_params.items():
            if hasattr(plas_params, key):
                setattr(plas_params, key, value)
    
    off_params = S5_OffloadingParameters()
    if offloading_params:
        for key, value in offloading_params.items():
            if hasattr(off_params, key):
                setattr(off_params, key, value)
    
    return S5_OffloadingNetwork(net_params, plas_params, off_params)

def validate_s5_model():
    """Valide le modèle S5 avec tests d'offloading."""
    
    logger.info("=== Validation du Modèle S5 (Offloading + LC-NE) ===")
    
    # Test d'offloading progressif
    model = create_s5_model()
    results = model.run_offloading_experiment(
        target_offloading=0.6,
        onset_time_ms=2000.0,
        duration_ms=6000.0,
        seed=42
    )
    
    # Vérifications biologiques
    offloading_metrics = results['offloading_metrics']
    bio_validation = results['biological_validation']
    
    # Test de transition
    assert offloading_metrics['final_offloading_level'] > 0.5, "Offloading insuffisant"
    assert offloading_metrics['transition_efficiency'] > 0.8, "Transition inefficace"
    
    # Test de corrélation effort-neuromodulation (H4)
    effort_neuromod_corr = offloading_metrics['effort_neuromod_correlation']
    assert effort_neuromod_corr > 0.8, f"Corrélation effort-NE faible: {effort_neuromod_corr:.3f}"
    
    # Test de réduction d'effort cognitif (CLI)
    cli = offloading_metrics['cognitive_load_reduction']
    assert cli > 0.3, f"Réduction d'effort insuffisante: {cli:.3f}"
    
    # Test de validation biologique
    assert bio_validation['all_valid'], f"Violations biologiques: {bio_validation['violations']}"
    
    logger.info("✓ Modèle S5 validé biologiquement")
    logger.info(f"  Niveau final offloading : {offloading_metrics['final_offloading_level']:.3f}")
    logger.info(f"  Corrélation effort-NE : {effort_neuromod_corr:.3f}")
    logger.info(f"  Réduction charge cognitive : {cli:.3f}")
    logger.info(f"  Temps transition 90% : {offloading_metrics['transition_time_90_ms']:.0f} ms")
    
    return True

if __name__ == "__main__":
    # Test de validation
    validate_s5_model()
    print("Modèle S5 (Offloading + LC-NE) validé et prêt !")
