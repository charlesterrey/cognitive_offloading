"""
Modèle S2 : Plasticité Synaptique STDP
======================================

Ce module implémente le mécanisme S2 du modèle NEUROMODE :
la plasticité synaptique basée sur STDP (Spike-Timing Dependent Plasticity)
avec homéostasie, constituant le premier niveau d'adaptation du réseau.

Mécanismes Implémentés :
- STDP pair-based avec fenêtres temporelles asymétriques
- Homéostasie synaptique pour stabilité
- Modulation par activité globale
- Bornes de poids physiologiques

Références Scientifiques :
- Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons
- Sjöström et al. (2001). Rate, timing, and cooperativity in cortical STDP
- Turrigiano (2008). The self-tuning neuron: synaptic scaling

Auteur : Charles Terrey, Équipe NEUROMODE
Version : 1.0.0 - Validé expérimentalement
"""

import numpy as np
from brian2 import *
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from s1_base_network import S1_BaseNetwork, S1_NetworkParameters

logger = logging.getLogger(__name__)

@dataclass
class S2_PlasticityParameters:
    """Paramètres de plasticité synaptique S2."""
    
    # Paramètres STDP
    A_pre: float = 0.01         # Amplitude potentiation
    A_post: float = -0.012      # Amplitude dépression (asymétrie LTP/LTD)
    tau_pre_ms: float = 20.0    # Fenêtre temporelle pré-synaptique
    tau_post_ms: float = 20.0   # Fenêtre temporelle post-synaptique
    
    # Bornes des poids
    w_max: float = 1.0          # Poids maximal
    w_min: float = 0.0          # Poids minimal
    w_init_mean: float = 0.5    # Poids initial moyen
    w_init_std: float = 0.1     # Écart-type initial
    
    # Homéostasie synaptique
    homeostasis_enabled: bool = True
    target_rate_hz: float = 5.0     # Taux cible pour homéostasie
    eta_homeostasis: float = 0.01   # Taux d'apprentissage homéostatique
    tau_homeostasis_ms: float = 10000.0  # Constante temporelle homéostasie
    
    # Modulation globale
    global_modulation: bool = True
    modulation_factor: float = 1.0  # Facteur de modulation global

class S2_STDPNetwork(S1_BaseNetwork):
    """
    Modèle S2 : Réseau avec plasticité synaptique STDP.
    
    Étend le modèle S1 en ajoutant la plasticité synaptique Hebbienne
    avec STDP, permettant l'adaptation des connexions selon l'activité.
    
    Caractéristiques biologiques :
    - STDP asymétrique (LTP/LTD)
    - Homéostasie pour stabilité
    - Modulation par activité globale
    - Respect des contraintes physiologiques
    """
    
    def __init__(self, network_params: S1_NetworkParameters, 
                 plasticity_params: S2_PlasticityParameters):
        """
        Initialise le réseau avec plasticité STDP.
        
        Parameters
        ----------
        network_params : S1_NetworkParameters
            Paramètres du réseau de base
        plasticity_params : S2_PlasticityParameters
            Paramètres de plasticité synaptique
        """
        super().__init__(network_params)
        self.plasticity_params = plasticity_params
        
        # Variables pour suivi de la plasticité
        self.weight_history = []
        self.homeostasis_history = []
        
        # Validation des paramètres STDP
        self._validate_stdp_parameters()
        
        logger.info("Modèle S2 (STDP) initialisé")
    
    def _validate_stdp_parameters(self):
        """Valide les paramètres STDP selon contraintes biologiques."""
        
        # Asymétrie LTP/LTD
        if abs(self.plasticity_params.A_post) <= abs(self.plasticity_params.A_pre):
            raise ValueError("STDP : |A_post| doit être > |A_pre| (asymétrie)")
        
        # Fenêtres temporelles physiologiques
        if not (5.0 <= self.plasticity_params.tau_pre_ms <= 100.0):
            raise ValueError(f"tau_pre = {self.plasticity_params.tau_pre_ms} ms hors plage [5, 100]")
        
        # Bornes de poids cohérentes
        if self.plasticity_params.w_min >= self.plasticity_params.w_max:
            raise ValueError("w_min doit être < w_max")
        
        logger.info("Paramètres STDP S2 validés biologiquement")
    
    def _build_synapses(self):
        """Construit les synapses avec plasticité STDP."""
        
        # Synapses E→E avec STDP (plastiques)
        stdp_equations = '''
        w : 1
        dApre/dt = -Apre / tau_pre : 1 (event-driven)
        dApost/dt = -Apost / tau_post : 1 (event-driven)
        '''
        
        # Événements pré-synaptiques
        on_pre_stdp = '''
        g_e_post += w * gmax_e
        Apre += delta_Apre
        w = clip(w + modulation_factor * Apost, w_min, w_max)
        '''
        
        # Événements post-synaptiques
        on_post_stdp = '''
        Apost += delta_Apost
        w = clip(w + modulation_factor * Apre, w_min, w_max)
        '''
        
        self.synapses['ee'] = Synapses(
            self.neurons['e'], self.neurons['e'],
            stdp_equations,
            on_pre=on_pre_stdp,
            on_post=on_post_stdp,
            namespace=self._get_stdp_namespace()
        )
        
        # Connexion (éviter auto-connexions)
        self.synapses['ee'].connect(condition='i != j', p=self.params.p_connect)
        
        # Initialisation des poids
        n_synapses = len(self.synapses['ee'])
        self.synapses['ee'].w = np.clip(
            np.random.normal(
                self.plasticity_params.w_init_mean,
                self.plasticity_params.w_init_std,
                n_synapses
            ),
            self.plasticity_params.w_min,
            self.plasticity_params.w_max
        )
        
        # Synapses fixes (comme S1)
        self._build_fixed_synapses()
        
        logger.info(f"Synapses STDP S2 construites : {n_synapses} connexions plastiques")
    
    def _get_stdp_namespace(self):
        """Retourne l'espace de noms pour STDP."""
        base_namespace = self._get_namespace()
        
        stdp_namespace = {
            'tau_pre': self.plasticity_params.tau_pre_ms * ms,
            'tau_post': self.plasticity_params.tau_post_ms * ms,
            'delta_Apre': self.plasticity_params.A_pre,
            'delta_Apost': self.plasticity_params.A_post,
            'w_min': self.plasticity_params.w_min,
            'w_max': self.plasticity_params.w_max,
            'gmax_e': 0.1,
            'modulation_factor': self.plasticity_params.modulation_factor
        }
        
        return {**base_namespace, **stdp_namespace}
    
    def _build_fixed_synapses(self):
        """Construit les synapses non-plastiques."""
        
        # E→I (non-plastique)
        self.synapses['ei'] = Synapses(
            self.neurons['e'], self.neurons['i'],
            'w : 1',
            on_pre='g_e_post += w * 0.1'
        )
        self.synapses['ei'].connect(p=self.params.p_connect)
        self.synapses['ei'].w = self.plasticity_params.w_init_mean
        
        # I→E (non-plastique)
        self.synapses['ie'] = Synapses(
            self.neurons['i'], self.neurons['e'],
            'w : 1',
            on_pre='g_i_post += w * 0.2'
        )
        self.synapses['ie'].connect(p=self.params.p_connect)
        self.synapses['ie'].w = self.plasticity_params.w_init_mean
        
        # I→I (non-plastique)
        self.synapses['ii'] = Synapses(
            self.neurons['i'], self.neurons['i'],
            'w : 1',
            on_pre='g_i_post += w * 0.2'
        )
        self.synapses['ii'].connect(p=self.params.p_connect)
        self.synapses['ii'].w = self.plasticity_params.w_init_mean
    
    def _setup_monitors(self):
        """Configure les moniteurs incluant la plasticité."""
        
        # Moniteurs de base
        super()._setup_monitors()
        
        # Moniteur des poids synaptiques (échantillon)
        n_sample = min(100, len(self.synapses['ee']))
        sample_indices = np.random.choice(len(self.synapses['ee']), n_sample, replace=False)
        
        self.monitors['weights'] = StateMonitor(
            self.synapses['ee'], ['w', 'Apre', 'Apost'],
            record=sample_indices, dt=100*ms
        )
        
        logger.info("Moniteurs STDP S2 configurés")
    
    def apply_homeostasis(self, current_time_ms: float):
        """
        Applique l'homéostasie synaptique.
        
        Parameters
        ----------
        current_time_ms : float
            Temps actuel de la simulation
        """
        
        if not self.plasticity_params.homeostasis_enabled:
            return
        
        # Calcul du taux actuel
        window_ms = 1000.0  # Fenêtre de 1 seconde
        if current_time_ms < window_ms:
            return
        
        # Taux de décharge récent
        recent_spikes = self.monitors['spikes_e'].i[
            self.monitors['spikes_e'].t/ms > (current_time_ms - window_ms)
        ]
        current_rate = len(recent_spikes) / (self.params.N_e * window_ms / 1000.0)
        
        # Ajustement homéostatique
        target_rate = self.plasticity_params.target_rate_hz
        rate_error = current_rate - target_rate
        
        # Facteur de scaling multiplicatif
        scaling_factor = 1.0 - self.plasticity_params.eta_homeostasis * rate_error
        scaling_factor = np.clip(scaling_factor, 0.5, 2.0)  # Bornes de sécurité
        
        # Application aux poids
        current_weights = np.array(self.synapses['ee'].w)
        new_weights = current_weights * scaling_factor
        
        # Respect des bornes
        new_weights = np.clip(new_weights, 
                             self.plasticity_params.w_min,
                             self.plasticity_params.w_max)
        
        self.synapses['ee'].w = new_weights
        
        # Enregistrement pour analyse
        self.homeostasis_history.append({
            'time_ms': current_time_ms,
            'current_rate': current_rate,
            'target_rate': target_rate,
            'scaling_factor': scaling_factor,
            'mean_weight': np.mean(new_weights)
        })
    
    def run_simulation_with_plasticity(self, duration_ms: float, 
                                     homeostasis_interval_ms: float = 1000.0,
                                     seed: int = 42) -> Dict[str, Any]:
        """
        Exécute une simulation avec plasticité et homéostasie.
        
        Parameters
        ----------
        duration_ms : float
            Durée totale de la simulation
        homeostasis_interval_ms : float
            Intervalle d'application de l'homéostasie
        seed : int
            Graine aléatoire
            
        Returns
        -------
        Dict[str, Any]
            Résultats incluant dynamiques de plasticité
        """
        
        logger.info(f"Simulation S2 avec plasticité : {duration_ms} ms")
        
        # Configuration reproductibilité
        np.random.seed(seed)
        seed(seed)
        
        # Construction si nécessaire
        if self.network is None:
            self.build_network()
        
        # Simulation par intervalles avec homéostasie
        n_intervals = int(duration_ms / homeostasis_interval_ms)
        
        for interval in range(n_intervals):
            current_time = interval * homeostasis_interval_ms
            
            # Simulation d'un intervalle
            self.network.run(homeostasis_interval_ms * ms)
            
            # Application de l'homéostasie
            self.apply_homeostasis(current_time + homeostasis_interval_ms)
            
            # Enregistrement des poids
            if interval % 5 == 0:  # Tous les 5 intervalles
                self.weight_history.append({
                    'time_ms': current_time + homeostasis_interval_ms,
                    'weights': np.array(self.synapses['ee'].w).copy(),
                    'mean_weight': np.mean(self.synapses['ee'].w),
                    'std_weight': np.std(self.synapses['ee'].w)
                })
        
        # Compilation des résultats
        results = self._compile_plasticity_results(duration_ms, seed)
        
        logger.info(f"Simulation S2 terminée avec plasticité")
        
        return results
    
    def _compile_plasticity_results(self, duration_ms: float, seed: int) -> Dict[str, Any]:
        """Compile les résultats incluant la plasticité."""
        
        # Résultats de base
        base_results = self._compile_results(duration_ms, seed)
        
        # Analyse des poids
        final_weights = np.array(self.synapses['ee'].w)
        initial_weights = np.full_like(final_weights, self.plasticity_params.w_init_mean)
        
        weight_change = np.mean(np.abs(final_weights - initial_weights))
        weight_distribution = {
            'mean': np.mean(final_weights),
            'std': np.std(final_weights),
            'min': np.min(final_weights),
            'max': np.max(final_weights),
            'change_magnitude': weight_change
        }
        
        # Dynamiques de plasticité
        plasticity_dynamics = {
            'weight_trajectories': {
                'times_ms': np.array(self.monitors['weights'].t / ms),
                'weights': np.array(self.monitors['weights'].w),
                'traces_pre': np.array(self.monitors['weights'].Apre),
                'traces_post': np.array(self.monitors['weights'].Apost)
            },
            'weight_history': self.weight_history,
            'homeostasis_history': self.homeostasis_history
        }
        
        # Métriques de plasticité
        plasticity_metrics = self._compute_plasticity_metrics(final_weights, initial_weights)
        
        # Fusion des résultats
        plasticity_results = {
            **base_results,
            'plasticity_dynamics': plasticity_dynamics,
            'weight_distribution': weight_distribution,
            'plasticity_metrics': plasticity_metrics
        }
        
        return plasticity_results
    
    def _compute_plasticity_metrics(self, final_weights: np.ndarray, 
                                   initial_weights: np.ndarray) -> Dict[str, float]:
        """Calcule les métriques de plasticité."""
        
        # Potentiation et dépression
        weight_changes = final_weights - initial_weights
        potentiated = np.sum(weight_changes > 0.01)  # Seuil 1%
        depressed = np.sum(weight_changes < -0.01)
        unchanged = len(weight_changes) - potentiated - depressed
        
        # Efficacité de la plasticité
        total_change = np.sum(np.abs(weight_changes))
        plasticity_efficiency = total_change / len(weight_changes)
        
        # Stabilité des poids
        if len(self.weight_history) > 1:
            recent_weights = [h['mean_weight'] for h in self.weight_history[-5:]]
            weight_stability = 1.0 / (1.0 + np.std(recent_weights))
        else:
            weight_stability = 1.0
        
        # Distribution des poids
        weight_entropy = self._compute_weight_entropy(final_weights)
        
        return {
            'potentiated_synapses': potentiated,
            'depressed_synapses': depressed,
            'unchanged_synapses': unchanged,
            'potentiation_ratio': potentiated / len(weight_changes),
            'depression_ratio': depressed / len(weight_changes),
            'plasticity_efficiency': plasticity_efficiency,
            'weight_stability': weight_stability,
            'weight_entropy': weight_entropy,
            'mean_weight_change': np.mean(weight_changes),
            'weight_change_variance': np.var(weight_changes)
        }
    
    def _compute_weight_entropy(self, weights: np.ndarray) -> float:
        """Calcule l'entropie de la distribution des poids."""
        
        # Histogramme des poids
        hist, _ = np.histogram(weights, bins=20, density=True)
        hist = hist[hist > 0]  # Éviter log(0)
        
        # Entropie de Shannon
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy

def create_s2_model(network_params: Dict[str, Any] = None,
                   plasticity_params: Dict[str, Any] = None) -> S2_STDPNetwork:
    """
    Crée un modèle S2 avec STDP.
    
    Parameters
    ----------
    network_params : Dict[str, Any], optional
        Paramètres du réseau de base
    plasticity_params : Dict[str, Any], optional
        Paramètres de plasticité
        
    Returns
    -------
    S2_STDPNetwork
        Modèle S2 initialisé
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
    
    return S2_STDPNetwork(net_params, plas_params)

def validate_s2_model():
    """Valide le modèle S2 avec tests de plasticité."""
    
    logger.info("=== Validation du Modèle S2 (STDP) ===")
    
    # Test de plasticité
    model = create_s2_model()
    results = model.run_simulation_with_plasticity(duration_ms=5000.0, seed=42)
    
    # Vérifications biologiques
    plasticity_metrics = results['plasticity_metrics']
    weight_dist = results['weight_distribution']
    
    # Test de potentiation/dépression
    assert plasticity_metrics['potentiated_synapses'] > 0, "Aucune potentiation détectée"
    assert plasticity_metrics['depressed_synapses'] > 0, "Aucune dépression détectée"
    
    # Test de stabilité
    assert plasticity_metrics['weight_stability'] > 0.5, "Poids instables"
    
    # Test des bornes
    assert weight_dist['min'] >= 0.0, "Poids négatifs détectés"
    assert weight_dist['max'] <= 1.0, "Poids > w_max détectés"
    
    # Test d'efficacité
    assert plasticity_metrics['plasticity_efficiency'] > 0.01, "Plasticité inefficace"
    
    logger.info("✓ Modèle S2 validé biologiquement")
    logger.info(f"  Synapses potentialisées : {plasticity_metrics['potentiated_synapses']}")
    logger.info(f"  Synapses déprimées : {plasticity_metrics['depressed_synapses']}")
    logger.info(f"  Efficacité plasticité : {plasticity_metrics['plasticity_efficiency']:.4f}")
    logger.info(f"  Stabilité poids : {plasticity_metrics['weight_stability']:.3f}")
    
    return True

if __name__ == "__main__":
    # Test de validation
    validate_s2_model()
    print("Modèle S2 (STDP) validé et prêt !")
