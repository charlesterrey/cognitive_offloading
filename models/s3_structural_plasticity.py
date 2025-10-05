"""
Modèle S3 : Plasticité Structurelle Développementale
===================================================

Ce module implémente le mécanisme S3 du modèle NEUROMODE :
la plasticité structurelle avec phases de croissance (GROW) et d'élagage (PRUNE)
activité-dépendant, simulant le développement neuronal réaliste.

Mécanismes Implémentés :
- Phase GROW : Surcroissance synaptique contrôlée (0 → T_grow)
- Phase PRUNE : Élagage activité-dépendant (T_grow → T_total)
- Score d'activité A : Accumulation locale de l'activité synaptique
- Coûts de câblage : Pénalisation des connexions longues
- Contraintes énergétiques : Budget métabolique réaliste

Références Scientifiques :
- Chechik et al. (1998). Synaptic pruning in development: a computational account
- Huttenlocher & Dabholkar (1997). Regional differences in synaptogenesis
- Innocenti & Price (2005). Exuberance in the development of cortical networks

Auteur : Charles Terrey
Version : 1.0.0 - Validé développementalement
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
class S3_StructuralParameters:
    """Paramètres de plasticité structurelle développementale S3."""
    
    # Phases développementales
    T_grow_ms: float = 3000.0       # Durée phase GROW
    T_prune_ms: float = 5000.0      # Durée phase PRUNE
    T_total_ms: float = 8000.0      # Durée totale
    
    # Cibles de densité
    rho_initial: float = 0.05       # Densité initiale
    rho_target_grow: float = 0.25   # Densité cible en phase GROW
    rho_final_target: float = 0.15  # Densité finale cible
    
    # Règle d'élagage
    theta_act: float = 0.15         # Seuil d'activité pour élagage
    k1: float = 8.0                 # Poids terme activité
    k2: float = 3.0                 # Poids coût câblage
    
    # Score d'activité
    beta_pre: float = 0.02          # Incrément pré-synaptique
    beta_post: float = 0.02         # Incrément post-synaptique
    tau_A_ms: float = 1000.0        # Constante temporelle décroissance
    
    # Contraintes énergétiques
    energy_budget: float = 5000.0   # Budget énergétique
    c_spike: float = 1.0            # Coût par spike
    c_syn: float = 0.1              # Coût par événement synaptique
    c_length: float = 0.01          # Coût par unité de longueur
    
    # Paramètres de croissance
    p_grow_base: float = 0.001      # Probabilité de base de croissance
    growth_saturation: float = 0.9  # Facteur de saturation

class S3_StructuralPlasticityNetwork(S2_STDPNetwork):
    """
    Modèle S3 : Réseau avec plasticité structurelle développementale.
    
    Étend le modèle S2 en ajoutant les mécanismes de croissance et d'élagage
    synaptiques développementaux, simulant les phases critiques du
    développement cortical.
    
    Caractéristiques biologiques :
    - Phases GROW/PRUNE distinctes
    - Élagage activité-dépendant
    - Contraintes spatiales et énergétiques
    - Score d'activité local
    - Validation développementale
    """
    
    def __init__(self, network_params: S1_NetworkParameters,
                 plasticity_params: S2_PlasticityParameters,
                 structural_params: S3_StructuralParameters):
        """
        Initialise le réseau avec plasticité structurelle.
        
        Parameters
        ----------
        network_params : S1_NetworkParameters
            Paramètres du réseau de base
        plasticity_params : S2_PlasticityParameters
            Paramètres de plasticité synaptique
        structural_params : S3_StructuralParameters
            Paramètres de plasticité structurelle
        """
        super().__init__(network_params, plasticity_params)
        self.structural_params = structural_params
        
        # Variables pour suivi développemental
        self.developmental_history = []
        self.growth_events = []
        self.pruning_events = []
        self.energy_history = []
        
        # État développemental
        self.current_phase = 'GROW'
        self.phase_start_time = 0.0
        
        # Validation des paramètres structurels
        self._validate_structural_parameters()
        
        logger.info("Modèle S3 (Plasticité structurelle) initialisé")
    
    def _validate_structural_parameters(self):
        """Valide les paramètres structurels selon contraintes développementales."""
        
        # Cohérence temporelle des phases
        if self.structural_params.T_grow_ms >= self.structural_params.T_total_ms:
            raise ValueError("T_grow doit être < T_total")
        
        if self.structural_params.T_prune_ms <= 0:
            raise ValueError("T_prune doit être > 0")
        
        # Cohérence des densités cibles
        if not (self.structural_params.rho_initial < 
                self.structural_params.rho_target_grow):
            raise ValueError("rho_initial doit être < rho_target_grow")
        
        # Paramètres d'élagage réalistes
        if not (0.05 <= self.structural_params.theta_act <= 0.5):
            raise ValueError(f"theta_act = {self.structural_params.theta_act} hors plage [0.05, 0.5]")
        
        # Budget énergétique positif
        if self.structural_params.energy_budget <= 0:
            raise ValueError("Budget énergétique doit être > 0")
        
        logger.info("Paramètres structurels S3 validés développementalement")
    
    def _build_synapses(self):
        """Construit les synapses avec mécanismes structurels."""
        
        # Synapses E→E avec plasticité structurelle complète
        structural_equations = '''
        w : 1
        A : 1                                    # Score d'activité
        alive : 1                                # État binaire (0/1)
        len_cost : 1                             # Coût de câblage spatial
        age : second                             # Âge de la synapse
        last_active : second                     # Dernier temps d'activité
        dApre/dt = -Apre / tau_pre : 1 (event-driven)
        dApost/dt = -Apost / tau_post : 1 (event-driven)
        dA/dt = -A / tau_A : 1 (clock-driven)   # Décroissance du score
        '''
        
        # Événements pré-synaptiques avec mise à jour activité
        on_pre_structural = '''
        g_e_post += w * alive * gmax_e
        Apre += delta_Apre
        A += beta_pre
        last_active = t
        w = clip(w + gamma_t * g_NE_t * Apost, w_min, w_max)
        '''
        
        # Événements post-synaptiques
        on_post_structural = '''
        Apost += delta_Apost
        A += beta_post
        w = clip(w + gamma_t * g_NE_t * Apre, w_min, w_max)
        '''
        
        self.synapses['ee'] = Synapses(
            self.neurons['e'], self.neurons['e'],
            structural_equations,
            on_pre=on_pre_structural,
            on_post=on_post_structural,
            namespace=self._get_structural_namespace()
        )
        
        # Connexion complète (toutes les paires possibles)
        self.synapses['ee'].connect(condition='i != j')
        
        # Initialisation structurelle
        self._initialize_structural_synapses()
        
        # Synapses fixes (héritées de S2)
        self._build_fixed_synapses()
        
        logger.info(f"Synapses structurelles S3 : {len(self.synapses['ee'])} connexions")
    
    def _get_structural_namespace(self):
        """Retourne l'espace de noms pour plasticité structurelle."""
        base_namespace = self._get_stdp_namespace()
        
        structural_namespace = {
            'tau_A': self.structural_params.tau_A_ms * ms,
            'beta_pre': self.structural_params.beta_pre,
            'beta_post': self.structural_params.beta_post,
            'gamma_t': 1.0,  # Sera mis à jour dynamiquement
            'g_NE_t': 1.0    # Sera mis à jour dynamiquement
        }
        
        return {**base_namespace, **structural_namespace}
    
    def _initialize_structural_synapses(self):
        """Initialise les synapses avec paramètres structurels."""
        
        n_synapses = len(self.synapses['ee'])
        
        # Poids initiaux
        self.synapses['ee'].w = np.clip(
            np.random.normal(self.plasticity_params.w_init_mean,
                           self.plasticity_params.w_init_std, n_synapses),
            self.plasticity_params.w_min,
            self.plasticity_params.w_max
        )
        
        # Score d'activité initial
        self.synapses['ee'].A = np.zeros(n_synapses)
        
        # État initial : densité faible
        target_active = int(self.structural_params.rho_initial * n_synapses)
        alive_states = np.zeros(n_synapses)
        active_indices = np.random.choice(n_synapses, target_active, replace=False)
        alive_states[active_indices] = 1
        self.synapses['ee'].alive = alive_states
        
        # Âge des synapses
        self.synapses['ee'].age = 0 * ms
        self.synapses['ee'].last_active = -1000 * ms  # Jamais actives initialement
        
        # Calcul des coûts de câblage
        self._compute_wiring_costs()
        
        logger.info(f"Synapses initialisées : {target_active}/{n_synapses} actives "
                   f"(densité {self.structural_params.rho_initial:.3f})")
    
    def _compute_wiring_costs(self):
        """Calcule les coûts de câblage spatial."""
        
        # Positions spatiales en grille 2D
        grid_size = int(np.ceil(np.sqrt(self.params.N_e)))
        positions = np.array([[i % grid_size, i // grid_size] 
                             for i in range(self.params.N_e)])
        
        # Distances euclidiennes
        pre_indices = self.synapses['ee'].i
        post_indices = self.synapses['ee'].j
        
        distances = np.sqrt(np.sum((positions[pre_indices] - positions[post_indices])**2, axis=1))
        max_distance = np.sqrt(2) * grid_size
        
        # Normalisation et pondération
        self.synapses['ee'].len_cost = distances / max_distance
    
    def determine_developmental_phase(self, time_ms: float) -> str:
        """Détermine la phase développementale actuelle."""
        
        if time_ms < self.structural_params.T_grow_ms:
            return 'GROW'
        elif time_ms < self.structural_params.T_total_ms:
            return 'PRUNE'
        else:
            return 'MATURE'
    
    def apply_growth_phase(self, time_ms: float):
        """Applique les mécanismes de la phase GROW."""
        
        current_density = np.mean(self.synapses['ee'].alive)
        target_density = self.structural_params.rho_target_grow
        
        if current_density < target_density:
            # Probabilité de croissance proportionnelle au déficit
            density_deficit = target_density - current_density
            p_grow = self.structural_params.p_grow_base * density_deficit / target_density
            
            # Facteur de saturation temporelle
            time_factor = min(1.0, time_ms / self.structural_params.T_grow_ms)
            p_grow *= time_factor
            
            # Sélection des synapses inactives
            inactive_mask = self.synapses['ee'].alive == 0
            inactive_indices = np.where(inactive_mask)[0]
            
            if len(inactive_indices) > 0:
                # Nombre de synapses à activer
                n_to_grow = int(p_grow * len(inactive_indices))
                
                if n_to_grow > 0:
                    # Sélection préférentielle des connexions courtes
                    len_costs = np.array(self.synapses['ee'].len_cost)[inactive_indices]
                    growth_probabilities = 1.0 / (1.0 + len_costs * 2.0)  # Préférence courtes
                    growth_probabilities /= np.sum(growth_probabilities)
                    
                    # Sélection pondérée
                    selected_indices = np.random.choice(
                        inactive_indices, 
                        min(n_to_grow, len(inactive_indices)),
                        replace=False,
                        p=growth_probabilities
                    )
                    
                    # Activation des synapses
                    self.synapses['ee'].alive[selected_indices] = 1
                    self.synapses['ee'].age[selected_indices] = time_ms * ms
                    
                    # Enregistrement de l'événement
                    self.growth_events.append({
                        'time_ms': time_ms,
                        'n_grown': len(selected_indices),
                        'new_density': np.mean(self.synapses['ee'].alive),
                        'phase': 'GROW'
                    })
    
    def apply_pruning_phase(self, time_ms: float):
        """Applique les mécanismes de la phase PRUNE."""
        
        # Calcul de la pression énergétique
        energy_pressure = self._compute_energy_pressure(time_ms)
        
        # Sélection des synapses actives
        active_mask = self.synapses['ee'].alive == 1
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) > 0:
            # Scores d'activité et coûts
            A_values = np.array(self.synapses['ee'].A)[active_indices]
            len_costs = np.array(self.synapses['ee'].len_cost)[active_indices]
            
            # Règle d'élagage avec pression énergétique
            activity_term = self.structural_params.k1 * (self.structural_params.theta_act - A_values)
            wiring_term = self.structural_params.k2 * len_costs
            energy_term = energy_pressure * 2.0  # Amplification par pression
            
            # Probabilité d'élagage (sigmoïde)
            pruning_logits = activity_term + wiring_term + energy_term
            p_prune = 1.0 / (1.0 + np.exp(-pruning_logits))
            
            # Application stochastique
            prune_mask = np.random.random(len(active_indices)) < p_prune * 0.01  # Temporisation
            prune_indices = active_indices[prune_mask]
            
            if len(prune_indices) > 0:
                # Élagage des synapses
                self.synapses['ee'].alive[prune_indices] = 0
                self.synapses['ee'].w[prune_indices] = 0
                self.synapses['ee'].A[prune_indices] = 0
                
                # Enregistrement de l'événement
                self.pruning_events.append({
                    'time_ms': time_ms,
                    'n_pruned': len(prune_indices),
                    'new_density': np.mean(self.synapses['ee'].alive),
                    'energy_pressure': energy_pressure,
                    'phase': 'PRUNE'
                })
    
    def _compute_energy_pressure(self, time_ms: float) -> float:
        """Calcule la pression énergétique actuelle."""
        
        # Fenêtre temporelle pour calcul énergétique
        window_ms = 200.0
        
        if time_ms < window_ms:
            return 0.0
        
        # Coût des spikes récents
        recent_spikes_e = len([t for t in self.monitors['spikes_e'].t/ms 
                              if time_ms - window_ms <= t <= time_ms])
        recent_spikes_i = len([t for t in self.monitors['spikes_i'].t/ms 
                              if time_ms - window_ms <= t <= time_ms])
        
        spike_cost = (recent_spikes_e + recent_spikes_i) * self.structural_params.c_spike
        
        # Coût des événements synaptiques (approximation)
        active_synapses = np.sum(self.synapses['ee'].alive)
        syn_events = recent_spikes_e * active_synapses * 0.1  # Approximation
        syn_cost = syn_events * self.structural_params.c_syn
        
        # Coût de câblage
        len_costs = np.array(self.synapses['ee'].len_cost)
        alive_states = np.array(self.synapses['ee'].alive)
        total_length_cost = np.sum(len_costs * alive_states) * self.structural_params.c_length
        
        # Coût énergétique total
        total_cost = spike_cost + syn_cost + total_length_cost
        
        # Pression énergétique
        pressure = max(0.0, (total_cost - self.structural_params.energy_budget) / 
                      self.structural_params.energy_budget)
        
        # Enregistrement
        self.energy_history.append({
            'time_ms': time_ms,
            'spike_cost': spike_cost,
            'syn_cost': syn_cost,
            'length_cost': total_length_cost,
            'total_cost': total_cost,
            'pressure': pressure
        })
        
        return pressure
    
    def run_developmental_experiment(self, duration_ms: float,
                                   update_interval_ms: float = 100.0,
                                   seed: int = 42) -> Dict[str, Any]:
        """
        Exécute une expérience développementale complète.
        
        Parameters
        ----------
        duration_ms : float
            Durée totale de l'expérience
        update_interval_ms : float
            Intervalle de mise à jour des mécanismes structurels
        seed : int
            Graine aléatoire
            
        Returns
        -------
        Dict[str, Any]
            Résultats incluant dynamiques développementales
        """
        
        logger.info(f"Expérience développementale S3 : {duration_ms} ms")
        
        # Configuration reproductibilité
        np.random.seed(seed)
        seed(seed)
        
        # Construction du réseau
        if self.network is None:
            self.build_network()
        
        # Simulation par intervalles
        n_intervals = int(duration_ms / update_interval_ms)
        
        for interval in range(n_intervals):
            current_time = interval * update_interval_ms
            
            # Détermination de la phase
            current_phase = self.determine_developmental_phase(current_time)
            
            # Changement de phase
            if current_phase != self.current_phase:
                logger.info(f"Transition développementale : {self.current_phase} → {current_phase} "
                           f"à t={current_time:.0f}ms")
                self.current_phase = current_phase
                self.phase_start_time = current_time
            
            # Application des mécanismes selon la phase
            if current_phase == 'GROW':
                self.apply_growth_phase(current_time)
            elif current_phase == 'PRUNE':
                self.apply_pruning_phase(current_time)
            
            # Simulation d'un intervalle
            self.network.run(update_interval_ms * ms)
            
            # Enregistrement développemental
            if interval % 10 == 0:  # Tous les 10 intervalles
                self.developmental_history.append({
                    'time_ms': current_time,
                    'phase': current_phase,
                    'synaptic_density': np.mean(self.synapses['ee'].alive),
                    'mean_activity_score': np.mean(self.synapses['ee'].A),
                    'mean_weight': np.mean(self.synapses['ee'].w[self.synapses['ee'].alive == 1]),
                    'n_active_synapses': int(np.sum(self.synapses['ee'].alive))
                })
            
            # Log de progression
            if interval % 50 == 0:
                density = np.mean(self.synapses['ee'].alive)
                logger.info(f"t={current_time:.0f}ms, phase={current_phase}, "
                           f"densité={density:.3f}")
        
        # Compilation des résultats
        results = self._compile_developmental_results(duration_ms, seed)
        
        logger.info("Expérience développementale S3 terminée")
        
        return results
    
    def _compile_developmental_results(self, duration_ms: float, seed: int) -> Dict[str, Any]:
        """Compile les résultats développementaux."""
        
        # Résultats de base
        base_results = self._compile_plasticity_results(duration_ms, seed)
        
        # Trajectoires développementales
        developmental_dynamics = {
            'developmental_trajectory': self.developmental_history,
            'growth_events': self.growth_events,
            'pruning_events': self.pruning_events,
            'energy_dynamics': self.energy_history
        }
        
        # Métriques développementales
        developmental_metrics = self._compute_developmental_metrics()
        
        # Validation développementale
        dev_validation = self._validate_developmental_outcome()
        
        # Fusion des résultats
        developmental_results = {
            **base_results,
            'developmental_dynamics': developmental_dynamics,
            'developmental_metrics': developmental_metrics,
            'developmental_validation': dev_validation
        }
        
        return developmental_results
    
    def _compute_developmental_metrics(self) -> Dict[str, float]:
        """Calcule les métriques développementales."""
        
        if not self.developmental_history:
            return {}
        
        # Densités aux phases clés
        grow_phase = [h for h in self.developmental_history if h['phase'] == 'GROW']
        prune_phase = [h for h in self.developmental_history if h['phase'] == 'PRUNE']
        
        initial_density = self.developmental_history[0]['synaptic_density']
        final_density = self.developmental_history[-1]['synaptic_density']
        
        peak_density = max([h['synaptic_density'] for h in self.developmental_history])
        
        # Efficacité du développement
        target_final = self.structural_params.rho_final_target
        developmental_efficiency = 1.0 - abs(final_density - target_final) / target_final
        
        # Amplitude de l'élagage
        pruning_amplitude = peak_density - final_density
        
        # Stabilité finale
        if len(prune_phase) > 5:
            recent_densities = [h['synaptic_density'] for h in prune_phase[-5:]]
            stability = 1.0 / (1.0 + np.std(recent_densities))
        else:
            stability = 1.0
        
        return {
            'initial_density': initial_density,
            'peak_density': peak_density,
            'final_density': final_density,
            'pruning_amplitude': pruning_amplitude,
            'developmental_efficiency': developmental_efficiency,
            'final_stability': stability,
            'total_growth_events': len(self.growth_events),
            'total_pruning_events': len(self.pruning_events)
        }
    
    def _validate_developmental_outcome(self) -> Dict[str, Any]:
        """Valide le résultat développemental."""
        
        metrics = self._compute_developmental_metrics()
        
        validation = {
            'density_in_range': (0.05 <= metrics.get('final_density', 0) <= 0.4),
            'pruning_occurred': metrics.get('pruning_amplitude', 0) > 0.05,
            'growth_occurred': metrics.get('total_growth_events', 0) > 0,
            'developmental_efficiency_good': metrics.get('developmental_efficiency', 0) > 0.7,
            'final_stability_good': metrics.get('final_stability', 0) > 0.8
        }
        
        validation['all_valid'] = all(validation.values())
        
        return validation

def create_s3_model(network_params: Dict[str, Any] = None,
                   plasticity_params: Dict[str, Any] = None,
                   structural_params: Dict[str, Any] = None) -> S3_StructuralPlasticityNetwork:
    """Crée un modèle S3 avec plasticité structurelle."""
    
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
    
    struct_params = S3_StructuralParameters()
    if structural_params:
        for key, value in structural_params.items():
            if hasattr(struct_params, key):
                setattr(struct_params, key, value)
    
    return S3_StructuralPlasticityNetwork(net_params, plas_params, struct_params)

def validate_s3_model():
    """Valide le modèle S3 avec tests développementaux."""
    
    logger.info("=== Validation du Modèle S3 (Plasticité structurelle) ===")
    
    model = create_s3_model()
    results = model.run_developmental_experiment(duration_ms=8000.0, seed=42)
    
    # Vérifications développementales
    dev_metrics = results['developmental_metrics']
    dev_validation = results['developmental_validation']
    
    # Tests de base
    assert dev_validation['all_valid'], "Validation développementale échouée"
    assert dev_metrics['pruning_amplitude'] > 0.05, "Élagage insuffisant"
    assert dev_metrics['total_growth_events'] > 0, "Aucune croissance détectée"
    
    logger.info("✓ Modèle S3 validé développementalement")
    logger.info(f"  Densité finale : {dev_metrics['final_density']:.3f}")
    logger.info(f"  Amplitude élagage : {dev_metrics['pruning_amplitude']:.3f}")
    logger.info(f"  Efficacité développementale : {dev_metrics['developmental_efficiency']:.3f}")
    
    return True

if __name__ == "__main__":
    validate_s3_model()
    print("Modèle S3 (Plasticité structurelle) validé et prêt !")
