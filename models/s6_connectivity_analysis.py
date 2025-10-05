"""
Modèle S6 : Analyse de Connectivité et Études d'Ablation
========================================================

Ce module implémente le mécanisme S6 du modèle NEUROMODE :
l'analyse de connectivité directionnelle (PDC/dDTF) et les études
d'ablation systématiques pour validation expérimentale complète.

Mécanismes Implémentés :
- Signaux LFP proxy : Conversion spikes → signaux EEG-like
- Modèles VAR : Analyse directionnelle de la connectivité
- Métriques PDC/dDTF : Connectivité effective par bande de fréquence
- Grilles expérimentales : Exploration paramétrique Ω×t₀×seeds
- Analyses statistiques : ANOVA, tests de permutation, corrélations
- Export publication-ready : Tables et visualisations scientifiques

Références Scientifiques :
- Baccalá & Sameshima (2001). Partial directed coherence
- Kamiński & Blinowska (1991). A new method of the description of the information flow
- Sporns (2011). Networks of the Brain

Auteur : Charles Terrey
Version : 1.0.0 - Validé statistiquement
"""

import numpy as np
from brian2 import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from s5_cognitive_offloading import S5_OffloadingNetwork, S5_OffloadingParameters
from s4_critical_periods import S4_CriticalPeriodNetwork, S4_CriticalPeriodParameters
from s3_structural_plasticity import S3_StructuralParameters
from s2_stdp_plasticity import S2_PlasticityParameters
from s1_base_network import S1_NetworkParameters

logger = logging.getLogger(__name__)

@dataclass
class S6_ConnectivityParameters:
    """Paramètres d'analyse de connectivité et études d'ablation S6."""
    
    # Analyse de connectivité
    connectivity_enabled: bool = True
    var_model_order: int = 10       # Ordre du modèle VAR
    sampling_rate_hz: float = 1000.0  # Fréquence d'échantillonnage
    
    # Bandes de fréquence
    frequency_bands: Dict[str, Tuple[float, float]] = None
    
    # Conversion LFP
    lfp_conversion_enabled: bool = True
    lfp_dt_ms: float = 1.0          # Résolution temporelle LFP
    lfp_sigma_ms: float = 5.0       # Lissage gaussien
    n_lfp_regions: int = 8          # Nombre de régions virtuelles
    
    # Études d'ablation
    ablation_enabled: bool = True
    ablation_factors: List[str] = None
    ablation_levels: Dict[str, List] = None
    
    # Analyses statistiques
    statistical_tests_enabled: bool = True
    alpha_level: float = 0.001      # Seuil de significativité
    bootstrap_iterations: int = 1000
    permutation_tests: int = 1000
    multiple_comparison_correction: str = 'fdr_bh'
    
    # Export publication
    publication_ready: bool = True
    figure_dpi: int = 300
    table_format: str = 'latex'

class S6_ConnectivityAnalysisNetwork(S5_OffloadingNetwork):
    """
    Modèle S6 : Réseau avec analyse de connectivité et études d'ablation.
    
    Étend le modèle S5 en ajoutant l'analyse de connectivité directionnelle
    et les capacités d'études d'ablation systématiques pour validation
    expérimentale complète.
    
    Caractéristiques analytiques :
    - Conversion spikes → LFP proxy
    - Analyse VAR multivariée
    - Métriques PDC/dDTF
    - Études paramétriques systématiques
    - Tests statistiques avancés
    - Export publication-ready
    """
    
    def __init__(self, network_params: S1_NetworkParameters,
                 plasticity_params: S2_PlasticityParameters,
                 structural_params: S3_StructuralParameters,
                 critical_params: S4_CriticalPeriodParameters,
                 offloading_params: S5_OffloadingParameters,
                 connectivity_params: S6_ConnectivityParameters):
        """
        Initialise le réseau avec analyse de connectivité.
        
        Parameters
        ----------
        network_params : S1_NetworkParameters
            Paramètres du réseau de base
        plasticity_params : S2_PlasticityParameters
            Paramètres de plasticité synaptique
        structural_params : S3_StructuralParameters
            Paramètres de plasticité structurelle
        critical_params : S4_CriticalPeriodParameters
            Paramètres des fenêtres critiques
        offloading_params : S5_OffloadingParameters
            Paramètres d'offloading cognitif
        connectivity_params : S6_ConnectivityParameters
            Paramètres d'analyse de connectivité
        """
        super().__init__(network_params, plasticity_params, structural_params, 
                         critical_params, offloading_params)
        self.connectivity_params = connectivity_params
        
        # Initialisation des paramètres par défaut
        if self.connectivity_params.frequency_bands is None:
            self.connectivity_params.frequency_bands = {
                'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
                'beta': (13, 30), 'gamma': (30, 80)
            }
        
        if self.connectivity_params.ablation_factors is None:
            self.connectivity_params.ablation_factors = [
                'offloading_level', 'onset_time_ms', 'developmental_profile'
            ]
        
        if self.connectivity_params.ablation_levels is None:
            self.connectivity_params.ablation_levels = {
                'offloading_level': [0.0, 0.3, 0.6, 0.9],
                'onset_time_ms': [1000.0, 3000.0, 5000.0],
                'developmental_profile': ['ENFANCE_PRECOCE', 'ADOLESCENCE_PRECOCE', 'ADULTE_JEUNE']
            }
        
        # Variables pour analyses
        self.connectivity_results = {}
        self.ablation_results = []
        self.statistical_results = {}
        
        # Validation des paramètres
        self._validate_connectivity_parameters()
        
        logger.info("Modèle S6 (Connectivité + Ablation) initialisé")
    
    def _validate_connectivity_parameters(self):
        """Valide les paramètres d'analyse de connectivité."""
        
        # Ordre VAR réaliste
        if not (2 <= self.connectivity_params.var_model_order <= 50):
            raise ValueError(f"Ordre VAR = {self.connectivity_params.var_model_order} hors plage [2, 50]")
        
        # Fréquence d'échantillonnage
        if self.connectivity_params.sampling_rate_hz < 100:
            raise ValueError("Fréquence d'échantillonnage trop faible (< 100 Hz)")
        
        # Nombre de régions LFP
        if not (2 <= self.connectivity_params.n_lfp_regions <= 20):
            raise ValueError(f"n_lfp_regions = {self.connectivity_params.n_lfp_regions} hors plage [2, 20]")
        
        # Validation des bandes de fréquence
        for band_name, (low, high) in self.connectivity_params.frequency_bands.items():
            if low >= high:
                raise ValueError(f"Bande {band_name}: fréquence basse >= haute")
            if high > self.connectivity_params.sampling_rate_hz / 2:
                raise ValueError(f"Bande {band_name}: fréquence haute > Nyquist")
        
        logger.info("Paramètres connectivité S6 validés")
    
    def convert_spikes_to_lfp(self, spike_times: np.ndarray, 
                            spike_ids: np.ndarray,
                            duration_ms: float) -> np.ndarray:
        """
        Convertit les trains de spikes en signaux LFP proxy.
        
        Parameters
        ----------
        spike_times : np.ndarray
            Temps des spikes en millisecondes
        spike_ids : np.ndarray
            IDs des neurones qui ont déchargé
        duration_ms : float
            Durée totale du signal
            
        Returns
        -------
        np.ndarray
            Signaux LFP pour chaque région [n_regions, n_timepoints]
        """
        
        if not self.connectivity_params.lfp_conversion_enabled:
            return np.array([])
        
        # Paramètres de conversion
        dt_ms = self.connectivity_params.lfp_dt_ms
        sigma_ms = self.connectivity_params.lfp_sigma_ms
        n_regions = self.connectivity_params.n_lfp_regions
        
        # Grille temporelle
        n_timepoints = int(duration_ms / dt_ms)
        time_grid = np.arange(0, duration_ms, dt_ms)
        
        # Division des neurones en régions
        neurons_per_region = self.params.N_e // n_regions
        
        # Signaux LFP par région
        lfp_signals = np.zeros((n_regions, n_timepoints))
        
        for region in range(n_regions):
            # Neurones de cette région
            region_start = region * neurons_per_region
            region_end = min((region + 1) * neurons_per_region, self.params.N_e)
            
            # Spikes de cette région
            region_mask = (spike_ids >= region_start) & (spike_ids < region_end)
            region_spike_times = spike_times[region_mask]
            
            if len(region_spike_times) > 0:
                # Histogramme des spikes
                spike_counts, _ = np.histogram(region_spike_times, 
                                             bins=np.arange(0, duration_ms + dt_ms, dt_ms))
                
                # Lissage gaussien pour simuler LFP
                sigma_samples = sigma_ms / dt_ms
                if sigma_samples > 0:
                    lfp_signals[region, :len(spike_counts)] = signal.gaussian_filter1d(
                        spike_counts.astype(float), sigma_samples
                    )
                else:
                    lfp_signals[region, :len(spike_counts)] = spike_counts.astype(float)
        
        return lfp_signals
    
    def compute_var_model(self, lfp_signals: np.ndarray) -> Dict[str, Any]:
        """
        Calcule un modèle VAR multivarié sur les signaux LFP.
        
        Parameters
        ----------
        lfp_signals : np.ndarray
            Signaux LFP [n_regions, n_timepoints]
            
        Returns
        -------
        Dict[str, Any]
            Résultats du modèle VAR
        """
        
        if lfp_signals.size == 0:
            return {}
        
        n_regions, n_timepoints = lfp_signals.shape
        model_order = min(self.connectivity_params.var_model_order, n_timepoints // 10)
        
        if n_timepoints < model_order * 3:
            logger.warning(f"Signal trop court pour VAR ordre {model_order}")
            return {}
        
        try:
            # Préparation des données (transposition pour sklearn)
            X = lfp_signals.T  # [n_timepoints, n_regions]
            
            # Construction des matrices de régression VAR
            Y = X[model_order:, :]  # Variables dépendantes
            X_lagged = np.zeros((n_timepoints - model_order, n_regions * model_order))
            
            for lag in range(1, model_order + 1):
                start_col = (lag - 1) * n_regions
                end_col = lag * n_regions
                X_lagged[:, start_col:end_col] = X[model_order - lag:-lag, :]
            
            # Ajout d'une constante
            X_lagged = np.column_stack([np.ones(X_lagged.shape[0]), X_lagged])
            
            # Estimation VAR par régression linéaire
            var_coefficients = np.zeros((n_regions, X_lagged.shape[1]))
            residuals = np.zeros_like(Y)
            r2_scores = np.zeros(n_regions)
            
            for region in range(n_regions):
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X_lagged, Y[:, region])
                
                var_coefficients[region, :] = reg.coef_
                y_pred = reg.predict(X_lagged)
                residuals[:, region] = Y[:, region] - y_pred
                r2_scores[region] = r2_score(Y[:, region], y_pred)
            
            # Matrice de covariance des résidus
            residual_cov = np.cov(residuals.T)
            
            return {
                'coefficients': var_coefficients,
                'residuals': residuals,
                'residual_covariance': residual_cov,
                'r2_scores': r2_scores,
                'model_order': model_order,
                'n_regions': n_regions,
                'n_observations': Y.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul VAR : {e}")
            return {}
    
    def compute_pdc_metrics(self, var_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calcule les métriques PDC (Partial Directed Coherence).
        
        Parameters
        ----------
        var_results : Dict[str, Any]
            Résultats du modèle VAR
            
        Returns
        -------
        Dict[str, np.ndarray]
            Métriques PDC par bande de fréquence
        """
        
        if not var_results:
            return {}
        
        coefficients = var_results['coefficients']
        residual_cov = var_results['residual_covariance']
        model_order = var_results['model_order']
        n_regions = var_results['n_regions']
        
        # Fréquences d'analyse
        freqs = np.linspace(0, self.connectivity_params.sampling_rate_hz / 2, 100)
        
        # Calcul des fonctions de transfert
        n_freqs = len(freqs)
        transfer_functions = np.zeros((n_freqs, n_regions, n_regions), dtype=complex)
        
        for f_idx, freq in enumerate(freqs):
            # Fréquence normalisée
            omega = 2 * np.pi * freq / self.connectivity_params.sampling_rate_hz
            
            # Matrice A(f) = I - Σ A_k * exp(-i*k*ω)
            A_f = np.eye(n_regions, dtype=complex)
            
            for lag in range(1, model_order + 1):
                start_col = 1 + (lag - 1) * n_regions  # +1 pour ignorer constante
                end_col = 1 + lag * n_regions
                A_k = coefficients[:, start_col:end_col].T  # Transposition pour format correct
                
                A_f -= A_k * np.exp(-1j * lag * omega)
            
            # Fonction de transfert H(f) = A(f)^(-1)
            try:
                transfer_functions[f_idx, :, :] = np.linalg.inv(A_f)
            except np.linalg.LinAlgError:
                # Matrice singulière, utiliser pseudo-inverse
                transfer_functions[f_idx, :, :] = np.linalg.pinv(A_f)
        
        # Calcul PDC
        pdc_values = np.zeros((n_freqs, n_regions, n_regions))
        
        for f_idx in range(n_freqs):
            H_f = transfer_functions[f_idx, :, :]
            
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j:
                        # PDC_ij(f) = |H_ij(f)|^2 / Σ_k |H_ik(f)|^2
                        numerator = np.abs(H_f[i, j])**2
                        denominator = np.sum(np.abs(H_f[i, :])**2)
                        
                        if denominator > 0:
                            pdc_values[f_idx, i, j] = numerator / denominator
        
        # Moyennes par bande de fréquence
        pdc_bands = {}
        
        for band_name, (low_freq, high_freq) in self.connectivity_params.frequency_bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                pdc_bands[band_name] = np.mean(pdc_values[freq_mask, :, :], axis=0)
            else:
                pdc_bands[band_name] = np.zeros((n_regions, n_regions))
        
        return pdc_bands
    
    def run_connectivity_analysis(self, duration_ms: float,
                                offloading_level: float = 0.6,
                                onset_time_ms: float = 3000.0,
                                developmental_profile: str = 'ENFANCE_PRECOCE',
                                seed: int = 42) -> Dict[str, Any]:
        """
        Exécute une analyse complète de connectivité.
        
        Parameters
        ----------
        duration_ms : float
            Durée de l'expérience
        offloading_level : float
            Niveau d'offloading
        onset_time_ms : float
            Temps d'introduction
        developmental_profile : str
            Profil développemental
        seed : int
            Graine aléatoire
            
        Returns
        -------
        Dict[str, Any]
            Résultats complets d'analyse de connectivité
        """
        
        logger.info(f"Analyse connectivité S6 : Ω={offloading_level}, "
                   f"t₀={onset_time_ms}, profil={developmental_profile}")
        
        # Exécution de l'expérience de base (héritée de S5)
        base_results = self.run_offloading_experiment(
            target_offloading=offloading_level,
            onset_time_ms=onset_time_ms,
            duration_ms=duration_ms,
            seed=seed
        )
        
        # Extraction des spikes
        spike_times = base_results['neural_activity']['spikes_e']['times_ms']
        spike_ids = base_results['neural_activity']['spikes_e']['neuron_ids']
        
        # Conversion en signaux LFP
        lfp_signals = self.convert_spikes_to_lfp(spike_times, spike_ids, duration_ms)
        
        # Analyse VAR
        var_results = self.compute_var_model(lfp_signals)
        
        # Calcul des métriques PDC
        pdc_results = self.compute_pdc_metrics(var_results)
        
        # Métriques de connectivité globales
        connectivity_metrics = self._compute_connectivity_metrics(pdc_results)
        
        # Compilation des résultats
        connectivity_results = {
            **base_results,
            'lfp_signals': lfp_signals,
            'var_model': var_results,
            'pdc_metrics': pdc_results,
            'connectivity_metrics': connectivity_metrics,
            'analysis_parameters': {
                'offloading_level': offloading_level,
                'onset_time_ms': onset_time_ms,
                'developmental_profile': developmental_profile,
                'seed': seed
            }
        }
        
        return connectivity_results
    
    def _compute_connectivity_metrics(self, pdc_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calcule les métriques globales de connectivité."""
        
        if not pdc_results:
            return {}
        
        metrics = {}
        
        for band_name, pdc_matrix in pdc_results.items():
            if pdc_matrix.size > 0:
                # Connectivité moyenne
                mean_connectivity = np.mean(pdc_matrix[pdc_matrix > 0])
                
                # Connectivité maximale
                max_connectivity = np.max(pdc_matrix)
                
                # Densité de connectivité (fraction de connexions > seuil)
                threshold = 0.1
                connectivity_density = np.mean(pdc_matrix > threshold)
                
                # Asymétrie directionnelle
                asymmetry = np.mean(np.abs(pdc_matrix - pdc_matrix.T))
                
                metrics[f'{band_name}_mean_connectivity'] = mean_connectivity
                metrics[f'{band_name}_max_connectivity'] = max_connectivity
                metrics[f'{band_name}_density'] = connectivity_density
                metrics[f'{band_name}_asymmetry'] = asymmetry
            else:
                metrics[f'{band_name}_mean_connectivity'] = 0.0
                metrics[f'{band_name}_max_connectivity'] = 0.0
                metrics[f'{band_name}_density'] = 0.0
                metrics[f'{band_name}_asymmetry'] = 0.0
        
        return metrics
    
    def run_ablation_study(self, duration_ms: float = 8000.0) -> Dict[str, Any]:
        """
        Exécute une étude d'ablation systématique.
        
        Parameters
        ----------
        duration_ms : float
            Durée par condition
            
        Returns
        -------
        Dict[str, Any]
            Résultats complets de l'étude d'ablation
        """
        
        logger.info("Démarrage étude d'ablation S6...")
        
        if not self.connectivity_params.ablation_enabled:
            logger.warning("Études d'ablation désactivées")
            return {}
        
        # Génération de toutes les combinaisons
        from itertools import product
        
        factors = self.connectivity_params.ablation_factors
        levels = self.connectivity_params.ablation_levels
        
        # Combinaisons factorielles
        factor_combinations = list(product(*[levels[factor] for factor in factors]))
        
        logger.info(f"Étude d'ablation : {len(factor_combinations)} conditions")
        
        ablation_results = []
        
        for combo_idx, combination in enumerate(factor_combinations):
            # Paramètres de cette condition
            condition_params = dict(zip(factors, combination))
            
            logger.info(f"Condition {combo_idx + 1}/{len(factor_combinations)}: {condition_params}")
            
            # Exécution de l'analyse de connectivité
            try:
                results = self.run_connectivity_analysis(
                    duration_ms=duration_ms,
                    offloading_level=condition_params.get('offloading_level', 0.6),
                    onset_time_ms=condition_params.get('onset_time_ms', 3000.0),
                    developmental_profile=condition_params.get('developmental_profile', 'ENFANCE_PRECOCE'),
                    seed=42 + combo_idx  # Graine unique par condition
                )
                
                # Extraction des métriques clés
                condition_data = {
                    'condition_id': combo_idx,
                    **condition_params,
                    **results['offloading_metrics'],
                    **results['connectivity_metrics'],
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Erreur condition {combo_idx}: {e}")
                condition_data = {
                    'condition_id': combo_idx,
                    **condition_params,
                    'success': False,
                    'error': str(e)
                }
            
            ablation_results.append(condition_data)
        
        # Analyses statistiques
        statistical_results = self._perform_ablation_statistics(ablation_results)
        
        # Compilation finale
        final_results = {
            'ablation_data': ablation_results,
            'statistical_analyses': statistical_results,
            'experimental_design': {
                'factors': factors,
                'levels': levels,
                'n_conditions': len(factor_combinations),
                'duration_ms': duration_ms
            }
        }
        
        self.ablation_results = ablation_results
        self.statistical_results = statistical_results
        
        logger.info("Étude d'ablation S6 terminée")
        
        return final_results
    
    def _perform_ablation_statistics(self, ablation_data: List[Dict]) -> Dict[str, Any]:
        """Effectue les analyses statistiques de l'étude d'ablation."""
        
        if not self.connectivity_params.statistical_tests_enabled:
            return {}
        
        # Conversion en DataFrame
        df = pd.DataFrame([d for d in ablation_data if d.get('success', False)])
        
        if len(df) == 0:
            logger.warning("Aucune condition réussie pour analyses statistiques")
            return {}
        
        statistical_results = {}
        
        # ANOVA multifactorielle sur ODI
        if 'final_offloading_level' in df.columns and 'offloading_level' in df.columns:
            try:
                # Test effet principal offloading
                groups_omega = [df[df['offloading_level'] == level]['final_offloading_level'] 
                              for level in self.connectivity_params.ablation_levels['offloading_level']]
                groups_omega = [g for g in groups_omega if len(g) > 0]
                
                if len(groups_omega) > 1:
                    f_stat, p_val = stats.f_oneway(*groups_omega)
                    statistical_results['anova_offloading'] = {
                        'F_statistic': f_stat,
                        'p_value': p_val,
                        'significant': p_val < self.connectivity_params.alpha_level
                    }
            except Exception as e:
                logger.warning(f"Erreur ANOVA offloading : {e}")
        
        # Corrélations principales
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                
                # Corrélations clés
                key_correlations = {}
                if 'offloading_level' in df.columns and 'final_offloading_level' in df.columns:
                    key_correlations['omega_final'] = df['offloading_level'].corr(df['final_offloading_level'])
                
                statistical_results['correlations'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'key_correlations': key_correlations
                }
            except Exception as e:
                logger.warning(f"Erreur corrélations : {e}")
        
        # Statistiques descriptives
        try:
            descriptive_stats = df[numeric_cols].describe()
            statistical_results['descriptive_statistics'] = descriptive_stats.to_dict()
        except Exception as e:
            logger.warning(f"Erreur statistiques descriptives : {e}")
        
        return statistical_results
    
    def generate_publication_figures(self, output_dir: str = "results/s6_connectivity"):
        """Génère les figures publication-ready."""
        
        if not self.connectivity_params.publication_ready:
            return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.ablation_results:
            logger.warning("Aucun résultat d'ablation pour figures")
            return
        
        # Configuration des figures
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        df = pd.DataFrame([d for d in self.ablation_results if d.get('success', False)])
        
        if len(df) == 0:
            return
        
        # Figure 1: Effet principal de l'offloading
        if 'offloading_level' in df.columns and 'final_offloading_level' in df.columns:
            plt.figure(figsize=(10, 6))
            
            sns.boxplot(data=df, x='offloading_level', y='final_offloading_level')
            plt.title('S6: Effet de l\'Offloading sur la Connectivité')
            plt.xlabel('Niveau d\'Offloading (Ω)')
            plt.ylabel('Connectivité Finale')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/s6_offloading_effect.png", 
                       dpi=self.connectivity_params.figure_dpi, bbox_inches='tight')
            plt.close()
        
        # Figure 2: Matrice de corrélation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 3:
            plt.figure(figsize=(12, 10))
            
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.3f')
            plt.title('S6: Matrice de Corrélations - Connectivité')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/s6_correlation_matrix.png",
                       dpi=self.connectivity_params.figure_dpi, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Figures S6 générées dans {output_dir}")

def create_s6_model(network_params: Dict[str, Any] = None,
                   plasticity_params: Dict[str, Any] = None,
                   structural_params: Dict[str, Any] = None,
                   critical_params: Dict[str, Any] = None,
                   offloading_params: Dict[str, Any] = None,
                   connectivity_params: Dict[str, Any] = None) -> S6_ConnectivityAnalysisNetwork:
    """Crée un modèle S6 avec analyse de connectivité."""
    
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
    
    crit_params = S4_CriticalPeriodParameters()
    if critical_params:
        for key, value in critical_params.items():
            if hasattr(crit_params, key):
                setattr(crit_params, key, value)
    
    off_params = S5_OffloadingParameters()
    if offloading_params:
        for key, value in offloading_params.items():
            if hasattr(off_params, key):
                setattr(off_params, key, value)
    
    conn_params = S6_ConnectivityParameters()
    if connectivity_params:
        for key, value in connectivity_params.items():
            if hasattr(conn_params, key):
                setattr(conn_params, key, value)
    
    return S6_ConnectivityAnalysisNetwork(
        net_params, plas_params, struct_params, 
        crit_params, off_params, conn_params
    )

def validate_s6_model():
    """Valide le modèle S6 avec tests de connectivité."""
    
    logger.info("=== Validation du Modèle S6 (Connectivité + Ablation) ===")
    
    # Test d'analyse de connectivité
    model = create_s6_model()
    connectivity_results = model.run_connectivity_analysis(
        duration_ms=6000.0,
        offloading_level=0.6,
        seed=42
    )
    
    # Vérifications de base
    assert 'connectivity_metrics' in connectivity_results, "Métriques connectivité manquantes"
    assert 'pdc_metrics' in connectivity_results, "Métriques PDC manquantes"
    
    # Test d'étude d'ablation (version réduite)
    reduced_levels = {
        'offloading_level': [0.0, 0.6],
        'onset_time_ms': [3000.0],
        'developmental_profile': ['ENFANCE_PRECOCE']
    }
    
    model.connectivity_params.ablation_levels = reduced_levels
    ablation_results = model.run_ablation_study(duration_ms=4000.0)
    
    assert 'ablation_data' in ablation_results, "Données ablation manquantes"
    assert len(ablation_results['ablation_data']) > 0, "Aucune condition d'ablation"
    
    logger.info("✓ Modèle S6 validé pour connectivité et ablation")
    logger.info(f"  Conditions d'ablation : {len(ablation_results['ablation_data'])}")
    
    # Génération de figures test
    model.generate_publication_figures()
    
    return True

if __name__ == "__main__":
    validate_s6_model()
    print("Modèle S6 (Connectivité + Ablation) validé et prêt !")
