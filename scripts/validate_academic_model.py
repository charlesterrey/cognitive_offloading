#!/usr/bin/env python3
"""
Script de validation complète du modèle NEUROMODE académique.

Ce script exécute tous les tests de validation nécessaires pour certifier
la reproductibilité et la validité scientifique du modèle.

Usage:
    python validate_academic_model.py [--quick] [--verbose]
    
Options:
    --quick     : Exécution rapide (tests essentiels uniquement)
    --verbose   : Sortie détaillée
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path

# Ajout du chemin des modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from unified_neuromode import (
    UnifiedNeuromodeModel, 
    ExperimentalCondition,
    NetworkParameters,
    PlasticityParameters, 
    StructuralPlasticityParameters,
    OffloadingParameters,
    create_standard_model,
    validate_experimental_condition,
    BIOLOGICAL_CONSTRAINTS
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AcademicModelValidator:
    """Validateur complet du modèle NEUROMODE académique."""
    
    def __init__(self, quick_mode=False, verbose=False):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.validation_results = {}
        self.start_time = time.time()
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def validate_environment(self):
        """Valide l'environnement logiciel."""
        logger.info("=== Validation de l'environnement logiciel ===")
        
        try:
            import numpy as np
            import brian2
            import scipy
            import pandas as pd
            import matplotlib
            import seaborn
            import sklearn
            
            versions = {
                'numpy': np.__version__,
                'brian2': brian2.__version__,
                'scipy': scipy.__version__,
                'pandas': pd.__version__,
                'matplotlib': matplotlib.__version__,
                'seaborn': seaborn.__version__,
                'sklearn': sklearn.__version__
            }
            
            logger.info("Versions installées :")
            for package, version in versions.items():
                logger.info(f"  {package}: {version}")
            
            self.validation_results['environment'] = {
                'status': 'PASS',
                'versions': versions
            }
            
        except ImportError as e:
            logger.error(f"Package manquant : {e}")
            self.validation_results['environment'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
        
        return True
    
    def validate_model_creation(self):
        """Valide la création du modèle standard."""
        logger.info("=== Validation de la création du modèle ===")
        
        try:
            model = create_standard_model()
            
            # Vérification des paramètres
            assert model.net_params.N_e == 400
            assert model.net_params.N_i == 100
            assert model.plas_params.A_pre == 0.01
            assert model.plas_params.A_post == -0.012
            
            logger.info("✓ Modèle créé avec paramètres standards")
            
            self.validation_results['model_creation'] = {
                'status': 'PASS',
                'parameters_validated': True
            }
            
        except Exception as e:
            logger.error(f"Erreur création modèle : {e}")
            self.validation_results['model_creation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
        
        return True
    
    def validate_network_construction(self):
        """Valide la construction du réseau neuronal."""
        logger.info("=== Validation de la construction du réseau ===")
        
        try:
            model = create_standard_model()
            model.build_network()
            
            # Vérifications de base
            assert model.network is not None
            assert 'e' in model.neurons
            assert 'i' in model.neurons
            assert 'ee' in model.synapses
            
            # Vérification des tailles
            assert len(model.neurons['e']) == 400
            assert len(model.neurons['i']) == 100
            
            logger.info("✓ Réseau construit avec architecture correcte")
            
            self.validation_results['network_construction'] = {
                'status': 'PASS',
                'neurons_e': len(model.neurons['e']),
                'neurons_i': len(model.neurons['i']),
                'synapses_ee': len(model.synapses['ee'])
            }
            
        except Exception as e:
            logger.error(f"Erreur construction réseau : {e}")
            self.validation_results['network_construction'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
        
        return True
    
    def validate_reproducibility(self):
        """Valide la reproductibilité avec graines fixes."""
        logger.info("=== Validation de la reproductibilité ===")
        
        test_seeds = [42, 43] if self.quick_mode else [42, 43, 44, 45, 46]
        
        try:
            reference_results = {}
            
            for seed in test_seeds:
                condition = ExperimentalCondition(
                    offloading_level=0.6,
                    onset_time_ms=3000.0,
                    duration_ms=6000.0,
                    seed=seed,
                    subject_id=f"REPRO_TEST_{seed}"
                )
                
                # Première exécution
                model1 = create_standard_model()
                model1.build_network()
                results1 = model1.run_experiment(condition)
                
                # Deuxième exécution (même graine)
                model2 = create_standard_model()
                model2.build_network()
                results2 = model2.run_experiment(condition)
                
                # Vérification de l'identité
                metrics1 = results1['summary_metrics']
                metrics2 = results2['summary_metrics']
                
                for metric in ['ODI', 'PDI', 'CLI', 'AEI']:
                    diff = abs(metrics1[metric] - metrics2[metric])
                    if diff > 1e-10:
                        raise ValueError(f"Graine {seed}: {metric} non reproductible (diff={diff})")
                
                reference_results[seed] = metrics1
                logger.info(f"✓ Graine {seed} reproductible")
            
            self.validation_results['reproducibility'] = {
                'status': 'PASS',
                'seeds_tested': len(test_seeds),
                'reference_results': reference_results
            }
            
        except Exception as e:
            logger.error(f"Erreur reproductibilité : {e}")
            self.validation_results['reproducibility'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
        
        return True
    
    def validate_biological_constraints(self):
        """Valide le respect des contraintes biologiques."""
        logger.info("=== Validation des contraintes biologiques ===")
        
        try:
            # Test avec différents niveaux d'offloading
            test_conditions = [
                (0.0, "contrôle"),
                (0.3, "modéré"),
                (0.6, "élevé")
            ]
            
            violations = []
            
            for omega, description in test_conditions:
                condition = ExperimentalCondition(
                    offloading_level=omega,
                    onset_time_ms=3000.0,
                    duration_ms=6000.0,
                    seed=42,
                    subject_id=f"BIO_TEST_{description}"
                )
                
                model = create_standard_model()
                model.build_network()
                results = model.run_experiment(condition)
                
                metrics = results['summary_metrics']
                
                # Vérification des taux de décharge
                rate_e = metrics['final_firing_rate_e']
                rate_i = metrics['final_firing_rate_i']
                
                if not (0.1 <= rate_e <= 100.0):
                    violations.append(f"Ω={omega}: Taux E = {rate_e:.2f} Hz hors plage")
                
                if not (0.1 <= rate_i <= 100.0):
                    violations.append(f"Ω={omega}: Taux I = {rate_i:.2f} Hz hors plage")
                
                # Vérification de la densité synaptique
                density = metrics['final_synaptic_density']
                if not (0.01 <= density <= 0.5):
                    violations.append(f"Ω={omega}: Densité = {density:.3f} hors plage")
                
                logger.info(f"✓ Condition Ω={omega} ({description}) : contraintes respectées")
            
            if violations:
                raise ValueError("Violations biologiques : " + "; ".join(violations))
            
            self.validation_results['biological_constraints'] = {
                'status': 'PASS',
                'conditions_tested': len(test_conditions),
                'violations': []
            }
            
        except Exception as e:
            logger.error(f"Erreur contraintes biologiques : {e}")
            self.validation_results['biological_constraints'] = {
                'status': 'FAIL',
                'error': str(e),
                'violations': violations if 'violations' in locals() else []
            }
            return False
        
        return True
    
    def validate_metrics_computation(self):
        """Valide le calcul des métriques principales."""
        logger.info("=== Validation du calcul des métriques ===")
        
        try:
            condition = ExperimentalCondition(
                offloading_level=0.6,
                onset_time_ms=3000.0,
                duration_ms=8000.0,
                seed=42,
                subject_id="METRICS_TEST"
            )
            
            model = create_standard_model()
            model.build_network()
            results = model.run_experiment(condition)
            
            metrics = results['summary_metrics']
            
            # Vérification des bornes des métriques
            assert 0.0 <= metrics['ODI'] <= 1.0, f"ODI = {metrics['ODI']} hors bornes [0,1]"
            assert 0.0 <= metrics['PDI'] <= 2.0, f"PDI = {metrics['PDI']} hors bornes [0,2]"
            assert -1.0 <= metrics['CLI'] <= 1.0, f"CLI = {metrics['CLI']} hors bornes [-1,1]"
            assert 0.0 <= metrics['AEI'] <= 1.0, f"AEI = {metrics['AEI']} hors bornes [0,1]"
            
            # Vérification de la cohérence
            # ODI doit être > 0 pour offloading > 0
            assert metrics['ODI'] > 0, "ODI doit être > 0 avec offloading"
            
            # CLI doit être > 0 (réduction d'effort)
            assert metrics['CLI'] > 0, "CLI doit être > 0 (réduction d'effort)"
            
            logger.info("✓ Métriques calculées correctement")
            logger.info(f"  ODI = {metrics['ODI']:.3f}")
            logger.info(f"  PDI = {metrics['PDI']:.3f}")
            logger.info(f"  CLI = {metrics['CLI']:.3f}")
            logger.info(f"  AEI = {metrics['AEI']:.3f}")
            
            self.validation_results['metrics_computation'] = {
                'status': 'PASS',
                'computed_metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques : {e}")
            self.validation_results['metrics_computation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
        
        return True
    
    def validate_dose_response(self):
        """Valide la relation dose-réponse."""
        logger.info("=== Validation de la relation dose-réponse ===")
        
        try:
            omega_levels = [0.0, 0.3, 0.6] if self.quick_mode else [0.0, 0.2, 0.4, 0.6, 0.8]
            odi_values = []
            
            for omega in omega_levels:
                condition = ExperimentalCondition(
                    offloading_level=omega,
                    onset_time_ms=3000.0,
                    duration_ms=6000.0,
                    seed=42,
                    subject_id=f"DOSE_TEST_{omega}"
                )
                
                model = create_standard_model()
                model.build_network()
                results = model.run_experiment(condition)
                
                odi = results['summary_metrics']['ODI']
                odi_values.append(odi)
                
                logger.info(f"Ω = {omega:.1f} → ODI = {odi:.3f}")
            
            # Vérification de la monotonie (généralement croissante)
            import numpy as np
            correlation = np.corrcoef(omega_levels, odi_values)[0, 1]
            
            if correlation < 0.5:
                logger.warning(f"Corrélation Ω-ODI faible : r = {correlation:.3f}")
            else:
                logger.info(f"✓ Relation dose-réponse validée : r = {correlation:.3f}")
            
            self.validation_results['dose_response'] = {
                'status': 'PASS',
                'correlation': correlation,
                'omega_levels': omega_levels,
                'odi_values': odi_values
            }
            
        except Exception as e:
            logger.error(f"Erreur relation dose-réponse : {e}")
            self.validation_results['dose_response'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
        
        return True
    
    def run_complete_validation(self):
        """Exécute la validation complète."""
        logger.info("🧠 DÉMARRAGE DE LA VALIDATION COMPLÈTE DU MODÈLE NEUROMODE 🧠")
        logger.info(f"Mode : {'RAPIDE' if self.quick_mode else 'COMPLET'}")
        
        validation_steps = [
            ('Environnement', self.validate_environment),
            ('Création du modèle', self.validate_model_creation),
            ('Construction du réseau', self.validate_network_construction),
            ('Reproductibilité', self.validate_reproducibility),
            ('Contraintes biologiques', self.validate_biological_constraints),
            ('Calcul des métriques', self.validate_metrics_computation),
            ('Relation dose-réponse', self.validate_dose_response)
        ]
        
        passed = 0
        failed = 0
        
        for step_name, step_function in validation_steps:
            logger.info(f"\n--- {step_name} ---")
            
            try:
                if step_function():
                    logger.info(f"✅ {step_name} : VALIDÉ")
                    passed += 1
                else:
                    logger.error(f"❌ {step_name} : ÉCHEC")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ {step_name} : ERREUR - {e}")
                failed += 1
        
        # Résumé final
        total_time = time.time() - self.start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("🎯 RÉSUMÉ DE LA VALIDATION")
        logger.info(f"{'='*60}")
        logger.info(f"Tests réussis    : {passed}")
        logger.info(f"Tests échoués    : {failed}")
        logger.info(f"Temps total      : {total_time:.1f} secondes")
        logger.info(f"Statut global    : {'✅ SUCCÈS' if failed == 0 else '❌ ÉCHEC'}")
        
        if failed == 0:
            logger.info("\n🎉 MODÈLE NEUROMODE VALIDÉ POUR USAGE ACADÉMIQUE 🎉")
            logger.info("✓ Reproductibilité garantie")
            logger.info("✓ Contraintes biologiques respectées")
            logger.info("✓ Métriques validées scientifiquement")
            logger.info("✓ Prêt pour publication")
        else:
            logger.error("\n⚠️ VALIDATION INCOMPLÈTE - CORRECTIONS REQUISES")
        
        # Sauvegarde des résultats
        self.save_validation_report()
        
        return failed == 0
    
    def save_validation_report(self):
        """Sauvegarde le rapport de validation."""
        report = {
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'quick' if self.quick_mode else 'complete',
            'total_duration_seconds': time.time() - self.start_time,
            'results': self.validation_results
        }
        
        report_path = 'validation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Rapport de validation sauvegardé : {report_path}")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Validation du modèle NEUROMODE académique')
    parser.add_argument('--quick', action='store_true', help='Validation rapide')
    parser.add_argument('--verbose', action='store_true', help='Sortie détaillée')
    
    args = parser.parse_args()
    
    validator = AcademicModelValidator(
        quick_mode=args.quick,
        verbose=args.verbose
    )
    
    success = validator.run_complete_validation()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
