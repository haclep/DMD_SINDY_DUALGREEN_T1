# ==============================================================================
# THREE-PHASE DRUG DISCOVERY TRADING SYSTEM
# ==============================================================================
# Phase 1: R&D - Stochastic exploration to find best model configurations
# Phase 2: Clinical Trials - Validate best candidates with rigorous testing  
# Phase 3: Production - Use proven models deterministically for trading
# ==============================================================================

import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TradingDrugDiscovery:
    """
    Three-phase system for discovering and validating trading models
    """
    
    def __init__(self, base_seed=42):
        self.base_seed = base_seed
        self.phase1_results = []  # R&D results
        self.phase2_results = []  # Clinical trial results  
        self.production_models = {}  # Proven models for production
        
    # =========================================================================
    # PHASE 1: R&D - STOCHASTIC EXPLORATION
    # =========================================================================
    
    def phase1_rd_exploration(self, sample_folder, n_drug_candidates=10, prediction_horizon=5, save_results=True):  # ADD prediction_horizon
        """
        Phase 1: R&D - Discover promising model configurations
        
        Like pharmaceutical R&D, we test many random combinations to find
        promising "drug candidates" (model configurations)
        """
        print("🧬 PHASE 1: R&D - STOCHASTIC DRUG DISCOVERY")
        print("="*80)
        print(f"🔬 Testing {n_drug_candidates} different 'drug formulations'")
        print(f"📅 Prediction Horizon: {prediction_horizon} days")  # NEW
        print(f"📁 Using sample data from: {sample_folder}")
        print("="*80)
        
        # Get a representative sample file for testing
        sample_files = list(Path(sample_folder).glob("*.csv"))
        if not sample_files:
            raise ValueError(f"No CSV files found in {sample_folder}")
        
        # Use first file as representative sample
        sample_file = sample_files[0]
        print(f"🧪 Using {sample_file.name} as R&D test subject")
        
        # Test different "drug formulations" (model configurations)
        candidates = []
        
        for drug_id in range(n_drug_candidates):
            print(f"\n🔬 Testing Drug Candidate #{drug_id + 1}")
            
            # Each "drug" has a unique random seed (different formulation)
            drug_seed = self.base_seed + drug_id * 137  # Prime number spacing
            np.random.seed(drug_seed)
            
            try:
                # Test this drug candidate
                analyzer = IndividualMethodsAnalyzer(target_metric='log_price_ratio_sma20', prediction_horizon=prediction_horizon)  # CHANGED
                insight = analyzer.analyze_single_file(str(sample_file))
                
                # Evaluate drug effectiveness
                effectiveness_score = self._evaluate_drug_effectiveness(insight)
                
                # Store drug candidate info
                drug_candidate = {
                    'drug_id': drug_id + 1,
                    'seed': drug_seed,
                    'effectiveness_score': effectiveness_score,
                    'dmd_available': insight.dmd_prediction is not None,
                    'sindy_available': insight.sindy_prediction is not None,
                    'test_ticker': sample_file.stem,
                    'discovery_date': datetime.now().isoformat()
                }
                
                # Add detailed results if both models worked
                if insight.dmd_prediction and insight.sindy_prediction:
                    drug_candidate.update({
                        'dmd_confidence': insight.dmd_prediction['confidence'],
                        'sindy_confidence': insight.sindy_prediction['confidence'],
                        'dmd_action': insight.dmd_prediction['action'],
                        'sindy_action': insight.sindy_prediction['action'],
                        'agreement': self._check_agreement(insight)
                    })
                
                candidates.append(drug_candidate)
                
                print(f"   💊 Drug #{drug_id + 1}: Effectiveness = {effectiveness_score:.3f}")
                
            except Exception as e:
                print(f"   ☠️  Drug #{drug_id + 1}: TOXIC - {str(e)[:50]}...")
                continue
        
        # Sort candidates by effectiveness
        candidates.sort(key=lambda x: x['effectiveness_score'], reverse=True)
        
        # Store R&D results
        self.phase1_results = candidates
        
        # Display R&D results
        print(f"\n🏆 PHASE 1 R&D RESULTS:")
        print("="*60)
        print("Top Drug Candidates for Clinical Trials:")
        
        for i, drug in enumerate(candidates[:5]):  # Show top 5
            print(f"{i+1}. Drug #{drug['drug_id']} (Seed {drug['seed']})")
            print(f"   Effectiveness: {drug['effectiveness_score']:.3f}")
            print(f"   Agreement: {drug.get('agreement', 'N/A')}")
            print(f"   Both Models: {'✅' if drug['dmd_available'] and drug['sindy_available'] else '❌'}")
        
        # Save results
        if save_results:
            filename = f"phase1_rd_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.phase1_results, f, indent=2)
            print(f"\n💾 R&D results saved to: {filename}")
        
        print(f"\n🎯 Ready for Phase 2: Clinical Trials with top candidates")
        return self.phase1_results
    
    # =========================================================================
    # PHASE 2: CLINICAL TRIALS - RIGOROUS VALIDATION
    # =========================================================================
    
    def phase2_clinical_trials(self, test_folder, n_top_candidates=3, prediction_horizon=5, save_results=True):  # ADD prediction_horizon
        """
        Phase 2: Clinical Trials - Rigorous testing of top candidates
        
        Like clinical trials, we test our best drug candidates on a larger,
        more diverse population (multiple tickers) with strict protocols
        """
        if not self.phase1_results:
            raise ValueError("Must run Phase 1 R&D first!")
        
        print("🧪 PHASE 2: CLINICAL TRIALS - RIGOROUS VALIDATION")
        print("="*80)
        print(f"🏥 Testing top {n_top_candidates} drug candidates")
        print(f"📅 Prediction Horizon: {prediction_horizon} days")  # NEW
        print(f"👥 Patient population: All tickers in {test_folder}")
        print("="*80)
        
        # Get all test files (patient population)
        test_files = list(Path(test_folder).glob("*.csv"))
        if not test_files:
            raise ValueError(f"No CSV files found in {test_folder}")
        
        print(f"👥 Patient Population: {len(test_files)} tickers")
        
        # Test top candidates
        top_candidates = self.phase1_results[:n_top_candidates]
        clinical_results = []
        
        for trial_num, drug in enumerate(top_candidates):
            print(f"\n🏥 CLINICAL TRIAL #{trial_num + 1}: Drug #{drug['drug_id']}")
            print(f"   Seed: {drug['seed']}")
            print(f"   R&D Effectiveness: {drug['effectiveness_score']:.3f}")
            
            # Set seed for reproducible clinical trial
            np.random.seed(drug['seed'])
            
            # Test on full patient population
            trial_results = {
                'drug_id': drug['drug_id'],
                'seed': drug['seed'],
                'rd_effectiveness': drug['effectiveness_score'],
                'clinical_results': [],
                'success_rate': 0,
                'side_effects': 0,
                'total_patients': len(test_files)
            }
            
            successful_treatments = 0
            side_effects = 0
            
            for patient_file in test_files:
                try:
                    # Treat this patient
                    analyzer = IndividualMethodsAnalyzer(target_metric='log_price_ratio_sma20', prediction_horizon=prediction_horizon)  # CHANGED
                    insight = analyzer.analyze_single_file(str(patient_file))
                    
                    # Evaluate treatment outcome
                    effectiveness = self._evaluate_drug_effectiveness(insight)
                    
                    if effectiveness > 0.5:  # Treatment success threshold
                        successful_treatments += 1
                    
                    # Record patient outcome
                    trial_results['clinical_results'].append({
                        'patient': patient_file.stem,
                        'effectiveness': effectiveness,
                        'dmd_success': insight.dmd_prediction is not None,
                        'sindy_success': insight.sindy_prediction is not None
                    })
                    
                except Exception as e:
                    # Side effect (treatment failure)
                    side_effects += 1
                    trial_results['clinical_results'].append({
                        'patient': patient_file.stem,
                        'effectiveness': 0,
                        'side_effect': str(e)[:50]
                    })
            
            # Calculate clinical trial outcomes
            trial_results['success_rate'] = successful_treatments / len(test_files)
            trial_results['side_effect_rate'] = side_effects / len(test_files)
            trial_results['trial_date'] = datetime.now().isoformat()
            
            clinical_results.append(trial_results)
            
            print(f"   ✅ Success Rate: {trial_results['success_rate']:.1%}")
            print(f"   ⚠️  Side Effects: {trial_results['side_effect_rate']:.1%}")
        
        # Store clinical trial results
        self.phase2_results = clinical_results
        
        # Determine which drugs pass clinical trials
        print(f"\n🏥 CLINICAL TRIAL RESULTS:")
        print("="*60)
        
        approved_drugs = []
        for trial in clinical_results:
            success_rate = trial['success_rate']
            side_effect_rate = trial['side_effect_rate']
            
            # FDA approval criteria
            if success_rate >= 0.6 and side_effect_rate <= 0.2:
                status = "✅ FDA APPROVED"
                approved_drugs.append(trial)
            elif success_rate >= 0.5 and side_effect_rate <= 0.3:
                status = "⚡ CONDITIONAL APPROVAL"
                approved_drugs.append(trial)
            else:
                status = "❌ REJECTED"
            
            print(f"Drug #{trial['drug_id']}: {status}")
            print(f"   Success: {success_rate:.1%} | Side Effects: {side_effect_rate:.1%}")
        
        # Save clinical trial results
        if save_results:
            filename = f"phase2_clinical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.phase2_results, f, indent=2)
            print(f"\n💾 Clinical trial results saved to: {filename}")
        
        print(f"\n🎯 {len(approved_drugs)} drugs approved for production")
        return self.phase2_results
    
    # =========================================================================
    # PHASE 3: PRODUCTION - DETERMINISTIC MANUFACTURING
    # =========================================================================
    
    def phase3_production_setup(self, save_models=True):
        """
        Phase 3: Production Setup - Create deterministic manufacturing process
        
        Like pharmaceutical manufacturing, we create exact, reproducible
        formulations for the drugs that passed clinical trials
        """
        if not self.phase2_results:
            raise ValueError("Must run Phase 2 Clinical Trials first!")
        
        print("🏭 PHASE 3: PRODUCTION SETUP - DETERMINISTIC MANUFACTURING")
        print("="*80)
        
        # Identify approved drugs
        approved_drugs = []
        for trial in self.phase2_results:
            success_rate = trial['success_rate']
            side_effect_rate = trial['side_effect_rate']
            
            if success_rate >= 0.5 and side_effect_rate <= 0.3:
                approved_drugs.append(trial)
        
        if not approved_drugs:
            print("❌ No drugs passed clinical trials!")
            return {}
        
        print(f"🏭 Setting up manufacturing for {len(approved_drugs)} approved drugs")
        
        # Create production models
        production_models = {}
        
        for drug in approved_drugs:
            drug_id = drug['drug_id']
            seed = drug['seed']
            
            print(f"\n🏭 Manufacturing Drug #{drug_id} (Seed {seed})")
            
            # Create deterministic production model
            production_models[f"drug_{drug_id}"] = {
                'seed': seed,
                'success_rate': drug['success_rate'],
                'side_effect_rate': drug['side_effect_rate'],
                'manufacturing_date': datetime.now().isoformat(),
                'quality_control': 'PASSED',
                'batch_number': f"BATCH_{drug_id}_{datetime.now().strftime('%Y%m%d')}"
            }
            
            print(f"   📦 Batch: {production_models[f'drug_{drug_id}']['batch_number']}")
            print(f"   ✅ Quality Control: PASSED")
        
        self.production_models = production_models
        
        # Save production models
        if save_models:
            filename = f"production_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.production_models, f, indent=2)
            print(f"\n💾 Production models saved to: {filename}")
        
        print(f"\n🎯 Production ready! Use run_production_analysis() for trading")
        return self.production_models
    
    def run_production_analysis(self, folder_path, prediction_horizon=5, export_csv=True):  # ADD prediction_horizon
        """
        Phase 3: Production Run - Deterministic trading analysis
        
        Like dispensing approved drugs, this gives consistent, reproducible
        results using the exact formulations that passed clinical trials
        """
        if not self.production_models:
            raise ValueError("Must run Phase 3 Production Setup first!")
        
        print("🏭 PRODUCTION RUN - DISPENSING APPROVED DRUGS")
        print("="*80)
        print(f"📁 Treating patients in: {folder_path}")
        print(f"📅 Prediction Horizon: {prediction_horizon} days")  # NEW
        
        # Get best approved drug
        best_drug = max(self.production_models.items(), 
                       key=lambda x: x[1]['success_rate'])
        
        drug_name, drug_info = best_drug
        print(f"💊 Using: {drug_name.upper()}")
        print(f"📦 Batch: {drug_info['batch_number']}")
        print(f"✅ Success Rate: {drug_info['success_rate']:.1%}")
        print("="*80)
        
        # Set deterministic seed for reproducible production
        np.random.seed(drug_info['seed'])
        
        # Run analysis with approved drug formulation
        analyzer = IndividualMethodsAnalyzer(target_metric='log_price_ratio_sma20', prediction_horizon=prediction_horizon)  # CHANGED
        insights = analyzer.analyze_folder(folder_path)
        
        if not insights:
            print("❌ No insights generated")
            return []
        
        # Enhanced production output
        print(f"\n💊 TREATMENT RESULTS - {drug_name.upper()}")
        print("="*150)
        
        # Enhanced table header
        print(f"{'Ticker':<12} {'Price':<10} {'DMD Treatment':<35} {'SINDy Treatment':<35} {'Outcome':<15}")
        print(f"{'':12} {'':10} {'(Signal-Action | Change | Conf)':<35} {'(Signal-Action | Change | Conf)':<35} {'':15}")
        print("-" * 150)
        
        successful_treatments = 0
        
        for insight in insights:
            ticker = insight.ticker
            price = f"${insight.current_price:.2f}" if insight.current_price else "N/A"
            
            # DMD treatment result
            if insight.dmd_prediction:
                dmd_signal = insight.dmd_prediction['signal'][:4]
                dmd_action = insight.dmd_prediction['action']
                dmd_change = insight.dmd_prediction['projected_change']
                dmd_conf = insight.dmd_prediction['confidence'][:1]
                dmd_summary = f"{dmd_signal}-{dmd_action} | {dmd_change} | {dmd_conf}"
            else:
                dmd_summary = "TREATMENT FAILED"
            
            # SINDy treatment result
            if insight.sindy_prediction:
                sindy_signal = insight.sindy_prediction['signal'][:4]
                sindy_action = insight.sindy_prediction['action']
                sindy_change = insight.sindy_prediction['projected_change']
                sindy_conf = insight.sindy_prediction['confidence'][:1]
                sindy_summary = f"{sindy_signal}-{sindy_action} | {sindy_change} | {sindy_conf}"
            else:
                sindy_summary = "TREATMENT FAILED"
            
            # Treatment outcome
            if insight.dmd_prediction and insight.sindy_prediction:
                dmd_action = insight.dmd_prediction['action']
                sindy_action = insight.sindy_prediction['action']
                
                if dmd_action == sindy_action == 'BUY':
                    outcome = "🟢 STRONG BUY"
                    successful_treatments += 1
                elif dmd_action == sindy_action == 'SELL':
                    outcome = "🔴 STRONG SELL"
                    successful_treatments += 1
                elif dmd_action == sindy_action:
                    outcome = "➡️ AGREE"
                    successful_treatments += 1
                else:
                    outcome = "⚠️ MIXED"
            else:
                outcome = "❌ FAILED"
            
            print(f"{ticker:<12} {price:<10} {dmd_summary:<35} {sindy_summary:<35} {outcome:<15}")
        
        # Production quality metrics
        treatment_success_rate = successful_treatments / len(insights)
        print(f"\n📊 PRODUCTION QUALITY METRICS:")
        print(f"   Patients Treated: {len(insights)}")
        print(f"   Successful Treatments: {successful_treatments}")
        print(f"   Success Rate: {treatment_success_rate:.1%}")
        print(f"   Expected Success Rate: {drug_info['success_rate']:.1%}")
        print(f"   Quality Control: {'✅ PASSED' if treatment_success_rate >= drug_info['success_rate'] * 0.8 else '⚠️ REVIEW NEEDED'}")
        
        # Export production results
        if export_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = Path(folder_path).name
            filename = f"production_results_{drug_name}_{folder_name}_{timestamp}.csv"
            
            # Add production metadata to insights
            for insight in insights:
                insight.drug_used = drug_name
                insight.batch_number = drug_info['batch_number']
                insight.production_date = timestamp
            
            export_individual_insights_to_csv(insights, filename)
            print(f"\n💾 Production results exported: {filename}")
        
        return insights
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _evaluate_drug_effectiveness(self, insight):
        """Evaluate how effective a drug candidate is"""
        score = 0
        
        # Both models working adds effectiveness
        if insight.dmd_prediction:
            score += 0.3
        if insight.sindy_prediction:
            score += 0.3
        
        # High confidence adds effectiveness
        if insight.dmd_prediction and insight.dmd_prediction['confidence'] == 'HIGH':
            score += 0.1
        if insight.sindy_prediction and insight.sindy_prediction['confidence'] == 'HIGH':
            score += 0.1
        
        # Agreement between models adds effectiveness
        if insight.dmd_prediction and insight.sindy_prediction:
            if insight.dmd_prediction['action'] == insight.sindy_prediction['action']:
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _check_agreement(self, insight):
        """Check agreement between DMD and SINDy"""
        if not (insight.dmd_prediction and insight.sindy_prediction):
            return "N/A"
        
        dmd_action = insight.dmd_prediction['action']
        sindy_action = insight.sindy_prediction['action']
        
        if dmd_action == sindy_action:
            return "AGREE"
        else:
            return "CONFLICT"


# =============================================================================
# EASY-TO-USE WRAPPER FOR YOUR EXISTING WORKFLOW
# =============================================================================

def run_drug_discovery_workflow(rd_folder, clinical_folder, production_folder, 
                               n_candidates=10, n_top=3, prediction_horizon=5):  # ADD prediction_horizon
    """
    Complete 3-phase drug discovery workflow
    
    Args:
        rd_folder: Folder for R&D testing (small sample)
        clinical_folder: Folder for clinical trials (larger sample)  
        production_folder: Folder for final production run
    """
    
    print("🚀 STARTING COMPLETE DRUG DISCOVERY WORKFLOW")
    print("="*80)
    print(f"📅 Prediction Horizon: {prediction_horizon} days")  # NEW
    
    # Initialize the system
    discovery = TradingDrugDiscovery(base_seed=42)
    
    # Phase 1: R&D
    print("\n🧬 PHASE 1: R&D - DRUG DISCOVERY")
    discovery.phase1_rd_exploration(rd_folder, n_drug_candidates=n_candidates, prediction_horizon=prediction_horizon)  # CHANGED
    
    # Phase 2: Clinical Trials  
    print("\n🧪 PHASE 2: CLINICAL TRIALS")
    discovery.phase2_clinical_trials(clinical_folder, n_top_candidates=n_top, prediction_horizon=prediction_horizon)  # CHANGED
    
    # Phase 3: Production Setup
    print("\n🏭 PHASE 3: PRODUCTION SETUP")
    discovery.phase3_production_setup()
    
    # Phase 3: Production Run
    print("\n💊 PHASE 3: PRODUCTION RUN")
    results = discovery.run_production_analysis(production_folder, prediction_horizon=prediction_horizon)  # CHANGED
    
    return discovery, results


# =============================================================================
# SIMPLE REPLACEMENT FOR YOUR CURRENT run_enhanced_analysis()
# =============================================================================

def run_production_enhanced_analysis(folder_path, discovery_system=None, prediction_horizon=5, export_csv=True):  # ADD prediction_horizon

    """
    Drop-in replacement for your current run_enhanced_analysis()
    
    If discovery_system is provided, uses approved production models
    Otherwise, uses a default deterministic approach
    """
    
    if discovery_system and discovery_system.production_models:
        # Use approved production models
        return discovery_system.run_production_analysis(folder_path, prediction_horizon, export_csv)  # CHANGED
    else:
        # Fallback: deterministic version of your current system
        print("🔒 DETERMINISTIC ANALYSIS (No drug discovery system)")
        print("="*60)
        print(f"📅 Prediction Horizon: {prediction_horizon} days")  # NEW
        
        # Set fixed seed for reproducible results
        np.random.seed(42)
        
        # Run your normal analysis
        analyzer = IndividualMethodsAnalyzer(target_metric='log_price_ratio_sma20', prediction_horizon=prediction_horizon)  # CHANGED
        insights = analyzer.analyze_folder(folder_path)
        
        # Your existing enhanced display code here...
        # (Same as your current run_enhanced_analysis function)
        
        return insights