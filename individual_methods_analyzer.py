# Enhanced table display function
def run_enhanced_analysis(folder_path, target_metric='log_price_ratio_sma20', prediction_horizon=5, export_csv=True):  # ADD prediction_horizon
    """Run analysis with enhanced table showing change % and confidence"""
    
    print("🚀 ENHANCED BATCH ANALYSIS")
    print("="*120)
    print(f"📁 Folder: {folder_path}")
    print(f"🎯 Target: {target_metric}")
    print(f"📅 Prediction Horizon: {prediction_horizon} days")  # NEW
    print("="*120)
    
    # Initialize analyzer
    analyzer = IndividualMethodsAnalyzer(target_metric=target_metric, prediction_horizon=prediction_horizon)  # CHANGED
    
    # Analyze the folder
    insights = analyzer.analyze_folder(folder_path)
    
    if not insights:
        print("❌ No insights generated")
        return []
    
    # Display enhanced summary
    print(f"\n📊 ENHANCED TRADING INSIGHTS")
    print("="*150)
    
    # Enhanced table header
    print(f"{'Ticker':<12} {'Price':<10} {'DMD Prediction':<35} {'SINDy Prediction':<35} {'Agreement':<15}")
    print(f"{'':12} {'':10} {'(Signal-Action | Change | Conf)':<35} {'(Signal-Action | Change | Conf)':<35} {'':15}")
    print("-" * 150)
    
    for insight in insights:
        ticker = insight.ticker
        price = f"${insight.current_price:.2f}" if insight.current_price else "N/A"
        
        # Enhanced DMD summary
        if insight.dmd_prediction:
            dmd_signal = insight.dmd_prediction['signal'][:4]  # BULL, BEAR, NEUT
            dmd_action = insight.dmd_prediction['action']
            dmd_change = insight.dmd_prediction['projected_change']  # Already formatted as "+2.3%"
            dmd_conf = insight.dmd_prediction['confidence'][:1]  # H, M, L
            dmd_summary = f"{dmd_signal}-{dmd_action} | {dmd_change} | {dmd_conf}"
        else:
            dmd_summary = "FAILED"
        
        # Enhanced SINDy summary
        if insight.sindy_prediction:
            sindy_signal = insight.sindy_prediction['signal'][:4]  # BULL, BEAR, NEUT
            sindy_action = insight.sindy_prediction['action']
            sindy_change = insight.sindy_prediction['projected_change']  # Already formatted as "+2.3%"
            sindy_conf = insight.sindy_prediction['confidence'][:1]  # H, M, L
            sindy_summary = f"{sindy_signal}-{sindy_action} | {sindy_change} | {sindy_conf}"
        else:
            sindy_summary = "FAILED"
        
        # Agreement analysis
        if insight.dmd_prediction and insight.sindy_prediction:
            dmd_action = insight.dmd_prediction['action']
            sindy_action = insight.sindy_prediction['action']
            
            if dmd_action == sindy_action == 'BUY':
                agreement = "🟢 STRONG BUY"
            elif dmd_action == sindy_action == 'SELL':
                agreement = "🔴 STRONG SELL"
            elif dmd_action == sindy_action:
                agreement = "➡️ AGREE"
            else:
                agreement = "⚠️ CONFLICT"
        else:
            agreement = "N/A"
        
        print(f"{ticker:<12} {price:<10} {dmd_summary:<35} {sindy_summary:<35} {agreement:<15}")
    
    # Enhanced summary statistics
    print(f"\n🎯 ENHANCED SUMMARY")
    print("="*80)
    
    # Confidence breakdown
    dmd_high_conf = [i for i in insights if i.dmd_prediction and i.dmd_prediction['confidence'] == 'HIGH']
    dmd_med_conf = [i for i in insights if i.dmd_prediction and i.dmd_prediction['confidence'] == 'MEDIUM']
    dmd_low_conf = [i for i in insights if i.dmd_prediction and i.dmd_prediction['confidence'] == 'LOW']
    
    sindy_high_conf = [i for i in insights if i.sindy_prediction and i.sindy_prediction['confidence'] == 'HIGH']
    sindy_med_conf = [i for i in insights if i.sindy_prediction and i.sindy_prediction['confidence'] == 'MEDIUM']
    sindy_low_conf = [i for i in insights if i.sindy_prediction and i.sindy_prediction['confidence'] == 'LOW']
    
    print(f"🔵 DMD Confidence: HIGH={len(dmd_high_conf)} | MED={len(dmd_med_conf)} | LOW={len(dmd_low_conf)}")
    print(f"🔴 SINDy Confidence: HIGH={len(sindy_high_conf)} | MED={len(sindy_med_conf)} | LOW={len(sindy_low_conf)}")
    
    # Action counts with confidence
    dmd_buy = [i for i in insights if i.dmd_prediction and i.dmd_prediction['action'] == 'BUY']
    dmd_sell = [i for i in insights if i.dmd_prediction and i.dmd_prediction['action'] == 'SELL']
    sindy_buy = [i for i in insights if i.sindy_prediction and i.sindy_prediction['action'] == 'BUY']
    sindy_sell = [i for i in insights if i.sindy_prediction and i.sindy_prediction['action'] == 'SELL']
    
    print(f"\n📈 ACTION COUNTS:")
    print(f"🔵 DMD: BUY={len(dmd_buy)} | SELL={len(dmd_sell)}")
    print(f"🔴 SINDy: BUY={len(sindy_buy)} | SELL={len(sindy_sell)}")
    
    # High-confidence actionable signals
    print(f"\n⭐ HIGH-CONFIDENCE ACTIONABLE SIGNALS:")
    
    dmd_high_buy = [i for i in dmd_high_conf if i.dmd_prediction['action'] == 'BUY']
    dmd_high_sell = [i for i in dmd_high_conf if i.dmd_prediction['action'] == 'SELL']
    sindy_high_buy = [i for i in sindy_high_conf if i.sindy_prediction['action'] == 'BUY']
    sindy_high_sell = [i for i in sindy_high_conf if i.sindy_prediction['action'] == 'SELL']
    
    # FIXED versions without backslashes in f-strings:
    if dmd_high_buy:
        buy_list = [f"{i.ticker}({i.dmd_prediction['projected_change']})" for i in dmd_high_buy]
        print(f"🔵 DMD HIGH-CONF BUY: {', '.join(buy_list)}")

    if dmd_high_sell:
        sell_list = [f"{i.ticker}({i.dmd_prediction['projected_change']})" for i in dmd_high_sell]
        print(f"🔵 DMD HIGH-CONF SELL: {', '.join(sell_list)}")

    if sindy_high_buy:
        buy_list = [f"{i.ticker}({i.sindy_prediction['projected_change']})" for i in sindy_high_buy]
        print(f"🔴 SINDy HIGH-CONF BUY: {', '.join(buy_list)}")

    if sindy_high_sell:
        sell_list = [f"{i.ticker}({i.sindy_prediction['projected_change']})" for i in sindy_high_sell]
        print(f"🔴 SINDy HIGH-CONF SELL: {', '.join(sell_list)}")
    
    if not any([dmd_high_buy, dmd_high_sell, sindy_high_buy, sindy_high_sell]):
        print("   No high-confidence actionable signals found")
    
    # Strong consensus with details
    consensus_buy = []
    consensus_sell = []
    
    for insight in insights:
        if insight.dmd_prediction and insight.sindy_prediction:
            dmd_action = insight.dmd_prediction['action']
            sindy_action = insight.sindy_prediction['action']
            
            if dmd_action == sindy_action == 'BUY':
                dmd_change = insight.dmd_prediction['projected_change']
                sindy_change = insight.sindy_prediction['projected_change']
                consensus_buy.append(f"{insight.ticker}(DMD:{dmd_change}/SINDy:{sindy_change})")
            elif dmd_action == sindy_action == 'SELL':
                dmd_change = insight.dmd_prediction['projected_change']
                sindy_change = insight.sindy_prediction['projected_change']
                consensus_sell.append(f"{insight.ticker}(DMD:{dmd_change}/SINDy:{sindy_change})")
    
    print(f"\n🤝 STRONG CONSENSUS (Both Methods Agree):")
    if consensus_buy:
        print(f"✅ CONSENSUS BUY: {', '.join(consensus_buy)}")
    if consensus_sell:
        print(f"❌ CONSENSUS SELL: {', '.join(consensus_sell)}")
    if not consensus_buy and not consensus_sell:
        print("   No strong buy/sell consensus found")
    
    # Export results
    if export_csv:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = folder_path.replace('/', '_').replace('\\', '_')
        filename = f"enhanced_analysis_{folder_name}_{timestamp}.csv"
        export_individual_insights_to_csv(insights, filename)
        print(f"\n💾 Enhanced results exported: {filename}")
    
    print(f"\n✅ Enhanced analysis complete! Processed {len(insights)} tickers.")
    return insights

# Quick usage function
def analyze_folder_enhanced(folder_path, prediction_horizon=5):  # ADD prediction_horizon
    """Quick function to run enhanced analysis"""
    return run_enhanced_analysis(folder_path, target_metric='log_price_ratio_sma20', prediction_horizon=prediction_horizon)  # CHANGED