"""
Analyse des données SP500 avec corrélations et classification multi-classes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuration des warnings et affichage
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

def load_and_prepare_data(filepath='csv_data/consolidated_data/normalized_complete_data.csv'):
    """Charge et prépare les données avec calcul du return 7 jours"""
    print("=== Chargement des données ===")
    df = pd.read_csv(filepath, parse_dates=['Date']).sort_values('Date')
    
    # Calcul du return à 7 jours
    df['sp500_return_7d'] = (
        df['SP500_historical_data_Close'].shift(-7) - df['SP500_historical_data_Close']
    ) / df['SP500_historical_data_Close']
    
    # Suppression des valeurs manquantes
    initial_rows = len(df)
    df = df.dropna(subset=['sp500_return_7d'])
    print(f"Données chargées: {initial_rows} → {len(df)} lignes après nettoyage")
    
    return df

# =============================================================================
# 2. FONCTIONS DE CLASSIFICATION
# =============================================================================

def create_manual_target(df):
    """Crée une target multi-classes avec des seuils prédéfinis"""
    def label_change(x):
        if x <= -0.043:
            return 0  # Forte baisse
        elif x <= -0.009:
            return 1  # Baisse modérée
        elif x <= 0.017:
            return 2  # Stable
        elif x <= 0.041:
            return 3  # Hausse modérée
        else:
            return 4  # Forte hausse
    
    df['target_multi_manual'] = df['sp500_return_7d'].apply(label_change)
    return df

def create_optimal_target(df, k=5):
    """Crée une target multi-classes avec des seuils optimaux (quantiles équilibrés)"""
    # Calcul des seuils optimaux
    quantiles = np.linspace(0, 1, k + 1)
    thresholds = df['sp500_return_7d'].quantile(quantiles).values[1:-1]
    
    # Application des seuils
    def label_from_thresholds(x, thresholds):
        for i, t in enumerate(thresholds):
            if x <= t:
                return i
        return len(thresholds)
    
    df['target_multi_optimal'] = df['sp500_return_7d'].apply(
        lambda x: label_from_thresholds(x, thresholds)
    )
    
    print(f"\n=== Seuils optimaux calculés ===")
    print(f"Seuils: {thresholds}")
    
    return df, thresholds

# =============================================================================
# 3. ANALYSE STATISTIQUE
# =============================================================================

def analyze_returns_distribution(df):
    """Analyse la distribution des returns"""
    print('\n=== Analyse des returns SP500 (7 jours) ===')
    
    # Statistiques descriptives
    print(f"Moyenne: {df['sp500_return_7d'].mean():.4f}")
    print(f"Médiane: {df['sp500_return_7d'].median():.4f}")
    print(f"Écart-type: {df['sp500_return_7d'].std():.4f}")
    print(f"Min: {df['sp500_return_7d'].min():.4f}")
    print(f"Max: {df['sp500_return_7d'].max():.4f}")
    
    # Quantiles
    quantiles = df['sp500_return_7d'].quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    print('\n=== Quantiles ===')
    for q, value in quantiles.items():
        print(f"Q{q*100:02.0f}: {value:.4f}")

def analyze_class_distribution(df):
    """Analyse la distribution des classes pour les différentes targets"""
    print('\n=== Distribution des classes ===')
    
    if 'target_multi_manual' in df.columns:
        manual_dist = df['target_multi_manual'].value_counts().sort_index()
        print('\nTarget manuelle:')
        for class_id, count in manual_dist.items():
            print(f"  Classe {class_id}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'target_multi_optimal' in df.columns:
        optimal_dist = df['target_multi_optimal'].value_counts().sort_index()
        print('\nTarget optimale:')
        for class_id, count in optimal_dist.items():
            print(f"  Classe {class_id}: {count} ({count/len(df)*100:.1f}%)")

# =============================================================================
# 4. ANALYSE DES CORRÉLATIONS
# =============================================================================

def analyze_correlations(df, output_dir='8_analyze'):
    """Analyse des corrélations avec la target"""
    # Créer le dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Matrice de corrélation
    corr_matrix = df.corr(numeric_only=True)
    
    # Corrélations avec la target
    corr_with_target = corr_matrix['sp500_return_7d'].drop('sp500_return_7d').sort_values(key=abs, ascending=False)
    
    print('\n=== Top 20 corrélations avec sp500_return_7d ===')
    for feature, corr in corr_with_target.head(20).items():
        print(f"{feature}: {corr:.4f}")
    
    return corr_matrix, corr_with_target

def create_visualizations(corr_matrix, corr_with_target, output_dir='8_analyze'):
    """Crée les visualisations des corrélations"""
    
    # 1. Heatmap complète
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Masquer le triangle supérieur
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": 0.8})
    plt.title('Matrice de corrélation complète', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top corrélations avec target
    plt.figure(figsize=(12, 8))
    top_corr = corr_with_target.head(20).sort_values()
    colors = ['red' if x < 0 else 'green' for x in top_corr.values]
    top_corr.plot(kind='barh', color=colors, alpha=0.7)
    plt.title('Top 20 corrélations avec sp500_return_7d', fontsize=14, pad=15)
    plt.xlabel('Coefficient de corrélation', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_with_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution des returns
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(corr_with_target.dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(corr_with_target.mean(), color='red', linestyle='--', label=f'Moyenne: {corr_with_target.mean():.4f}')
    plt.title('Distribution des returns 7j SP500')
    plt.xlabel('Return')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(corr_with_target.dropna())
    plt.title('Box plot des returns 7j SP500')
    plt.ylabel('Return')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Visualisations sauvegardées dans {output_dir}/ ===")

# =============================================================================
# 5. FONCTION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale d'exécution de l'analyse"""
    # Chargement des données
    df = load_and_prepare_data()
    
    # Création des targets
    df = create_manual_target(df)
    df, optimal_thresholds = create_optimal_target(df, k=5)
    
    # Analyses statistiques
    analyze_returns_distribution(df)
    analyze_class_distribution(df)
    
    # Analyse des corrélations
    corr_matrix, corr_with_target = analyze_correlations(df)
    
    # Création des visualisations
    create_visualizations(corr_matrix, corr_with_target)
    
    print("\n=== Analyse terminée avec succès ===")
    
    return df, corr_matrix, corr_with_target, optimal_thresholds

# =============================================================================
# 6. EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    df, corr_matrix, corr_with_target, optimal_thresholds = main()