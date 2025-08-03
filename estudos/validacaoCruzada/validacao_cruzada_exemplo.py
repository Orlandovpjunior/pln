"""
Exemplo de Validação Cruzada (Cross-Validation)

Este script demonstra como implementar validação cruzada para avaliar 
a performance de um modelo de classificação.

O que é Validação Cruzada?
- É uma técnica para avaliar a generalização de um modelo
- Divide os dados em k folds (partes)
- Treina o modelo k vezes, cada vez usando k-1 folds para treino e 1 fold para teste
- Calcula a média das métricas de performance

Vantagens:
- Usa todos os dados para treino e teste
- Reduz overfitting
- Fornece estimativa mais confiável da performance real
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configurar para mostrar gráficos em português
plt.rcParams['font.size'] = 12

def carregar_dados():
    """
    Carrega o dataset Iris como exemplo
    """
    print("=== CARREGANDO DADOS ===")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"Forma dos dados: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Distribuição das classes: {np.bincount(y)}")
    print()
    
    return X, y

def validacao_cruzada_simples(X, y, k=5):
    """
    Implementação manual da validação cruzada para entender o conceito
    """
    print("=== VALIDAÇÃO CRUZADA MANUAL ===")
    print(f"Dividindo dados em {k} folds...")
    
    # Criar índices para os folds
    n_samples = len(X)
    fold_size = n_samples // k
    
    # Lista para armazenar as acurácias de cada fold
    acuracias = []
    
    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")
        
        # Definir índices de teste para este fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k-1 else n_samples
        
        # Criar máscaras para treino e teste
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[start_idx:end_idx] = True
        train_mask = ~test_mask
        
        # Dividir dados
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"Tamanho do treino: {len(X_train)}")
        print(f"Tamanho do teste: {len(X_test)}")
        
        # Treinar modelo
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = modelo.predict(X_test)
        
        # Calcular acurácia
        acuracia = accuracy_score(y_test, y_pred)
        acuracias.append(acuracia)
        
        print(f"Acurácia do fold {i+1}: {acuracia:.4f}")
    
    # Calcular métricas finais
    acuracia_media = np.mean(acuracias)
    acuracia_std = np.std(acuracias)
    
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"Acurácia média: {acuracia_media:.4f} (+/- {acuracia_std:.4f})")
    print(f"Desvio padrão: {acuracia_std:.4f}")
    
    return acuracias

def validacao_cruzada_sklearn(X, y, k=5):
    """
    Usando a implementação do scikit-learn (mais eficiente)
    """
    print("\n=== VALIDAÇÃO CRUZADA COM SCIKIT-LEARN ===")
    
    # Criar modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Validação cruzada simples
    print("1. Validação Cruzada Simples:")
    scores_simples = cross_val_score(modelo, X, y, cv=k, scoring='accuracy')
    print(f"Acurácias: {scores_simples}")
    print(f"Média: {scores_simples.mean():.4f} (+/- {scores_simples.std() * 2:.4f})")
    
    # Validação cruzada estratificada (mantém proporção das classes)
    print("\n2. Validação Cruzada Estratificada:")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores_estratificada = cross_val_score(modelo, X, y, cv=skf, scoring='accuracy')
    print(f"Acurácias: {scores_estratificada}")
    print(f"Média: {scores_estratificada.mean():.4f} (+/- {scores_estratificada.std() * 2:.4f})")
    
    return scores_simples, scores_estratificada

def comparar_metricas(X, y, k=5):
    """
    Compara diferentes métricas usando validação cruzada
    """
    print("\n=== COMPARANDO DIFERENTES MÉTRICAS ===")
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Lista de métricas para testar
    metricas = {
        'accuracy': 'Acurácia',
        'precision_macro': 'Precisão (macro)',
        'recall_macro': 'Recall (macro)',
        'f1_macro': 'F1-Score (macro)'
    }
    
    resultados = {}
    
    for metrica, nome in metricas.items():
        scores = cross_val_score(modelo, X, y, cv=skf, scoring=metrica)
        resultados[nome] = scores
        print(f"{nome}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return resultados

def visualizar_resultados(acuracias_manual, scores_simples, scores_estratificada, resultados_metricas):
    """
    Cria visualizações dos resultados da validação cruzada
    """
    print("\n=== CRIANDO VISUALIZAÇÕES ===")
    
    # Configurar figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Resultados da Validação Cruzada', fontsize=16, fontweight='bold')
    
    # 1. Comparação entre métodos
    ax1 = axes[0, 0]
    metodos = ['Manual', 'Sklearn Simples', 'Sklearn Estratificada']
    medias = [
        np.mean(acuracias_manual),
        scores_simples.mean(),
        scores_estratificada.mean()
    ]
    stds = [
        np.std(acuracias_manual),
        scores_simples.std(),
        scores_estratificada.std()
    ]
    
    bars = ax1.bar(metodos, medias, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_title('Comparação de Métodos de Validação Cruzada')
    ax1.set_ylabel('Acurácia Média')
    ax1.set_ylim(0.9, 1.0)
    
    # Adicionar valores nas barras
    for bar, media in zip(bars, medias):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{media:.4f}', ha='center', va='bottom')
    
    # 2. Distribuição dos scores
    ax2 = axes[0, 1]
    dados_box = [acuracias_manual, scores_simples, scores_estratificada]
    box_plot = ax2.boxplot(dados_box, labels=metodos, patch_artist=True)
    ax2.set_title('Distribuição dos Scores por Fold')
    ax2.set_ylabel('Acurácia')
    
    # Cores diferentes para cada método
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # 3. Comparação de métricas
    ax3 = axes[1, 0]
    metricas_nomes = list(resultados_metricas.keys())
    metricas_medias = [np.mean(scores) for scores in resultados_metricas.values()]
    metricas_stds = [np.std(scores) for scores in resultados_metricas.values()]
    
    bars = ax3.bar(metricas_nomes, metricas_medias, yerr=metricas_stds, capsize=5, alpha=0.7)
    ax3.set_title('Comparação de Diferentes Métricas')
    ax3.set_ylabel('Score Médio')
    ax3.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar, media in zip(bars, metricas_medias):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{media:.3f}', ha='center', va='bottom')
    
    # 4. Evolução dos scores por fold
    ax4 = axes[1, 1]
    folds = range(1, len(scores_estratificada) + 1)
    ax4.plot(folds, scores_estratificada, 'o-', linewidth=2, markersize=8, label='Estratificada')
    ax4.plot(folds, scores_simples, 's-', linewidth=2, markersize=8, label='Simples')
    ax4.axhline(y=np.mean(scores_estratificada), color='red', linestyle='--', alpha=0.7, label='Média')
    ax4.set_title('Evolução dos Scores por Fold')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Acurácia')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_validacao_cruzada.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráficos salvos em 'resultados_validacao_cruzada.png'")

def explicar_conceitos():
    """
    Explica os conceitos importantes da validação cruzada
    """
    print("\n" + "="*60)
    print("EXPLICAÇÃO DOS CONCEITOS")
    print("="*60)
    
    print("\n1. O QUE É VALIDAÇÃO CRUZADA?")
    print("- Técnica para avaliar a performance de um modelo")
    print("- Divide os dados em k partes (folds)")
    print("- Treina o modelo k vezes, cada vez usando k-1 folds para treino")
    print("- Usa 1 fold para teste em cada iteração")
    print("- Calcula a média das métricas de performance")
    
    print("\n2. VANTAGENS:")
    print("- Usa todos os dados para treino e teste")
    print("- Reduz overfitting (superajuste)")
    print("- Fornece estimativa mais confiável da performance real")
    print("- Permite avaliar a variabilidade do modelo")
    
    print("\n3. TIPOS DE VALIDAÇÃO CRUZADA:")
    print("- Simples: divide os dados sequencialmente")
    print("- Estratificada: mantém a proporção das classes em cada fold")
    print("- Shuffle: embaralha os dados antes de dividir")
    
    print("\n4. INTERPRETAÇÃO DOS RESULTADOS:")
    print("- Média alta: modelo performa bem")
    print("- Baixo desvio padrão: modelo é estável")
    print("- Alto desvio padrão: modelo é instável")
    print("- Diferença grande entre treino e teste: overfitting")
    
    print("\n5. QUANDO USAR:")
    print("- Datasets pequenos ou médios")
    print("- Quando queremos estimativa confiável da performance")
    print("- Para comparar diferentes modelos")
    print("- Para otimizar hiperparâmetros")

def main():
    """
    Função principal que executa todo o exemplo
    """
    print("VALIDAÇÃO CRUZADA - EXEMPLO DIDÁTICO")
    print("="*50)
    
    # Carregar dados
    X, y = carregar_dados()
    
    # Executar validação cruzada manual
    acuracias_manual = validacao_cruzada_simples(X, y, k=5)
    
    # Executar validação cruzada com sklearn
    scores_simples, scores_estratificada = validacao_cruzada_sklearn(X, y, k=5)
    
    # Comparar diferentes métricas
    resultados_metricas = comparar_metricas(X, y, k=5)
    
    # Criar visualizações
    visualizar_resultados(acuracias_manual, scores_simples, scores_estratificada, resultados_metricas)
    
    # Explicar conceitos
    explicar_conceitos()
    
    print("\n" + "="*50)
    print("EXEMPLO CONCLUÍDO!")
    print("Agora você entende como implementar e interpretar validação cruzada!")
    print("="*50)

if __name__ == "__main__":
    main() 