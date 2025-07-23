#!/usr/bin/env python3
"""
Script principal para classificação de texto com KNN
Replica os conceitos do Lab4 de forma organizada
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Importando nossos módulos
from data.data_loader import DataLoader
from models.vectorizer import TextVectorizer
from models.similarity import SimilarityMetrics
from models.knn_model import KNNClassifier
from utils.metrics import ModelEvaluator

def main():
    """Função principal que orquestra todo o processo"""
    
    print("🚀 INICIANDO CLASSIFICAÇÃO KNN - REPLICAÇÃO DO LAB4")
    print("=" * 60)
    
    # 1. CARREGAMENTO DE DADOS
    print("\n📊 1. CARREGANDO DATASET...")
    # Usando dataset local de teste
    data_loader = DataLoader(file_path="test_dataset.csv")
    df = data_loader.get_data()
    
    print(f"✅ Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    print(f"📋 Colunas: {df.columns.tolist()}")
    print("\n📄 Primeiras 5 linhas:")
    print(df.head())
    
    # Preparando dados
    textos = df['review']
    labels = df['sentiment']
    print(f"\n🎯 Classes únicas: {labels.unique()}")
    
    # 2. VETORIZAÇÃO
    print("\n🔤 2. VETORIZANDO TEXTOS...")
    
    # Pergunta ao usuário qual tipo de vetorização usar
    print("Tipos disponíveis: one_hot, bow, tfidf")
    tipo_vetorizacao = input("Escolha o tipo de vetorização (padrão: bow): ").strip().lower()
    
    if not tipo_vetorizacao:
        tipo_vetorizacao = 'bow'
    
    vectorizer = TextVectorizer(default_type=tipo_vetorizacao)
    
    print(f"✅ Usando vetorização: {tipo_vetorizacao.upper()}")
    
    # Vetorização com o tipo escolhido
    print(f"   📝 {tipo_vetorizacao.upper()}...")
    df_vetores, vetores, palavras = vectorizer.vectorize(textos, tipo_vetorizacao)
    print(f"   ✅ Vocabulário: {len(palavras)} palavras")
    
    # 3. ANÁLISE DE SIMILARIDADE
    print("\n🔍 3. ANÁLISE DE SIMILARIDADE...")
    doc_original = df_vetores.iloc[0]
    similares = SimilarityMetrics.find_similar_documents(doc_original, df_vetores, 3)
    
    print(f"📄 Documento original: {textos.iloc[0][:80]}...")
    print("\n👥 Documentos mais similares (One-Hot):")
    for i, sim in enumerate(similares):
        print(f"   {i+1}. ID {sim['id']} - Cosseno: {sim['similaridade_cosseno']:.4f}")
    
    # 4. CLASSIFICAÇÃO KNN
    print("\n🤖 4. CLASSIFICAÇÃO KNN...")
    
    # Separando dados
    X_train, X_test, y_train, y_test = train_test_split(
        vetores, labels, test_size=0.2, random_state=42
    )
    
    # Testando diferentes métricas
    knn = KNNClassifier(k=5)
    resultados = knn.compare_metrics(X_train, X_test, y_train, y_test)
    
    print("📊 Resultados por métrica de distância:")
    for metrica, acuracia in resultados.items():
        print(f"   🔸 {metrica.upper()}: {acuracia:.4f}")
    
    # 5. ANÁLISE DETALHADA
    print("\n🔬 5. ANÁLISE DETALHADA...")
    
    # Usando a melhor métrica
    melhor_metrica = max(resultados, key=resultados.get)
    print(f"🏆 Melhor métrica: {melhor_metrica.upper()}")
    
    knn.metric = melhor_metrica
    knn.train(X_train, y_train)
    acuracia, predicoes = knn.evaluate(X_test, y_test)
    
    # Avaliação detalhada
    ModelEvaluator.print_evaluation_summary(y_test, predicoes, f"KNN ({melhor_metrica})")
    
    # 6. EXEMPLO DE PREDIÇÃO
    print("\n🎯 6. EXEMPLO DE PREDIÇÃO...")
    exemplo_idx = 0
    X_exemplo = X_test[exemplo_idx]
    y_real = y_test.iloc[exemplo_idx]
    texto_exemplo = textos.iloc[y_test.index[exemplo_idx]]
    
    predicao = knn.predict(X_exemplo)
    distancias, indices = knn.get_neighbors(X_exemplo)
    
    print(f"📝 Texto: {texto_exemplo[:100]}...")
    print(f"✅ Classe real: {y_real}")
    print(f"📈 Classe prevista: {predicao[0]}")
    print(f"🎯 Acurácia: {'✅ Correto' if predicao[0] == y_real else '❌ Incorreto'}")
    
    print("\n👥 Vizinhos mais próximos:")
    for i, (dist, idx) in enumerate(zip(distancias[0], indices[0])):
        vizinho_texto = textos.iloc[y_train.index[idx]]
        vizinho_classe = y_train.iloc[idx]
        print(f"   {i+1}. \"{vizinho_texto[:80]}...\" → classe: {vizinho_classe}")
    
    print("\n🎉 ANÁLISE COMPLETA CONCLUÍDA!")
    print("=" * 60)

if __name__ == "__main__":
    main() 