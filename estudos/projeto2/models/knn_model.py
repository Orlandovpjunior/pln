import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNNClassifier:
    """Classe para classificação usando K-Nearest Neighbors"""
    
    def __init__(self, k=5, metric='euclidean'):
        """
        Inicializa o classificador KNN
        
        Args:
            k: Número de vizinhos
            metric: Métrica de distância ('euclidean', 'manhattan', 'chebyshev')
        """
        self.k = k
        self.metric = metric
        self.model = None
        self.X_train = None
        self.y_train = None
    
    def train(self, X_train, y_train):
        """
        Treina o modelo KNN
        
        Args:
            X_train: Dados de treinamento vetorizados
            y_train: Labels de treinamento
        """
        self.X_train = X_train
        self.y_train = y_train
        
        self.model = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric)
        self.model.fit(X_train.toarray(), y_train)
    
    def predict(self, X_test):
        """
        Faz predições
        
        Args:
            X_test: Dados de teste vetorizados
            
        Returns:
            Predições do modelo
        """
        if self.model is None:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")
        
        return self.model.predict(X_test.toarray())
    
    def get_neighbors(self, X_sample):
        """
        Retorna os vizinhos mais próximos de uma amostra
        
        Args:
            X_sample: Amostra para encontrar vizinhos
            
        Returns:
            distances: Distâncias dos vizinhos
            indices: Índices dos vizinhos
        """
        if self.model is None:
            raise ValueError("Modelo deve ser treinado antes")
        
        return self.model.kneighbors([X_sample.toarray()[0]])
    
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo
        
        Args:
            X_test: Dados de teste
            y_test: Labels reais
            
        Returns:
            acuracia: Acurácia do modelo
            predicoes: Predições feitas
        """
        predicoes = self.predict(X_test)
        acuracia = np.mean(predicoes == y_test)
        
        return acuracia, predicoes
    
    def compare_metrics(self, X_train, X_test, y_train, y_test, metrics=None):
        """
        Compara diferentes métricas de distância
        
        Args:
            X_train, X_test, y_train, y_test: Dados de treino e teste
            metrics: Lista de métricas para comparar
            
        Returns:
            Dicionário com acurácias para cada métrica
        """
        if metrics is None:
            metrics = ['euclidean', 'manhattan', 'chebyshev']
        
        results = {}
        
        for metric in metrics:
            self.metric = metric
            self.train(X_train, y_train)
            acuracia, _ = self.evaluate(X_test, y_test)
            results[metric] = acuracia
        
        return results 