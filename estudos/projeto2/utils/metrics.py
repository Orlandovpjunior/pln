import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ModelEvaluator:
    """Classe para avaliação de modelos de classificação"""
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calcula acurácia"""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true, y_pred):
        """Gera relatório de classificação detalhado"""
        return classification_report(y_true, y_pred)
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Gera matriz de confusão"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def print_evaluation_summary(y_true, y_pred, model_name="Modelo"):
        """Imprime resumo da avaliação"""
        print(f"\n=== AVALIAÇÃO: {model_name} ===")
        print(f"Acurácia: {ModelEvaluator.calculate_accuracy(y_true, y_pred):.4f}")
        print("\nRelatório de Classificação:")
        print(ModelEvaluator.get_classification_report(y_true, y_pred)) 