import numpy as np

class SimilarityMetrics:
    """Classe com diferentes métricas de similaridade"""
    
    @staticmethod
    def cosine_similarity(v1, v2):
        """Calcula similaridade do cosseno entre dois vetores"""
        v1 = np.array(v1)
        v2 = np.array(v2)
        numerador = np.dot(v1, v2)
        denominador = np.linalg.norm(v1) * np.linalg.norm(v2)
        return numerador / denominador if denominador != 0 else 0.0
    
    @staticmethod
    def jaccard_similarity(v1: set, v2: set):
        """Calcula similaridade de Jaccard entre dois conjuntos"""
        intersection = v1.intersection(v2)
        union = v1.union(v2)
        return len(intersection) / len(union) if len(union) > 0 else 0
    
    
    @staticmethod
    def find_similar_documents(doc_original, df_vetores, n_similares=5):
        """
        Encontra os documentos mais similares
        
        Args:
            doc_original: Documento de referência
            df_vetores: DataFrame com todos os vetores
            n_similares: Número de documentos similares a retornar
            
        Returns:
            Lista de dicionários com informações dos documentos similares
        """
        similaridades = []
        
        for idx, row in df_vetores.iloc[1:].iterrows():
                
            sim_cos = SimilarityMetrics.cosine_similarity(doc_original, row)
            prod_esc = np.dot(doc_original, row)
            
            similaridades.append({
                'id': idx,
                'similaridade_cosseno': sim_cos,
                'produto_escalar': prod_esc
            })
            
        similaridades.sort(key=lambda x: x['similaridade_cosseno'], reverse=True)
        
        return similaridades[:n_similares] 