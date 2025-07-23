import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class TextVectorizer:
    """Classe para vetorização de textos usando diferentes técnicas"""
    
    def __init__(self, default_type='bow'):
        
        self.default_type = default_type
        self.vectorizers = {
            'one_hot': CountVectorizer(binary=True),
            'bow': CountVectorizer(),
            'tfidf': TfidfVectorizer()
        }
        
        if default_type not in self.vectorizers:
            raise ValueError(f"Tipo padrão '{default_type}' inválido. Use: {list(self.vectorizers.keys())}")
    
    def get_available_types(self):
        """Retorna os tipos disponíveis"""
        return list(self.vectorizers.keys())
    
    def vectorize(self, textos, tipo='bow'):
        """
        Vetoriza textos usando diferentes técnicas
        
        Args:
            textos: Lista de textos para vetorizar
            tipo: 'one_hot', 'bow' ou 'tfidf'
            
        Returns:
            df_vetores: DataFrame com vetores
            vetores: Matriz esparsa dos vetores
            palavras: Lista de palavras do vocabulário
        """
        if tipo not in self.vectorizers:
            raise ValueError("Tipo deve ser 'one_hot', 'bow' ou 'tfidf'")
        
        vectorizer = self.vectorizers[tipo]
        vetores = vectorizer.fit_transform(textos)
        palavras = vectorizer.get_feature_names_out()
        
        df_vetores = pd.DataFrame(vetores.toarray(), columns=palavras)
        
        return df_vetores, vetores, palavras
    
    def get_vocabulary_size(self, textos, tipo='bow'):
        """Retorna o tamanho do vocabulário"""
        _, _, palavras = self.vectorize(textos, tipo)
        return len(palavras) 