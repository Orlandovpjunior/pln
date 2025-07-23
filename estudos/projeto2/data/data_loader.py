import pandas as pd

class DataLoader:
    """Classe para carregar datasets de diferentes fontes"""
    
    def __init__(self, file_path: str = None) -> None:
        self.file_path = file_path
        
    def get_data(self) -> pd.DataFrame:
        """Carrega dados do arquivo"""
        if not self.file_path:
            raise ValueError("file_path deve ser fornecido")
        
        try:
            df = pd.read_csv(self.file_path)
            return df
        except Exception as e:
            raise Exception(f'Erro ao carregar dataset de {self.file_path}: {str(e)}')