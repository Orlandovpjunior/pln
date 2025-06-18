from get_data import GetData
from preprocess import Preprocess

def processar_site(url: str) -> list:

    try:
        with GetData(url) as getter:
            texto_bruto = getter.read_data()
            print("Texto extraído do site:")
            print("-" * 50)
            print(texto_bruto[:200] + "...")
            print("-" * 50)
        
        preprocessor = Preprocess(texto_bruto)
        palavras_processadas = preprocessor.process()
        
        return palavras_processadas
        
    except Exception as e:
        print(f"Erro ao processar o site: {str(e)}")
        return []

if __name__ == "__main__":
    # Exemplo de uso
    url = input("Digite a URL do site: ")
    
    print("Iniciando processamento do site...")
    resultado = processar_site(url)
    
    print("\nPalavras processadas:")
    print("-" * 50)
    print(resultado[:20])
    print("-" * 50)
    
    print(f"\nTotal de palavras processadas: {len(resultado)}") 