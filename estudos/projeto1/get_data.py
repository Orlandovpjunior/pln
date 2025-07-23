import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

class GetData:
    def __init__(self, url: str, encoding: str = 'utf-8'):
        self.url = url
        self.encoding = encoding
        self._session = requests.Session()

    def get_data(self) -> str:

        try:
            response = self._session.get(self.url, timeout=10)
            response.raise_for_status()
            response.encoding = self.encoding
            return response.text
        except RequestException as e:
            raise RequestException(f"Erro ao acessar a URL {self.url}: {str(e)}")

    def read_data(self, clean_html: bool = True) -> str:
        try:
            soup = BeautifulSoup(self.get_data(), 'html.parser')
            
            for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
                element.decompose()
            
            if clean_html:
                text = ' '.join(soup.stripped_strings)
                text = ' '.join(text.split())
                return text
            else:
                return soup.get_text()
                
        except Exception as e:
            raise Exception(f"Erro ao processar o HTML: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()
    
