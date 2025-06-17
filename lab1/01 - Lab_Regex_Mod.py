import re
# Texto original do diário
diario = """
[DATA: 2024-11-03 | HORA: 02:17]

Na madrugada, finalizei o pentest no sistema legado da OmegaSec.
Consegui acesso via endpoint `/api/v1/users?token=21a8a8bbd3e1a7ad12b`.
O IP exposto era 10.0.0.142, mas logo percebi que ele redirecionava para 177.134.12.88:8080 via nginx.

Encontrei credenciais antigas armazenadas no HTML:
Email(s): roberta.souza@omegasec.com, roberta_souza123@sec.com
Senha: OmeGa$2021 (ridícula).
Senha alternativa: Rob_souza$2021 (ridícula).
O token da sessão era: Bearer 88ac32adbe10abc99122f.

Também encontrei um código aleatório. Porém é apenas de uma compra de uma roupa. O código é lojas2024$2025.

Usuário testava o cartão 1234 5678 9012 3456 para compras em staging.
Telefone(s) cadastrado(s):
(85) 99876-1234 | 859945-5464 | 85 98988-1479
CPF: 234.567.890-12
Nome completo: Roberta Souza.

OBS: Apaguei tudo do log público. Mas o backup do banco estava disponível em:
https://db-backup.omegasec.com/export/full.zip?access_token=abc321zzx

Mais tarde, revisei outra instância na AWS com IP 18.223.112.100 e usuário "admin@example.com".
Este email não deveria ser confundido com o link: https://example.com/admin@example.com/login

Por fim, notei que logs internos estavam expondo CPFs embutidos:
`user_cpf_998.877.665-44_data.log`

E pra variar, encontrei outro cartão, mas só salvei os últimos dígitos: 4321.

Agora vou dormir.
- Shadow_0x
"""
# 🧪 Tarefa 1: Crie uma função para mascarar todos os CPFs (inclusive embutidos)
def mascarar_cpf(texto):
    return re.sub(r'\d{3}\.\d{3}\.\d{3}-\d{2}', '<CPF>', texto)

# 🧪 Tarefa 2: Crie uma função para mascarar e-mails (deixe apenas o domínio visível)
def mascarar_email(texto):
    return re.sub(r'[^@\s]+(?=@)', '<EMAIL>', texto)

# 🧪 Tarefa 3: Crie uma função para mascarar números de telefone
def mascarar_telefone(texto):
    return re.sub(r'((\((\d{2})\))|(\d{2})?\s\d{4,5}-\d{4})|(\d{6}-\d{4})','<TELEFONE>',texto)

# 🧪 Tarefa 4: Crie uma função para ocultar IPs (ex: 192.168.0.1 --> <IP OCULTO>)
def ocultar_ips(texto):
    return re.sub(r'(\d{2}\.\d{3}\.\d{3}\.\d{3})|((\d{2}\.(\d{1})*)\.\d\.\d{3})|(\d{3}\.)+(\d{2}\.)+\d{2}:\d{4}','<IP OCULTO>',texto)

# 🧪 Tarefa 5: Crie uma função para remover tokens (Bearer e em URLs e querystrings)
def remover_tokens(texto):
    return re.sub(r'(?i)((?:Bearer\s)|(?:access_)?token=)([a-z0-9]+)', r'\1<TOKEN>', texto)

# 🧪 Tarefa 6: Crie uma função para anonimizar nomes próprios (ex: "Roberta Souza" --> [REDACTED])
def anonimizar_nomes(texto):
    return re.sub(r'\b[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)+\b', '[REDACTED]', texto)

# 🧪 Tarefa 7: Crie uma função para remover links inteiros por [LINK REMOVIDO]
def remover_links(texto):
    return re.sub(r'https?:\/\/[^\s]+', '[LINK REMOVIDO]', texto)

# � Tarefa 8: Crie uma função para mascarar cartões de crédito (mostrar apenas os 4 últimos)
def mascarar_cartao(texto):
    return re.sub(r'\d{4}\s\d{4}\s\d{4}', '<CARTAO>', texto)

# 🧪 Tarefa 8: Crie uma função para mascarar as senhas
def mascarar_senha(texto):
    return re.sub(r'OmeGa\$2021|Rob_souza\$2021', '<SENHA>', texto)

def processar_diario(texto):
    """Aplica todas as funções de mascaramento ao diário"""
    # A ordem de aplicação é importante!
    texto = mascarar_senha(texto)
    texto = anonimizar_nomes(texto)
    texto = remover_tokens(texto)
    texto = mascarar_cpf(texto)
    texto = mascarar_cartao(texto)
    texto = mascarar_telefone(texto)
    texto = ocultar_ips(texto)
    texto = mascarar_email(texto)
    texto = remover_links(texto)
    return texto

def main():
    """Função principal para execução do script"""
    print("=== Diário Original ===")
    print(diario)
    
    print("\n=== Diário Processado ===")  
    texto_processado = processar_diario(diario)
    print(texto_processado)

if __name__ == "__main__":
    main()