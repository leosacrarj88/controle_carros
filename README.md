# Agência de Carros — Streamlit (Community Cloud)

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Segredos (local)
Crie o arquivo `.streamlit/secrets.toml` (não commitar) com:

```toml
SUPABASE_URL = "https://SEU-PROJECT-REF.supabase.co"
SUPABASE_KEY = "SUA-ANON-KEY"
```

## Deploy no Streamlit Community Cloud
1. Suba este repositório no GitHub.
2. Acesse `share.streamlit.io` e clique em **Create app**.
3. Selecione o repositório/branch e o arquivo **app.py**.
4. Em **Advanced settings**, cole seus secrets (mesmo conteúdo do `secrets.toml`) e escolha a versão de Python.
5. Deploy.
