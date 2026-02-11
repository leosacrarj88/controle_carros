# CARRO_YAGO.py
# Streamlit + Supabase API (PostgREST) — Agência de Carros
#
# Requisitos:
#   pip install streamlit pandas openpyxl supabase
#
# Rodar no VS Code (Play/Run): python CARRO_YAGO.py  -> auto-boot no streamlit
# Rodar direto: streamlit run CARRO_YAGO.py

# ==========================================================
# BOOT AUTOMÁTICO STREAMLIT (Play no VS Code)
# ==========================================================
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path as _Path



def _find_free_port(start: int = 8501, tries: int = 50) -> int:
    import socket as _socket
    for p in range(start, start + tries):
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            return p
        except Exception:
            continue
        finally:
            try:
                s.close()
            except Exception:
                pass
    return start


def _ensure_streamlit_boot():
    """Permite rodar no VS Code (Play) com `python arquivo.py` e abrir o Streamlit automaticamente."""
    already_streamlit = (
        os.environ.get("STREAMLIT_SERVER_RUNNING") == "true"
        or "STREAMLIT_RUN_MAIN" in os.environ
    )
    if already_streamlit:
        return

    if os.environ.get("_AUTO_STREAMLIT_BOOT") == "1":
        return

    os.environ["_AUTO_STREAMLIT_BOOT"] = "1"
    this_file = _Path(__file__).resolve()

    # Porta padrão 8501; se estiver ocupada, escolhe a próxima livre.
    try:
        port = int((os.environ.get("STREAMLIT_PORT", "") or "").strip() or 0) or _find_free_port(8501, 50)
    except Exception:
        port = _find_free_port(8501, 50)

    os.environ["STREAMLIT_PORT"] = str(port)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(this_file),
        "--server.headless=false",
        "--server.runOnSave=true",
        "--server.port",
        str(port),
    ]

    url = f"http://localhost:{port}"
    print(f"[BOOT] Iniciando Streamlit automaticamente em {url} ...")

    subprocess.Popen(cmd, env=os.environ.copy())

    # Abre o navegador automaticamente (evita o “não apareceu nada”)
    try:
        import webbrowser as _webbrowser
        _webbrowser.open(url)
    except Exception:
        pass

    raise SystemExit(0)


_ensure_streamlit_boot()


# ==========================================================
# APP
# ==========================================================
from datetime import date, datetime
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st

from supabase import create_client, Client

# ==========================================================
# UI / VISUAL PROFISSIONAL (CSS + COMPONENTES)
# ==========================================================
import pandas as _pd


def ui_inject_global_style() -> None:
    st.markdown(
        """
        <style>
          /* remove elementos padrão */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}

          /* layout */
          .block-container {padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1320px;}
          h1, h2, h3 {letter-spacing: -0.2px;}

          /* inputs / botões */
          div.stButton>button {border-radius: 12px; padding: .55rem .9rem;}
          div[data-baseweb="input"]>div {border-radius: 12px;}
          div[data-baseweb="select"]>div {border-radius: 12px;}
          textarea {border-radius: 12px !important;}

          /* sidebar */
          section[data-testid="stSidebar"] > div {padding-top: 10px;}
          div[data-testid="stSidebar"] .stMarkdown {margin-bottom: 0.35rem;}

          /* metric cards (KPIs) */
          div[data-testid="stMetric"]{
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.03);
            border-radius: 18px;
            padding: 12px 14px;
          }
          div[data-testid="stMetricLabel"]{opacity:.70;}
          div[data-testid="stMetricValue"]{font-weight:850; letter-spacing:-0.3px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _pill(text: str) -> str:
    return f'<span class="pill">{text}</span>'


def ui_header(subtitle: str, pills: list[str]) -> None:
    left, right = st.columns([2.2, 1.4])
    with left:
        st.title(APP_TITLE)
        st.caption(subtitle)
    with right:
        pills = [p for p in pills if p]
        st.markdown(" ".join([f"`{p}`" for p in pills]) if pills else "`—`")
    st.divider()


def apply_period_df(df: _pd.DataFrame, col: str, date_ini: str | None, date_fim: str | None) -> _pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    d = df.copy()
    d[col] = _pd.to_datetime(d[col], errors="coerce")
    if date_ini:
        d = d[d[col] >= _pd.to_datetime(date_ini)]
    if date_fim:
        d = d[d[col] <= _pd.to_datetime(date_fim)]
    return d

def show_table_pro(
    df: _pd.DataFrame,
    *,
    title: str | None = None,
    money_cols: list[str] | None = None,
    date_cols: list[str] | None = None,
    hide_index: bool = True,
    height: int | None = 420,
) -> None:
    if title:
        st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("Sem dados para exibir.")
        return

    money_cols = money_cols or []
    date_cols = date_cols or []

    d = df.copy()
    for c in date_cols:
        if c in d.columns:
            d[c] = _pd.to_datetime(d[c], errors="coerce").dt.date

    cfg = {}
    for c in money_cols:
        if c in d.columns:
            cfg[c] = st.column_config.NumberColumn(c, format="R$ %.2f")

    for c in date_cols:
        if c in d.columns:
            cfg[c] = st.column_config.DateColumn(c, format="DD/MM/YYYY")

    st.dataframe(d, use_container_width=True, hide_index=hide_index, column_config=cfg, height=height)


def render_kpi_metrics(groups: list[tuple[str, list[tuple[str, str, str | None]]]]) -> None:
    """Renderiza KPIs em formato de cartões usando st.metric (sem HTML)."""
    for title, items in groups:
        st.markdown(f"#### {title}")
        cols = st.columns(4)
        for i, (label, value, hint) in enumerate(items):
            with cols[i % 4]:
                st.metric(label, value, help=hint)
        st.write("")


APP_TITLE = "Agência de Carros — Financeiro (Yago)"
LOCAL_CFG_FILE = _Path(__file__).with_name(".supabase_api.local.json")




def brl(v: float) -> str:
    try:
        v = float(v or 0)
    except Exception:
        v = 0.0
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def iso(d: date | datetime | str | None) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0



# ==========================================================
# Categorias de despesas (canonização)
# ==========================================================
# A ideia aqui é guardar SEMPRE um código estável no banco (ex.: "PECAS", "ANUNCIOS"),
# mas mostrar um rótulo amigável na tela.
EXPENSE_CATEGORIES = {
    "Peças": "PECAS",
    "Anúncios": "ANUNCIOS",
    "Mecânica": "MECANICA",
    "Elétrica": "ELETRICA",
    "Pintura": "PINTURA",
    "Lanternagem": "LANTERNAGEM",
    "Outros": "OUTROS",
}
EXPENSE_CATEGORY_LABEL = {v: k for k, v in EXPENSE_CATEGORIES.items()}


def normalize_expense_category(cat: str) -> str:
    cat = (cat or "").strip()
    if not cat:
        return "OUTROS"
    if cat in EXPENSE_CATEGORIES:
        return EXPENSE_CATEGORIES[cat]

    # se já veio como código, normaliza caracteres comuns (sem depender de libs extras)
    c = cat.upper()
    c = (
        c.replace("Ç", "C")
        .replace("Ã", "A").replace("Á", "A").replace("Â", "A").replace("À", "A")
        .replace("É", "E").replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O").replace("Ô", "O").replace("Õ", "O")
        .replace("Ú", "U").replace("Ü", "U")
    )
    c = c.replace(" ", "_")
    return c or "OUTROS"


def expense_category_label(code: str) -> str:
    code = (code or "").strip()
    return EXPENSE_CATEGORY_LABEL.get(code, code)


def bump_cache() -> None:
    st.session_state["cache_buster"] = int(st.session_state.get("cache_buster", 0)) + 1


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()


def load_local_cfg() -> dict:
    try:
        import json
        if LOCAL_CFG_FILE.exists():
            return json.loads(LOCAL_CFG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_local_cfg(cfg: dict) -> None:
    import json
    LOCAL_CFG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# ==========================================================
# Supabase client
# ==========================================================
def get_supabase_cfg() -> Tuple[str, str]:
    """
    Ordem:
      1) session_state (usuário colou no app)
      2) st.secrets (Community Cloud / .streamlit/secrets.toml local)
      3) variáveis de ambiente (SUPABASE_URL / SUPABASE_KEY)
      4) arquivo local opcional (inseguro)
    """
    if st.session_state.get("SUPABASE_URL") and st.session_state.get("SUPABASE_KEY"):
        return st.session_state["SUPABASE_URL"], st.session_state["SUPABASE_KEY"]

    # 2) Streamlit secrets
    try:
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            su = "https://gzsjcwzfkwezxjdxbexe.supabase.co"
            sk = "sb_publishable_DBMQvfKKCh_h072g1k4AHQ_YAQOIQuN"
            if su and sk:
                return su, sk
    except Exception:
        pass

    # 3) Variáveis de ambiente
    env_url = (os.environ.get("SUPABASE_URL") or "").strip()
    env_key = (os.environ.get("SUPABASE_KEY") or "").strip()
    if env_url and env_key:
        return env_url, env_key

    # 4) Arquivo local opcional
    local = load_local_cfg()
    su = str(local.get("SUPABASE_URL", "")).strip()
    sk = str(local.get("SUPABASE_KEY", "")).strip()
    if su and sk:
        return su, sk

    return "", ""


@st.cache_resource(show_spinner=False)
def supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)


def sb() -> Optional[Client]:
    url, key = get_supabase_cfg()
    if not url or not key:
        return None
    return supabase_client(url, key)


# ==========================================================
# Erros / Diagnóstico
# ==========================================================
def _err_text(e: Exception) -> str:
    try:
        return str(e)
    except Exception:
        return repr(e)


def is_auth_error(e: Exception) -> bool:
    t = _err_text(e).lower()
    # padrões comuns: 401/403, invalid api key, jwt, permission
    return (
        "401" in t
        or "403" in t
        or "invalid api key" in t
        or "jwt" in t
        or "not authorized" in t
        or "permission" in t
    )


def is_table_missing_error(e: Exception) -> bool:
    t = _err_text(e).lower()
    # padrões comuns PostgREST:
    # "Could not find the table 'settings' in the schema cache"
    # "Could not find the table 'public' in the schema cache"
    # "relation \"public.settings\" does not exist"
    return (
        "could not find the table" in t
        or "schema cache" in t
        or "does not exist" in t and "relation" in t
        or "pgrst" in t and "not found" in t
    )


# ==========================================================
# API wrappers (PostgREST)
# ==========================================================
def sb_select(table: str, columns: str = "*", filters: Optional[List[Tuple[str, str, Any]]] = None,
              order: Optional[Tuple[str, bool]] = None, limit: Optional[int] = None) -> pd.DataFrame:
    client = sb()
    if client is None:
        return pd.DataFrame()

    # MUITO IMPORTANTE: aqui é SOMENTE nome da tabela (sem "public.")
    q = client.table(table).select(columns)

    if filters:
        for col, op, val in filters:
            if op == "eq":
                q = q.eq(col, val)
            elif op == "neq":
                q = q.neq(col, val)
            elif op == "gte":
                q = q.gte(col, val)
            elif op == "lte":
                q = q.lte(col, val)
            elif op == "in":
                q = q.in_(col, val)
            elif op == "is":
                q = q.is_(col, val)
            else:
                raise ValueError(f"Filtro não suportado: {op}")

    if order:
        col, ascending = order
        q = q.order(col, desc=not ascending)

    if limit:
        q = q.limit(limit)

    res = q.execute()
    data = res.data or []
    return pd.DataFrame(data)


def sb_insert(table: str, payload: dict) -> dict:
    client = sb()
    if client is None:
        raise RuntimeError("Supabase não configurado.")
    res = client.table(table).insert(payload).execute()
    if not res.data:
        return {}
    return res.data[0]


def sb_update(table: str, payload: dict, where_col: str, where_val: Any) -> None:
    client = sb()
    if client is None:
        raise RuntimeError("Supabase não configurado.")
    client.table(table).update(payload).eq(where_col, where_val).execute()


def sb_upsert(table: str, payload: dict, on_conflict: str) -> None:
    client = sb()
    if client is None:
        raise RuntimeError("Supabase não configurado.")
    client.table(table).upsert(payload, on_conflict=on_conflict).execute()



def sb_delete(table: str, where_col: str, where_val: Any) -> None:
    client = sb()
    if client is None:
        raise RuntimeError("Supabase não configurado.")
    client.table(table).delete().eq(where_col, where_val).execute()


# ==========================================================
# Health checks
# ==========================================================
def supabase_healthcheck() -> Tuple[bool, bool, str]:
    """
    Retorna:
      (connected_ok, schema_ready, message)
    connected_ok: URL/KEY válidos (não-auth error)
    schema_ready: tabelas existem e API enxerga elas (settings)
    """
    client = sb()
    if client is None:
        return False, False, "Supabase não configurado (SUPABASE_URL/SUPABASE_KEY)."

    try:
        # 1) "ping" via tabela esperada
        _ = client.table("settings").select("key").limit(1).execute()
        return True, True, "Conexão OK e tabelas encontradas."
    except Exception as e:
        if is_auth_error(e):
            return False, False, "Chave/URL inválida ou sem permissão (401/403)."
        if is_table_missing_error(e):
            return True, False, "Conexão OK, mas as tabelas ainda não existem (ou API não está expondo o schema)."
        # outros erros
        return True, False, f"Conectou, mas houve erro ao ler schema/tabelas: {_err_text(e)}"


# ==========================================================
# Business logic
# ==========================================================
def get_setting(key: str) -> str:
    df = sb_select("settings", "key,value", filters=[("key", "eq", key)])
    if df.empty:
        return ""
    return str(df.loc[0, "value"])


def set_setting(key: str, value: str) -> None:
    sb_upsert("settings", {"key": key, "value": str(value)}, on_conflict="key")
    bump_cache()





# ==========================================================
# Reset da Base (perigoso!)
# ==========================================================
def sb_delete_all(table: str, pk: str = "id", pk_is_numeric: bool = True) -> int:
    """Apaga todos os registros de uma tabela (PostgREST exige filtro).
    Retorna a quantidade (aproximada) de linhas deletadas, quando disponível.
    """
    client = sb()
    if client is None:
        raise RuntimeError("Supabase não configurado.")

    q = client.table(table).delete()
    if pk_is_numeric:
        q = q.gte(pk, 0)
    else:
        q = q.neq(pk, "__never__")  # apaga tudo exceto um valor impossível

    res = q.execute()
    data = getattr(res, "data", None) or []
    try:
        return int(len(data))
    except Exception:
        return 0


def reset_database() -> None:
    """Zera a base via API (ordem dos deletes para evitar FK)."""
    # child tables primeiro
    for tbl in ["cash_movements", "retentions", "payables", "expenses", "sales", "vehicles"]:
        try:
            sb_delete_all(tbl, pk="id", pk_is_numeric=True)
        except Exception:
            pass

    # settings (key é texto)
    try:
        sb_delete_all("settings", pk="key", pk_is_numeric=False)
    except Exception:
        pass

    # volta saldo inicial para 0 (opcional)
    try:
        set_setting("opening_balance", "0")
    except Exception:
        pass

    bump_cache()

def list_vehicles(include_sold: bool = True, include_deleted: bool = False) -> pd.DataFrame:
    filters = []
    if not include_sold:
        filters.append(("status", "neq", "VENDIDO"))
    if not include_deleted:
        filters.append(("status", "neq", "EXCLUIDO"))

    df = sb_select("vehicles", "*", filters=filters, order=("id", False))
    if not df.empty:
        if "purchase_date" in df.columns:
            df["purchase_date"] = df["purchase_date"].astype(str)
        df = df.sort_values(
            by=["status", "purchase_date", "id"],
            ascending=[True, False, False],
            na_position="last",
        )
    return df


def list_sales() -> pd.DataFrame:
    df = sb_select("sales", "*", order=("sale_date", False))
    if df.empty:
        return df
    v = sb_select("vehicles", "id,model,plate,year,purchase_cost,status", order=("id", False))
    if not v.empty:
        df = df.merge(v, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_veh"))
        df = df.rename(columns={"model": "vehicle_model", "plate": "vehicle_plate", "year": "vehicle_year"})
    return df


def list_expenses() -> pd.DataFrame:
    df = sb_select("expenses", "*", order=("expense_date", False))
    if df.empty:
        return df
    v = sb_select("vehicles", "id,model,plate", order=("id", False))
    if not v.empty:
        df = df.merge(v, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_veh"))
        df = df.rename(columns={"model": "vehicle_model", "plate": "vehicle_plate"})
    return df


def list_cash() -> pd.DataFrame:
    df = sb_select("cash_movements", "*", order=("mov_date", False))
    if df.empty:
        return df
    v = sb_select("vehicles", "id,model,plate", order=("id", False))
    if not v.empty:
        df = df.merge(v, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_veh"))
        df = df.rename(columns={"model": "vehicle_model", "plate": "vehicle_plate"})
    return df





# ==========================================================
# Editar / Excluir lançamentos do Caixa (CRUD)
# ==========================================================
def update_cash_movement(
    cash_id: int,
    mov_date: str,
    direction: str,
    category: str,
    description: str,
    amount: float,
    vehicle_id: Optional[int] = None,
) -> None:
    payload = {
        "mov_date": mov_date or None,
        "direction": (direction or "IN").strip().upper(),
        "category": (category or "OUTROS").strip().upper(),
        "description": description.strip() or None,
        "amount": float(amount or 0),
        "vehicle_id": int(vehicle_id) if vehicle_id else None,
    }
    sb_update("cash_movements", payload, "id", int(cash_id))
    bump_cache()


def delete_cash_movement_hard(cash_id: int) -> None:
    sb_delete("cash_movements", "id", int(cash_id))
    bump_cache()


def add_cash_manual(
    mov_date: str,
    direction: str,
    category: str,
    description: str,
    amount: float,
    vehicle_id: Optional[int] = None,
) -> int:
    row = sb_insert("cash_movements", {
        "mov_date": mov_date or date.today().isoformat(),
        "direction": (direction or "IN").strip().upper(),
        "category": (category or "OUTROS").strip().upper(),
        "description": description.strip() or None,
        "amount": float(amount or 0),
        "vehicle_id": int(vehicle_id) if vehicle_id else None,
    })
    bump_cache()
    return int(row.get("id", 0) or 0)

def list_retentions_pending() -> pd.DataFrame:
    df = sb_select("retentions", "*", filters=[("status", "eq", "PENDENTE")], order=("created_date", False))
    if df.empty:
        return df
    sales = sb_select("sales", "id,vehicle_id,sale_date", filters=[("id", "in", df["sale_id"].tolist())])
    if not sales.empty:
        df = df.merge(sales, left_on="sale_id", right_on="id", how="left", suffixes=("", "_sale"))
    veh_ids = sales["vehicle_id"].dropna().astype(int).unique().tolist() if not sales.empty else []
    if veh_ids:
        v = sb_select("vehicles", "id,model,plate", filters=[("id", "in", veh_ids)])
        if not v.empty:
            df = df.merge(v, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_veh"))
            df = df.rename(columns={"model": "vehicle_model", "plate": "vehicle_plate"})
    return df


def list_payables_pending() -> pd.DataFrame:
    df = sb_select("payables", "*", filters=[("status", "eq", "PENDENTE")], order=("created_date", False))
    if df.empty:
        return df
    sales = sb_select("sales", "id,vehicle_id,sale_date", filters=[("id", "in", df["sale_id"].tolist())])
    if not sales.empty:
        df = df.merge(sales, left_on="sale_id", right_on="id", how="left", suffixes=("", "_sale"))
    veh_ids = sales["vehicle_id"].dropna().astype(int).unique().tolist() if not sales.empty else []
    if veh_ids:
        v = sb_select("vehicles", "id,model,plate", filters=[("id", "in", veh_ids)])
        if not v.empty:
            df = df.merge(v, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_veh"))
            df = df.rename(columns={"model": "vehicle_model", "plate": "vehicle_plate"})
    return df



def compute_kpis(date_ini: Optional[str] = None, date_fim: Optional[str] = None) -> Dict[str, float]:
    vehicles = list_vehicles(include_sold=True)
    sales = list_sales()
    expenses = list_expenses()

    def between(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty or col not in df.columns:
            return df
        out = df.copy()
        out[col] = out[col].astype(str)
        if date_ini:
            out = out[out[col] >= date_ini]
        if date_fim:
            out = out[out[col] <= date_fim]
        return out

    sales_f = between(sales, "sale_date")
    exp_f = between(expenses, "expense_date")
    veh_purch_f = between(vehicles, "purchase_date")

    # Compras de veículos no período (entra no total de despesas)
    purchases_period = safe_float(veh_purch_f["purchase_cost"].sum()) if not veh_purch_f.empty and "purchase_cost" in veh_purch_f.columns else 0.0

    # Custo (COGS) dos veículos vendidos no período (para cálculo do lucro)
    purchase_cost_sold = 0.0
    if not sales_f.empty and "vehicle_id" in sales_f.columns and not vehicles.empty:
        if "id" in vehicles.columns and "purchase_cost" in vehicles.columns:
            vmap = {int(r["id"]): safe_float(r.get("purchase_cost")) for _, r in vehicles.iterrows() if str(r.get("id", "")).strip() != ""}
            for _, r in sales_f.iterrows():
                try:
                    vid = int(r.get("vehicle_id") or 0)
                except Exception:
                    vid = 0
                purchase_cost_sold += safe_float(vmap.get(vid, 0.0))

    total_sales_price = safe_float(sales_f["sale_price"].sum()) if not sales_f.empty and "sale_price" in sales_f.columns else 0.0

    # Despesas por categoria (no período)
    if not exp_f.empty:
        exp_f = exp_f.copy()
        exp_f["category_norm"] = exp_f["category"].apply(lambda x: normalize_expense_category(str(x)))
        parts = safe_float(exp_f.loc[exp_f["category_norm"] == "PECAS", "amount"].sum())
        ads = safe_float(exp_f.loc[exp_f["category_norm"] == "ANUNCIOS", "amount"].sum())
        other_exp = safe_float(exp_f.loc[~exp_f["category_norm"].isin(["PECAS", "ANUNCIOS"]), "amount"].sum())
    else:
        parts = 0.0
        ads = 0.0
        other_exp = 0.0

    warranties = safe_float(sales_f["warranty_cost"].sum()) if not sales_f.empty else 0.0
    commissions = safe_float(sales_f["commission_amount"].sum()) if not sales_f.empty else 0.0
    retained_total = safe_float(sales_f["retained_amount"].sum()) if not sales_f.empty else 0.0

    total_expenses_including_purchases = purchases_period + parts + ads + other_exp + warranties + commissions

    # Lucro (período): vendas - COGS (vendidos) - despesas - garantias - comissões
    gross_profit = total_sales_price - purchase_cost_sold - parts - ads - other_exp - warranties - commissions

    # Estoque atual (não depende do filtro)
    stock = vehicles[vehicles["status"] != "VENDIDO"] if not vehicles.empty and "status" in vehicles.columns else vehicles
    stock_count = float(len(stock)) if stock is not None and not stock.empty else 0.0
    stock_value = safe_float(stock["purchase_cost"].sum()) if stock is not None and not stock.empty and "purchase_cost" in stock.columns else 0.0

    opening = safe_float(get_setting("opening_balance") or 0)
    cash_df = list_cash()
    cash_in = safe_float(cash_df.loc[cash_df["direction"] == "IN", "amount"].sum()) if not cash_df.empty else 0.0
    cash_out = safe_float(cash_df.loc[cash_df["direction"] == "OUT", "amount"].sum()) if not cash_df.empty else 0.0
    cash_balance = opening + cash_in - cash_out

    ret_pend = list_retentions_pending()
    retained_pending = safe_float(ret_pend["amount"].sum()) if not ret_pend.empty else 0.0

    pay_pend = list_payables_pending()
    payables_pending = safe_float(pay_pend["amount"].sum()) if not pay_pend.empty else 0.0

    return {
        "purchases_period": purchases_period,
        "purchase_cost_sold": purchase_cost_sold,
        "total_sales_price": total_sales_price,
        "gross_profit": gross_profit,
        "parts": parts,
        "ads": ads,
        "other_exp": other_exp,
        "warranties": warranties,
        "commissions": commissions,
        "retained_total": retained_total,
        "cash_balance": cash_balance,
        "stock_count": stock_count,
        "stock_value": stock_value,
        "retained_pending": retained_pending,
        "payables_pending": payables_pending,
        "total_expenses_including_purchases": total_expenses_including_purchases,
    }

def add_vehicle(model: str, plate: str, year: Optional[int], purchase_date: str, purchase_cost: float, notes: str, move_cash: bool) -> int:
    payload = {
        "plate": plate.strip() or None,
        "model": model.strip(),
        "year": int(year) if year else None,
        "purchase_date": purchase_date or None,
        "purchase_cost": float(purchase_cost or 0),
        "status": "EM_ESTOQUE",
        "notes": notes.strip() or None,
    }
    row = sb_insert("vehicles", payload)
    vid = int(row.get("id", 0) or 0)

    if move_cash and float(purchase_cost or 0) > 0:
        sb_insert("cash_movements", {
            "mov_date": purchase_date or date.today().isoformat(),
            "direction": "OUT",
            "category": "COMPRA",
            "description": f"Compra do veículo: {model} ({plate})",
            "amount": float(purchase_cost),
            "vehicle_id": vid,
        })

    bump_cache()
    return vid



def update_vehicle(
    vehicle_id: int,
    model: str,
    plate: str,
    year: Optional[int],
    purchase_date: str,
    purchase_cost: float,
    status: str,
    notes: str,
    update_purchase_cash_movement: bool = False,
) -> None:
    payload = {
        "plate": plate.strip() or None,
        "model": model.strip(),
        "year": int(year) if year else None,
        "purchase_date": purchase_date or None,
        "purchase_cost": float(purchase_cost or 0),
        "status": (status or "EM_ESTOQUE").strip() or "EM_ESTOQUE",
        "notes": notes.strip() or None,
    }
    sb_update("vehicles", payload, "id", int(vehicle_id))

    # opcional: se existir um lançamento de COMPRA no caixa, atualiza para manter coerência
    if update_purchase_cash_movement:
        df = sb_select(
            "cash_movements",
            "id",
            filters=[("vehicle_id", "eq", int(vehicle_id)), ("category", "eq", "COMPRA")],
            order=("id", True),
            limit=1,
        )
        if not df.empty:
            cmid = int(df.loc[0, "id"])
            sb_update(
                "cash_movements",
                {
                    "mov_date": purchase_date or date.today().isoformat(),
                    "amount": float(purchase_cost or 0),
                    "description": f"Compra do veículo: {payload['model']} ({payload.get('plate') or '-'})",
                },
                "id",
                cmid,
            )

    bump_cache()


def archive_vehicle(vehicle_id: int) -> None:
    # soft delete (recomendado): some das telas por padrão, mas mantém histórico
    sb_update("vehicles", {"status": "EXCLUIDO"}, "id", int(vehicle_id))
    bump_cache()


def delete_vehicle_hard(vehicle_id: int) -> None:
    # delete definitivo (irreversível) — cuidado com cascades
    sb_delete("vehicles", "id", int(vehicle_id))
    bump_cache()


def vehicle_related_counts(vehicle_id: int) -> Dict[str, int]:
    vid = int(vehicle_id)

    sales = sb_select("sales", "id", filters=[("vehicle_id", "eq", vid)])
    sale_ids = sales["id"].dropna().astype(int).tolist() if not sales.empty and "id" in sales.columns else []

    expenses = sb_select("expenses", "id", filters=[("vehicle_id", "eq", vid)])
    cash = sb_select("cash_movements", "id", filters=[("vehicle_id", "eq", vid)])

    retentions = sb_select("retentions", "id", filters=[("sale_id", "in", sale_ids)]) if sale_ids else pd.DataFrame()
    payables = sb_select("payables", "id", filters=[("sale_id", "in", sale_ids)]) if sale_ids else pd.DataFrame()

    return {
        "sales": int(len(sales)) if not sales.empty else 0,
        "expenses": int(len(expenses)) if not expenses.empty else 0,
        "cash_movements": int(len(cash)) if not cash.empty else 0,
        "retentions": int(len(retentions)) if not retentions.empty else 0,
        "payables": int(len(payables)) if not payables.empty else 0,
    }


def add_expense(vehicle_id: Optional[int], expense_date: str, category: str, description: str, amount: float, move_cash: bool) -> int:
    cat_code = normalize_expense_category(category)

    row = sb_insert("expenses", {
        "vehicle_id": int(vehicle_id) if vehicle_id else None,
        "expense_date": expense_date,
        "category": cat_code,
        "description": description.strip() or None,
        "amount": float(amount or 0),
    })
    eid = int(row.get("id", 0) or 0)

    if move_cash and float(amount or 0) > 0:
        sb_insert("cash_movements", {
            "mov_date": expense_date,
            "direction": "OUT",
            "category": cat_code,
            "description": (description.strip() or expense_category_label(cat_code) or cat_code),
            "amount": float(amount),
            "vehicle_id": int(vehicle_id) if vehicle_id else None,
            "expense_id": eid,
        })

    bump_cache()
    return eid





# ==========================================================
# Editar / Excluir despesas (CRUD)
# ==========================================================
def expense_cash_movement_ids(expense_id: int) -> list[int]:
    """Retorna IDs de lançamentos de caixa vinculados a uma despesa (via expense_id)."""
    df = sb_select("cash_movements", "id", filters=[("expense_id", "eq", int(expense_id))])
    if df.empty or "id" not in df.columns:
        return []
    ids: list[int] = []
    for v in df["id"].tolist():
        try:
            ids.append(int(v))
        except Exception:
            pass
    return ids


def update_expense(
    expense_id: int,
    vehicle_id: Optional[int],
    expense_date: str,
    category: str,
    description: str,
    amount: float,
    update_cash_movement: bool = True,
) -> None:
    eid = int(expense_id)
    cat_code = normalize_expense_category(category)

    payload = {
        "vehicle_id": int(vehicle_id) if vehicle_id else None,
        "expense_date": expense_date or None,
        "category": cat_code,
        "description": description.strip() or None,
        "amount": float(amount or 0),
    }
    sb_update("expenses", payload, "id", eid)

    if update_cash_movement:
        cm_ids = expense_cash_movement_ids(eid)
        if cm_ids:
            # atualiza todos os lançamentos vinculados (por segurança)
            for cmid in cm_ids:
                sb_update(
                    "cash_movements",
                    {
                        "mov_date": expense_date or date.today().isoformat(),
                        "category": cat_code,
                        "description": (description.strip() or expense_category_label(cat_code) or cat_code),
                        "amount": float(amount or 0),
                        "vehicle_id": int(vehicle_id) if vehicle_id else None,
                    },
                    "id",
                    int(cmid),
                )

    bump_cache()


def delete_expense_hard(expense_id: int, delete_cash: bool = True) -> None:
    eid = int(expense_id)
    if delete_cash:
        cm_ids = expense_cash_movement_ids(eid)
        for cmid in cm_ids:
            try:
                sb_delete("cash_movements", "id", int(cmid))
            except Exception:
                pass
    sb_delete("expenses", "id", eid)
    bump_cache()

def add_sale(
    vehicle_id: int,
    sale_date: str,
    sale_price: float,
    received_amount: float,
    retained_amount: float,
    commission_amount: float,
    warranty_cost: float,
    buyer: str,
    notes: str,
    move_cash_received: bool,
    commission_paid_now: bool,
    warranty_paid_now: bool,
) -> int:
    srow = sb_insert("sales", {
        "vehicle_id": int(vehicle_id),
        "sale_date": sale_date,
        "sale_price": float(sale_price or 0),
        "received_amount": float(received_amount or 0),
        "retained_amount": float(retained_amount or 0),
        "commission_amount": float(commission_amount or 0),
        "warranty_cost": float(warranty_cost or 0),
        "buyer": buyer.strip() or None,
        "notes": notes.strip() or None,
    })
    sid = int(srow.get("id", 0) or 0)

    sb_update("vehicles", {"status": "VENDIDO"}, "id", int(vehicle_id))

    if move_cash_received and float(received_amount or 0) > 0:
        sb_insert("cash_movements", {
            "mov_date": sale_date,
            "direction": "IN",
            "category": "VENDA",
            "description": "Recebimento de venda",
            "amount": float(received_amount),
            "vehicle_id": int(vehicle_id),
            "sale_id": sid,
        })

    if float(retained_amount or 0) > 0:
        sb_insert("retentions", {
            "sale_id": sid,
            "amount": float(retained_amount),
            "status": "PENDENTE",
            "created_date": sale_date,
            "released_date": None,
        })

    if float(commission_amount or 0) > 0:
        if commission_paid_now:
            sb_insert("cash_movements", {
                "mov_date": sale_date,
                "direction": "OUT",
                "category": "COMISSAO",
                "description": "Pagamento de comissão",
                "amount": float(commission_amount),
                "vehicle_id": int(vehicle_id),
                "sale_id": sid,
            })
        else:
            sb_insert("payables", {
                "sale_id": sid,
                "kind": "COMISSAO",
                "amount": float(commission_amount),
                "status": "PENDENTE",
                "created_date": sale_date,
                "paid_date": None,
            })

    if float(warranty_cost or 0) > 0:
        if warranty_paid_now:
            sb_insert("cash_movements", {
                "mov_date": sale_date,
                "direction": "OUT",
                "category": "GARANTIA",
                "description": "Custo de garantia",
                "amount": float(warranty_cost),
                "vehicle_id": int(vehicle_id),
                "sale_id": sid,
            })
        else:
            sb_insert("payables", {
                "sale_id": sid,
                "kind": "GARANTIA",
                "amount": float(warranty_cost),
                "status": "PENDENTE",
                "created_date": sale_date,
                "paid_date": None,
            })

    bump_cache()
    return sid



# ==========================================================
# Editar / Excluir vendas (CRUD)
# ==========================================================
def _parse_date_any(x: Any) -> date:
    if x is None or x == "":
        return date.today()
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x).date()
        except Exception:
            pass
    return date.today()


def _first_int(series: pd.Series, default: Optional[int] = None) -> Optional[int]:
    try:
        if series is None or len(series) == 0:
            return default
        v = series.iloc[0]
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def sale_link_snapshot(sale_id: int) -> Dict[str, Any]:
    sid = int(sale_id)
    ret = sb_select("retentions", "*", filters=[("sale_id", "eq", sid)])
    pay = sb_select("payables", "*", filters=[("sale_id", "eq", sid)])
    cash = sb_select("cash_movements", "*", filters=[("sale_id", "eq", sid)])

    out: Dict[str, Any] = {
        "retentions_total": int(len(ret)) if not ret.empty else 0,
        "retentions_pending": int(len(ret[ret["status"] == "PENDENTE"])) if not ret.empty and "status" in ret.columns else 0,
        "payables_total": int(len(pay)) if not pay.empty else 0,
        "payables_pending": int(len(pay[pay["status"] == "PENDENTE"])) if not pay.empty and "status" in pay.columns else 0,
        "cash_total": int(len(cash)) if not cash.empty else 0,
    }

    # IDs úteis
    out["cash_venda_in_id"] = None
    out["cash_comissao_out_id"] = None
    out["cash_garantia_out_id"] = None

    if not cash.empty:
        def _pick(direction: str, category: str) -> Optional[int]:
            df = cash.copy()
            if "direction" in df.columns:
                df = df[df["direction"] == direction]
            if "category" in df.columns:
                df = df[df["category"] == category]
            if df.empty:
                return None
            df = df.sort_values(by=["id"], ascending=False) if "id" in df.columns else df
            return _first_int(df["id"], None)

        out["cash_venda_in_id"] = _pick("IN", "VENDA")
        out["cash_comissao_out_id"] = _pick("OUT", "COMISSAO")
        out["cash_garantia_out_id"] = _pick("OUT", "GARANTIA")

    # Retention pendente (se houver)
    out["retention_pending_id"] = None
    if not ret.empty and "status" in ret.columns:
        rpend = ret[ret["status"] == "PENDENTE"].copy()
        if not rpend.empty:
            rpend = rpend.sort_values(by=["id"], ascending=False) if "id" in rpend.columns else rpend
            out["retention_pending_id"] = _first_int(rpend["id"], None)

    # Payables pendentes por tipo
    out["payable_comissao_pending_id"] = None
    out["payable_garantia_pending_id"] = None
    if not pay.empty and "status" in pay.columns and "kind" in pay.columns:
        pend = pay[pay["status"] == "PENDENTE"].copy()
        if not pend.empty:
            for kind, key in [("COMISSAO", "payable_comissao_pending_id"), ("GARANTIA", "payable_garantia_pending_id")]:
                dfk = pend[pend["kind"] == kind].copy()
                if not dfk.empty:
                    dfk = dfk.sort_values(by=["id"], ascending=False) if "id" in dfk.columns else dfk
                    out[key] = _first_int(dfk["id"], None)

    return out


def _maybe_update_vehicle_status_after_sale_change(old_vehicle_id: Optional[int], new_vehicle_id: Optional[int], sale_id: int) -> None:
    # garante que o veículo da venda fica VENDIDO e, se trocou, o antigo volta para EM_ESTOQUE se não tiver mais vendas
    if new_vehicle_id is not None:
        sb_update("vehicles", {"status": "VENDIDO"}, "id", int(new_vehicle_id))

    if old_vehicle_id is None or old_vehicle_id == new_vehicle_id:
        return

    other = sb_select(
        "sales",
        "id",
        filters=[("vehicle_id", "eq", int(old_vehicle_id)), ("id", "neq", int(sale_id))],
        limit=1,
    )
    if other.empty:
        # só volta se não estiver EXCLUIDO
        v = sb_select("vehicles", "id,status", filters=[("id", "eq", int(old_vehicle_id))], limit=1)
        if not v.empty and str(v.loc[0, "status"]) != "EXCLUIDO":
            sb_update("vehicles", {"status": "EM_ESTOQUE"}, "id", int(old_vehicle_id))


def _sync_cash_venda_received(sale_id: int, vehicle_id: int, received_amount: float, mov_date: str) -> None:
    sid = int(sale_id)
    df = sb_select(
        "cash_movements",
        "id",
        filters=[("sale_id", "eq", sid), ("category", "eq", "VENDA"), ("direction", "eq", "IN")],
        order=("id", False),
        limit=1,
    )
    if float(received_amount or 0) <= 0:
        if not df.empty:
            sb_delete("cash_movements", "id", int(df.loc[0, "id"]))
        return

    payload = {
        "mov_date": mov_date,
        "direction": "IN",
        "category": "VENDA",
        "description": "Recebimento de venda (editado)",
        "amount": float(received_amount),
        "vehicle_id": int(vehicle_id),
        "sale_id": sid,
    }
    if df.empty:
        sb_insert("cash_movements", payload)
    else:
        sb_update("cash_movements", payload, "id", int(df.loc[0, "id"]))


def _sync_retention_pending(sale_id: int, retained_amount: float, created_date: str) -> None:
    sid = int(sale_id)
    df = sb_select(
        "retentions",
        "id,amount,status",
        filters=[("sale_id", "eq", sid), ("status", "eq", "PENDENTE")],
        order=("id", False),
        limit=1,
    )

    if float(retained_amount or 0) <= 0:
        if not df.empty:
            sb_delete("retentions", "id", int(df.loc[0, "id"]))
        return

    payload = {
        "sale_id": sid,
        "amount": float(retained_amount),
        "status": "PENDENTE",
        "created_date": created_date,
        "released_date": None,
    }
    if df.empty:
        sb_insert("retentions", payload)
    else:
        sb_update("retentions", payload, "id", int(df.loc[0, "id"]))


def _sync_payable_or_paid_cash(
    *,
    sale_id: int,
    vehicle_id: int,
    kind: str,  # COMISSAO / GARANTIA
    amount: float,
    mode: str,  # PENDENTE / PAGO / NAO_ALTERAR
    pay_date: str,
) -> None:
    sid = int(sale_id)
    kind = (kind or "").strip().upper()
    if kind not in {"COMISSAO", "GARANTIA"}:
        return

    # pendência
    df_pend = sb_select(
        "payables",
        "id,amount,status",
        filters=[("sale_id", "eq", sid), ("kind", "eq", kind), ("status", "eq", "PENDENTE")],
        order=("id", False),
        limit=1,
    )

    if float(amount or 0) <= 0:
        # remove pendência pendente, mas não mexe em histórico de caixa
        if not df_pend.empty:
            sb_delete("payables", "id", int(df_pend.loc[0, "id"]))
        return

    if mode == "NAO_ALTERAR":
        return

    if mode == "PENDENTE":
        payload = {
            "sale_id": sid,
            "kind": kind,
            "amount": float(amount),
            "status": "PENDENTE",
            "created_date": pay_date,  # data base
            "paid_date": None,
        }
        if df_pend.empty:
            sb_insert("payables", payload)
        else:
            sb_update("payables", payload, "id", int(df_pend.loc[0, "id"]))
        return

    if mode == "PAGO":
        # marca pendência como paga (se existir)
        if not df_pend.empty:
            sb_update("payables", {"status": "PAGO", "paid_date": pay_date}, "id", int(df_pend.loc[0, "id"]))

        # cria/atualiza lançamento de caixa de pagamento
        df_cash = sb_select(
            "cash_movements",
            "id",
            filters=[("sale_id", "eq", sid), ("category", "eq", kind), ("direction", "eq", "OUT")],
            order=("id", False),
            limit=1,
        )
        payload = {
            "mov_date": pay_date,
            "direction": "OUT",
            "category": kind,
            "description": f"Pagamento de {kind.lower()} (editado)",
            "amount": float(amount),
            "vehicle_id": int(vehicle_id),
            "sale_id": sid,
        }
        if df_cash.empty:
            sb_insert("cash_movements", payload)
        else:
            sb_update("cash_movements", payload, "id", int(df_cash.loc[0, "id"]))
        return


def update_sale_record(
    *,
    sale_id: int,
    vehicle_id: int,
    sale_date: str,
    sale_price: float,
    received_amount: float,
    retained_amount: float,
    commission_amount: float,
    warranty_cost: float,
    buyer: str,
    notes: str,
    sync_received_cash: bool,
    received_cash_date: str,
    retention_mode: str,  # PENDENTE / NAO_ALTERAR
    commission_mode: str,  # PENDENTE / PAGO / NAO_ALTERAR
    commission_pay_date: str,
    warranty_mode: str,  # PENDENTE / PAGO / NAO_ALTERAR
    warranty_pay_date: str,
) -> None:
    sid = int(sale_id)

    # venda atual (para saber se trocou veículo)
    cur = sb_select("sales", "*", filters=[("id", "eq", sid)], limit=1)
    if cur.empty:
        raise RuntimeError("Venda não encontrada.")

    old_vehicle_id = int(cur.loc[0, "vehicle_id"]) if "vehicle_id" in cur.columns and pd.notna(cur.loc[0, "vehicle_id"]) else None

    # update da venda
    sb_update(
        "sales",
        {
            "vehicle_id": int(vehicle_id),
            "sale_date": sale_date,
            "sale_price": float(sale_price or 0),
            "received_amount": float(received_amount or 0),
            "retained_amount": float(retained_amount or 0),
            "commission_amount": float(commission_amount or 0),
            "warranty_cost": float(warranty_cost or 0),
            "buyer": buyer.strip() or None,
            "notes": notes.strip() or None,
        },
        "id",
        sid,
    )

    # se trocou veículo, ajusta status e referencia dos lançamentos
    _maybe_update_vehicle_status_after_sale_change(old_vehicle_id, int(vehicle_id), sid)

    # atualiza vehicle_id em cash_movements ligados à venda
    if old_vehicle_id is not None and int(vehicle_id) != old_vehicle_id:
        cash_ids = sb_select("cash_movements", "id", filters=[("sale_id", "eq", sid)])
        if not cash_ids.empty and "id" in cash_ids.columns:
            for cmid in cash_ids["id"].dropna().astype(int).tolist():
                sb_update("cash_movements", {"vehicle_id": int(vehicle_id)}, "id", int(cmid))

    # sincronizações opcionais
    if sync_received_cash:
        _sync_cash_venda_received(sid, int(vehicle_id), float(received_amount or 0), received_cash_date)

    if retention_mode == "PENDENTE":
        _sync_retention_pending(sid, float(retained_amount or 0), sale_date)

    _sync_payable_or_paid_cash(
        sale_id=sid,
        vehicle_id=int(vehicle_id),
        kind="COMISSAO",
        amount=float(commission_amount or 0),
        mode=commission_mode,
        pay_date=commission_pay_date,
    )

    _sync_payable_or_paid_cash(
        sale_id=sid,
        vehicle_id=int(vehicle_id),
        kind="GARANTIA",
        amount=float(warranty_cost or 0),
        mode=warranty_mode,
        pay_date=warranty_pay_date,
    )

    bump_cache()


def delete_sale_record(
    *,
    sale_id: int,
    delete_cash_categories: List[str],
    revert_vehicle_status: bool = True,
) -> None:
    sid = int(sale_id)
    cur = sb_select("sales", "id,vehicle_id", filters=[("id", "eq", sid)], limit=1)
    if cur.empty:
        raise RuntimeError("Venda não encontrada.")
    vehicle_id = int(cur.loc[0, "vehicle_id"]) if "vehicle_id" in cur.columns and pd.notna(cur.loc[0, "vehicle_id"]) else None

    # apaga lançamentos de caixa escolhidos (antes de apagar a venda)
    cats = [c.strip().upper() for c in (delete_cash_categories or []) if str(c).strip()]
    if cats:
        cash = sb_select("cash_movements", "id,category", filters=[("sale_id", "eq", sid)])
        if not cash.empty and "id" in cash.columns and "category" in cash.columns:
            for _, r in cash.iterrows():
                try:
                    if str(r["category"]).upper() in cats:
                        sb_delete("cash_movements", "id", int(r["id"]))
                except Exception:
                    pass

    # apaga a venda (retentions/payables cascadem; cash_movements ficaria sale_id NULL, mas já removemos o que quisermos)
    sb_delete("sales", "id", sid)

    # se não existir outra venda para o veículo, volta EM_ESTOQUE
    if revert_vehicle_status and vehicle_id is not None:
        other = sb_select("sales", "id", filters=[("vehicle_id", "eq", int(vehicle_id))], limit=1)
        if other.empty:
            v = sb_select("vehicles", "id,status", filters=[("id", "eq", int(vehicle_id))], limit=1)
            if not v.empty and str(v.loc[0, "status"]) != "EXCLUIDO":
                sb_update("vehicles", {"status": "EM_ESTOQUE"}, "id", int(vehicle_id))

    bump_cache()


def release_retention(retention_id: int, receive_date: str, description: str = "Liberação de retida") -> None:
    df = sb_select("retentions", "*", filters=[("id", "eq", int(retention_id))])
    if df.empty:
        return
    sale_id = int(df.loc[0, "sale_id"])
    amount = float(df.loc[0, "amount"] or 0)

    s = sb_select("sales", "id,vehicle_id", filters=[("id", "eq", sale_id)])
    vehicle_id = int(s.loc[0, "vehicle_id"]) if not s.empty else None

    sb_update("retentions", {"status": "LIBERADA", "released_date": receive_date}, "id", int(retention_id))
    sb_insert("cash_movements", {
        "mov_date": receive_date,
        "direction": "IN",
        "category": "RETIDA",
        "description": description,
        "amount": float(amount),
        "vehicle_id": vehicle_id,
        "sale_id": sale_id,
    })
    bump_cache()


def pay_payable(payable_id: int, pay_date: str, description: str = "Baixa de pendência") -> None:
    df = sb_select("payables", "*", filters=[("id", "eq", int(payable_id))])
    if df.empty:
        return
    sale_id = int(df.loc[0, "sale_id"])
    kind = str(df.loc[0, "kind"])
    amount = float(df.loc[0, "amount"] or 0)

    s = sb_select("sales", "id,vehicle_id", filters=[("id", "eq", sale_id)])
    vehicle_id = int(s.loc[0, "vehicle_id"]) if not s.empty else None

    sb_update("payables", {"status": "PAGO", "paid_date": pay_date}, "id", int(payable_id))
    sb_insert("cash_movements", {
        "mov_date": pay_date,
        "direction": "OUT",
        "category": kind,
        "description": description,
        "amount": float(amount),
        "vehicle_id": vehicle_id,
        "sale_id": sale_id,
    })
    bump_cache()


# ==========================================================
# UI
# ==========================================================
def page_connect() -> None:
    st.title(APP_TITLE)
    st.error("Supabase API não configurado (SUPABASE_URL / SUPABASE_KEY).")

    st.markdown("### Streamlit Community Cloud (recomendado)")
    st.caption("No Community Cloud: App settings → Secrets (ou **Advanced settings** no deploy).")
    st.code(
        """SUPABASE_URL = "https://SEU-PROJECT-REF.supabase.co"
SUPABASE_KEY = "SUA-ANON-KEY"
""",
        language="toml",
    )

    st.divider()

    st.markdown("### Rodando local (Windows) — opcional")
    st.code(
        """setx SUPABASE_URL "https://SEU-PROJECT-REF.supabase.co"
setx SUPABASE_KEY "SUA-ANON-KEY"
""",
        language="powershell",
    )
    st.caption("Depois feche/abra o terminal e o VS Code.")

    st.divider()
    st.markdown("### Ou cole aqui (sessão atual) — opcional")
    url = st.text_input("SUPABASE_URL", value="", placeholder="https://xxxx.supabase.co")
    key = st.text_input("SUPABASE_KEY", value="", type="password")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        save_local = st.checkbox("Salvar localmente (inseguro)", value=False)
    with col2:
        test = st.button("Testar", use_container_width=True)
    with col3:
        apply = st.button("Aplicar", use_container_width=True)

    if test or apply:
        if not url.strip() or not key.strip():
            st.warning("Preencha SUPABASE_URL e SUPABASE_KEY.")
            st.stop()

        st.session_state["SUPABASE_URL"] = url.strip()
        st.session_state["SUPABASE_KEY"] = key.strip()
        st.cache_resource.clear()
        st.cache_data.clear()
        bump_cache()

        ok, schema_ready, msg = supabase_healthcheck()
        if not ok:
            st.error(msg)
            st.stop()

        st.success("✅ Conectado na API.")
        if not schema_ready:
            st.warning("Conectou, mas as tabelas não existem (ou não estão expostas). Vá em **Setup (SQL)**.")
        if save_local:
            save_local_cfg({"SUPABASE_URL": url.strip(), "SUPABASE_KEY": key.strip()})
            st.info(f"Salvo em {LOCAL_CFG_FILE.name} (inseguro).")

        if apply:
            st.rerun()



def page_setup() -> None:
    st.subheader("🧱 Setup do Banco (Supabase)")
    st.write("1) Vá no Supabase → **SQL Editor** → cole e rode este SQL (1x).")
    st.code(SETUP_SQL, language="sql")
    st.write("2) Se estiver usando **ANON KEY** e ativou **RLS**, rode (opcional):")
    st.code(RLS_OPEN_SQL, language="sql")

    st.info(
        "Se aparecer erro de 'schema cache' após criar as tabelas: aguarde 10–30s e recarregue a página. "
        "Também confirme em Settings → API que **Exposed schemas** inclui **public**."
    )

    st.divider()

    with st.expander("🧨 Zerar a base (PERIGOSO)", expanded=False):
        st.warning(
            "Isso vai **apagar TODOS os dados**: veículos, vendas, despesas, caixa, retidas, pendências e settings. "
            "Use somente se você tiver certeza."
        )
        c1, c2 = st.columns(2)
        with c1:
            confirm = st.checkbox("Eu entendo que isso é IRREVERSÍVEL", value=False)
        with c2:
            typed = st.text_input("Digite ZERAR para confirmar", value="")

        if st.button("Zerar base agora", type="primary", use_container_width=True):
            if not confirm or typed.strip().upper() != "ZERAR":
                st.error("Para continuar, marque a confirmação e digite **ZERAR**.")
            else:
                try:
                    reset_database()
                    st.success("Base zerada com sucesso.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Falha ao zerar base: {e}")

def page_dashboard() -> None:
    st.subheader("📊 Visão Geral")

    date_ini = st.session_state.get("GLOBAL_DATE_INI")
    date_fim = st.session_state.get("GLOBAL_DATE_FIM")

    k = compute_kpis(date_ini, date_fim)

    alerts = []
    if safe_float(k.get("retained_pending")) > 0:
        alerts.append(f"Retidas pendentes: {brl(k.get('retained_pending'))}")
    if safe_float(k.get("payables_pending")) > 0:
        alerts.append(f"Pendências a pagar: {brl(k.get('payables_pending'))}")
    if alerts:
        st.warning(" ⚠️ " + " • ".join(alerts))

    render_kpi_metrics([
        ("Resultados", [
            ("💰 Caixa (saldo)", brl(k["cash_balance"]), "Saldo atual (inclui saldo inicial)"),
            ("📈 Vendas (total)", brl(k["total_sales_price"]), "Total no período"),
            ("🧾 Gastos totais", brl(k.get("total_expenses_including_purchases", 0.0)), "Inclui compras de veículos + despesas + garantias + comissões"),
            ("✅ Lucro (período)", brl(k["gross_profit"]), "Vendas - COGS (vendidos) - despesas"),
        ]),
        ("Gastos (detalhe)", [
            ("🚘 Compras de veículos", brl(k.get("purchases_period", 0.0)), "Custo de compra (somado às despesas)"),
            ("🧰 Peças", brl(k["parts"]), None),
            ("📣 Anúncios", brl(k["ads"]), None),
            ("🧾 Outras despesas", brl(k["other_exp"]), None),
        ]),
        ("Pendências / Operação", [
            ("🏷️ Retidas (total)", brl(k["retained_total"]), "Retidas registradas no período"),
            ("⏳ Retidas pendentes", brl(k["retained_pending"]), "Para receber"),
            ("⏳ A pagar (pendente)", brl(k["payables_pending"]), "Comissão/Garantia pendentes"),
            ("🚗 Em estoque", str(int(k.get("stock_count") or 0)), "Veículos não vendidos"),
        ]),
    ])

    st.divider()

    colg1, colg2 = st.columns([1.4, 1])

    with colg1:
        st.markdown("#### 📈 Evolução do Caixa (diário)")
        cash = list_cash()
        if cash.empty:
            st.info("Sem movimentos de caixa ainda.")
        else:
            c = cash.copy()
            c["mov_date"] = pd.to_datetime(c["mov_date"], errors="coerce")
            if date_ini or date_fim:
                c = apply_period_df(c, "mov_date", date_ini, date_fim)
            c["amount"] = c["amount"].apply(safe_float)
            c["in"] = c.apply(lambda r: r["amount"] if r.get("direction") == "IN" else 0.0, axis=1)
            c["out"] = c.apply(lambda r: r["amount"] if r.get("direction") == "OUT" else 0.0, axis=1)
            daily = c.groupby(c["mov_date"].dt.date, as_index=True)[["in", "out"]].sum().sort_index()
            opening = safe_float(get_setting("opening_balance") or 0)
            daily["saldo"] = opening + (daily["in"] - daily["out"]).cumsum()
            st.line_chart(daily[["saldo"]], height=260)

    with colg2:
        st.markdown("#### 🧮 Gastos por categoria (mensal)")
        exp = list_expenses()
        if exp.empty:
            base = pd.DataFrame(columns=["month", "category", "amount"])
        else:
            e = exp.copy()
            e["expense_date"] = pd.to_datetime(e["expense_date"], errors="coerce")
            if date_ini or date_fim:
                e = apply_period_df(e, "expense_date", date_ini, date_fim)
            e["amount"] = e["amount"].apply(safe_float)
            e["month"] = e["expense_date"].dt.to_period("M").astype(str)
            base = e[["month", "category", "amount"]].dropna()

        # Compras de veículos como categoria
        try:
            veh = list_vehicles(include_sold=True)
            if not veh.empty and "purchase_date" in veh.columns and "purchase_cost" in veh.columns:
                pv = veh.copy()
                pv["purchase_date"] = pd.to_datetime(pv["purchase_date"], errors="coerce")
                if date_ini or date_fim:
                    pv = apply_period_df(pv, "purchase_date", date_ini, date_fim)
                pv["amount"] = pv["purchase_cost"].apply(safe_float)
                pv["category"] = "COMPRA"
                pv["month"] = pv["purchase_date"].dt.to_period("M").astype(str)
                pv = pv[["month", "category", "amount"]].dropna()
                base = pd.concat([base, pv], ignore_index=True)
        except Exception:
            pass

        if base.empty:
            st.info("Sem gastos no período.")
        else:
            pivot = base.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0).sort_index()
            st.bar_chart(pivot, height=260)

    st.divider()

    st.markdown("#### 🏆 Lucro por veículo (Top 10)")
    sales = list_sales()
    vehicles = list_vehicles(include_sold=True, include_deleted=False)

    if sales.empty or vehicles.empty:
        st.info("Cadastre veículos e vendas para ver o lucro por veículo.")
        return

    s = sales.copy()
    v = vehicles.copy()

    if "sale_date" in s.columns:
        s = apply_period_df(s, "sale_date", date_ini, date_fim)

    s["sale_price"] = s["sale_price"].apply(safe_float)
    s["commission_amount"] = s["commission_amount"].apply(safe_float)
    s["warranty_cost"] = s["warranty_cost"].apply(safe_float)

    v["purchase_cost"] = v["purchase_cost"].apply(safe_float)

    exp = list_expenses()
    exp_by_vehicle = {}
    if not exp.empty and "vehicle_id" in exp.columns:
        ex = exp.copy()
        ex["amount"] = ex["amount"].apply(safe_float)
        if "expense_date" in ex.columns:
            ex = apply_period_df(ex, "expense_date", date_ini, date_fim)
        exp_by_vehicle = ex.groupby("vehicle_id")["amount"].sum().to_dict()

    rows = []
    for _, r in s.iterrows():
        vid = int(r.get("vehicle_id") or 0)
        vehicle = v.loc[v["id"] == vid]
        if vehicle.empty:
            continue
        sale_price = safe_float(r.get("sale_price"))
        comm = safe_float(r.get("commission_amount"))
        warr = safe_float(r.get("warranty_cost"))
        vc = safe_float(vehicle.iloc[0].get("purchase_cost"))
        model = str(r.get("vehicle_model") or vehicle.iloc[0].get("model") or f"#{vid}")
        plate = str(r.get("vehicle_plate") or vehicle.iloc[0].get("plate") or "-")
        expv = safe_float(exp_by_vehicle.get(vid, 0))
        profit = sale_price - vc - expv - comm - warr
        rows.append({"Veículo": f"{model} ({plate})", "Lucro": profit, "Venda": sale_price, "Custo": vc, "Despesas": expv})

    dfp = pd.DataFrame(rows)
    if dfp.empty:
        st.info("Sem dados suficientes para calcular lucro por veículo.")
        return

    dfp = dfp.sort_values("Lucro", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(
        dfp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lucro": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
            "Venda": st.column_config.NumberColumn("Venda", format="R$ %.2f"),
            "Custo": st.column_config.NumberColumn("Custo", format="R$ %.2f"),
            "Despesas": st.column_config.NumberColumn("Despesas", format="R$ %.2f"),
        },
        height=360,
    )

def page_vehicles() -> None:
    st.subheader("🚗 Veículos / Compras")

    # ----------------------------
    # Cadastro
    # ----------------------------
    with st.expander("➕ Cadastrar compra / veículo", expanded=True):
        with st.form("form_vehicle", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                model = st.text_input("Modelo / Descrição*", placeholder="Ex.: Onix LT 1.0")
                year = st.number_input("Ano", min_value=1900, max_value=2100, value=2020, step=1)
            with col2:
                plate = st.text_input("Placa", placeholder="ABC1D23")
                purchase_date = st.date_input("Data de compra", value=date.today(), format="DD/MM/YYYY")
            with col3:
                purchase_cost = st.number_input("Custo de compra (R$)", min_value=0.0, value=0.0, step=100.0)
                move_cash = st.checkbox("Movimentar caixa agora (saída)", value=True)

            notes = st.text_area("Observações", height=80)
            ok = st.form_submit_button("Salvar", use_container_width=True)

            if ok:
                if not model.strip():
                    st.error("Informe o **Modelo / Descrição**.")
                else:
                    add_vehicle(
                        model,
                        plate,
                        int(year) if year else None,
                        iso(purchase_date),
                        float(purchase_cost),
                        notes,
                        move_cash,
                    )
                    st.success("Veículo cadastrado.")

    st.divider()

    # ----------------------------
    # Dados
    # ----------------------------
    vehicles_all = list_vehicles(include_sold=True, include_deleted=True)
    if vehicles_all.empty:
        st.info("Nenhum veículo cadastrado.")
        return

    # ----------------------------
    # Editar / Excluir
    # ----------------------------
    with st.expander("✏️ Editar / Excluir veículo", expanded=False):
        options = {
            int(r["id"]): f"[{int(r['id'])}] {r.get('model') or '-'} ({r.get('plate') or '-'}) — "
                         f"{r.get('status') or '-'} — Custo: {brl(safe_float(r.get('purchase_cost')))}"
            for _, r in vehicles_all.iterrows()
        }

        selected_id = st.selectbox(
            "Selecione o veículo",
            options=list(options.keys()),
            format_func=lambda x: options.get(int(x), str(x)),
        )

        row = vehicles_all.loc[vehicles_all["id"] == selected_id].iloc[0].to_dict()

        def _to_date(v: Any) -> date:
            try:
                if v is None:
                    return date.today()
                s = str(v).strip()
                if not s or s.lower() == "nan":
                    return date.today()
                return datetime.fromisoformat(s).date()
            except Exception:
                return date.today()

        tab1, tab2 = st.tabs(["✏️ Editar", "🗑️ Excluir / Arquivar"])

        with tab1:
            with st.form("form_edit_vehicle", clear_on_submit=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    model_e = st.text_input("Modelo / Descrição*", value=str(row.get("model") or ""))
                    year_e = st.number_input(
                        "Ano",
                        min_value=1900,
                        max_value=2100,
                        value=int(row.get("year") or 2020),
                        step=1,
                    )
                with col2:
                    plate_e = st.text_input("Placa", value=str(row.get("plate") or ""))
                    purchase_date_e = st.date_input(
                        "Data de compra",
                        value=_to_date(row.get("purchase_date")),
                        format="DD/MM/YYYY",
                    )
                with col3:
                    purchase_cost_e = st.number_input(
                        "Custo de compra (R$)",
                        min_value=0.0,
                        value=float(safe_float(row.get("purchase_cost"))),
                        step=100.0,
                    )
                    status_e = st.selectbox(
                        "Status",
                        options=["EM_ESTOQUE", "VENDIDO", "EXCLUIDO"],
                        index=["EM_ESTOQUE", "VENDIDO", "EXCLUIDO"].index(str(row.get("status") or "EM_ESTOQUE"))
                        if str(row.get("status") or "EM_ESTOQUE") in ["EM_ESTOQUE", "VENDIDO", "EXCLUIDO"]
                        else 0,
                    )

                notes_e = st.text_area("Observações", value=str(row.get("notes") or ""), height=80)

                update_purchase_cash = st.checkbox(
                    "Atualizar (se existir) a movimentação de COMPRA no Caixa",
                    value=False,
                    help="Se você alterou data/custo da compra e já tinha lançamento no Caixa, isso atualiza 1 lançamento de COMPRA.",
                )

                ok_edit = st.form_submit_button("Salvar alterações", use_container_width=True)

                if ok_edit:
                    if not model_e.strip():
                        st.error("Informe o **Modelo / Descrição**.")
                    else:
                        update_vehicle(
                            int(selected_id),
                            model_e,
                            plate_e,
                            int(year_e) if year_e else None,
                            iso(purchase_date_e),
                            float(purchase_cost_e),
                            status_e,
                            notes_e,
                            update_purchase_cash_movement=update_purchase_cash,
                        )
                        st.success("Veículo atualizado.")
                        st.rerun()

        with tab2:
            counts = vehicle_related_counts(int(selected_id))

            st.markdown("**Impacto no histórico**")
            st.write(f"- Vendas vinculadas: **{counts['sales']}**")
            st.write(f"- Despesas vinculadas: **{counts['expenses']}**")
            st.write(f"- Movimentações de caixa vinculadas: **{counts['cash_movements']}**")
            st.write(f"- Retidas (via vendas): **{counts['retentions']}**")
            st.write(f"- Pendências (via vendas): **{counts['payables']}**")

            action = st.radio(
                "Ação",
                options=["Arquivar (recomendado)", "Excluir definitivamente (irreversível)"],
                index=0,
            )

            if action.startswith("Arquivar"):
                st.info("Arquivar mantém o histórico (vendas/despesas/caixa) e apenas esconde o veículo por padrão.")
                if st.button("Arquivar veículo", type="primary", use_container_width=True):
                    archive_vehicle(int(selected_id))
                    st.success("Veículo arquivado (status = EXCLUIDO).")
                    st.rerun()
            else:
                st.warning(
                    "⚠️ Excluir definitivamente pode apagar vendas vinculadas (cascade) e afetar KPIs/histórico. "
                    "Use somente se tiver certeza."
                )
                confirm = st.checkbox("Eu entendo que esta ação é IRREVERSÍVEL.", value=False)
                typed = st.text_input("Digite EXCLUIR para confirmar", value="")
                if st.button("Excluir definitivamente", type="primary", use_container_width=True):
                    if not confirm or typed.strip().upper() != "EXCLUIR":
                        st.error("Confirme a ação marcando a caixa e digitando **EXCLUIR**.")
                    else:
                        delete_vehicle_hard(int(selected_id))
                        st.success("Veículo excluído definitivamente.")
                        st.rerun()

    st.divider()

    # ----------------------------
    # Lista / Filtros
    # ----------------------------
    colf1, colf2, colf3 = st.columns([1, 1, 2])
    with colf1:
        show_sold = st.checkbox("Mostrar vendidos", value=True)
    with colf2:
        show_deleted = st.checkbox("Mostrar excluídos", value=False)
    with colf3:
        search = st.text_input("Buscar (modelo/placa)", value="").strip().lower()

    df = vehicles_all.copy()
    if not show_sold:
        df = df[df["status"] != "VENDIDO"]
    if not show_deleted:
        df = df[df["status"] != "EXCLUIDO"]
    if search:
        df = df[
            df["model"].fillna("").str.lower().str.contains(search)
            | df["plate"].fillna("").str.lower().str.contains(search)
        ]

    
    # ----------------------------
    # Custo total (Compra + Despesas do veículo)
    # ----------------------------
    exp_raw = sb_select("expenses", "vehicle_id,amount")
    exp_by_vehicle: Dict[int, float] = {}
    if not exp_raw.empty and "vehicle_id" in exp_raw.columns and "amount" in exp_raw.columns:
        ex = exp_raw.copy()
        ex["amount"] = ex["amount"].apply(safe_float)
        # remove despesas sem veículo
        ex = ex[ex["vehicle_id"].notna()]
        try:
            ex["vehicle_id"] = ex["vehicle_id"].astype(int)
        except Exception:
            pass
        exp_by_vehicle = ex.groupby("vehicle_id")["amount"].sum().to_dict()

    df["despesas_total"] = df["id"].apply(lambda vid: safe_float(exp_by_vehicle.get(int(vid), 0.0)))
    df["custo_total"] = df.apply(lambda r: safe_float(r.get("purchase_cost")) + safe_float(r.get("despesas_total")), axis=1)

    df_show = df.copy()
    if "purchase_cost" in df_show.columns:
        df_show["purchase_cost"] = df_show["purchase_cost"].map(lambda x: brl(safe_float(x)))
    if "despesas_total" in df_show.columns:
        df_show["despesas_total"] = df_show["despesas_total"].map(lambda x: brl(safe_float(x)))
    if "custo_total" in df_show.columns:
        df_show["custo_total"] = df_show["custo_total"].map(lambda x: brl(safe_float(x)))

    df_show = df_show.rename(
        columns={
            "id": "ID",
            "plate": "Placa",
            "model": "Modelo",
            "year": "Ano",
            "purchase_date": "Data compra",
            "purchase_cost": "Custo compra",
            "despesas_total": "Despesas",
            "custo_total": "Custo total",
            "status": "Status",
            "notes": "Obs",
        }
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)



def page_sales() -> None:
    st.subheader("🧾 Vendas")

    stock = list_vehicles(include_sold=False)
    if stock.empty:
        st.info("Cadastre um veículo em estoque para registrar venda.")
    else:
        with st.expander("➕ Registrar venda", expanded=True):
            options = {
                int(r["id"]): f"[{int(r['id'])}] {r['model']} ({r.get('plate') or '-'}) - Custo: {brl(safe_float(r.get('purchase_cost')))}"
                for _, r in stock.iterrows()
            }

            with st.form("form_sale", clear_on_submit=True):
                vehicle_id = st.selectbox(
                    "Veículo*",
                    options=list(options.keys()),
                    format_func=lambda x: options.get(x, str(x)),
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    sale_date = st.date_input("Data", value=date.today(), format="DD/MM/YYYY")
                    sale_price = st.number_input("Preço de venda (R$)", min_value=0.0, value=0.0, step=100.0)
                with col2:
                    received_amount = st.number_input("Recebido agora (R$)", min_value=0.0, value=0.0, step=100.0)
                    retained_amount = st.number_input("Retida (R$)", min_value=0.0, value=0.0, step=50.0)
                with col3:
                    commission_amount = st.number_input("Comissão (R$)", min_value=0.0, value=0.0, step=50.0)
                    warranty_cost = st.number_input("Garantia (custo) (R$)", min_value=0.0, value=0.0, step=50.0)

                buyer = st.text_input("Comprador (opcional)")
                notes = st.text_area("Observações", height=80)

                st.markdown("**Movimentação de caixa**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    move_cash_received = st.checkbox("Lançar recebimento no caixa", value=True)
                with c2:
                    commission_paid_now = st.checkbox("Comissão paga agora", value=False)
                with c3:
                    warranty_paid_now = st.checkbox("Garantia paga agora", value=False)

                ok = st.form_submit_button("Salvar venda", use_container_width=True)
                if ok:
                    if float(sale_price or 0) <= 0:
                        st.error("Informe o **preço de venda**.")
                    else:
                        add_sale(
                            int(vehicle_id),
                            iso(sale_date),
                            float(sale_price),
                            float(received_amount),
                            float(retained_amount),
                            float(commission_amount),
                            float(warranty_cost),
                            buyer,
                            notes,
                            move_cash_received,
                            commission_paid_now,
                            warranty_paid_now,
                        )
                        st.success("Venda registrada.")

    st.divider()

    sales = list_sales()
    if sales.empty:
        st.info("Nenhuma venda registrada.")
        return

    # Tabela (profissional)
    df_view = sales.copy().rename(
        columns={
            "id": "ID Venda",
            "sale_date": "Data",
            "vehicle_model": "Veículo",
            "vehicle_plate": "Placa",
            "sale_price": "Venda",
            "received_amount": "Recebido",
            "retained_amount": "Retida",
            "commission_amount": "Comissão",
            "warranty_cost": "Garantia",
            "buyer": "Comprador",
        }
    )

    date_ini = st.session_state.get("GLOBAL_DATE_INI")
    date_fim = st.session_state.get("GLOBAL_DATE_FIM")
    if "Data" in df_view.columns:
        df_view = apply_period_df(df_view, "Data", date_ini, date_fim)

    show_table_pro(
        df_view,
        title="📋 Vendas registradas",
        money_cols=["Venda", "Recebido", "Retida", "Comissão", "Garantia"],
        date_cols=["Data"],
        height=420,
    )

    st.divider()

    # ==========================================================
    # Editar / Excluir venda
    # ==========================================================
    with st.expander("✏️ Editar / Excluir venda", expanded=False):
        # opções de venda
        sales_opts = {}
        for _, r in sales.iterrows():
            sid = int(r["id"])
            label = f"[{sid}] {r.get('vehicle_model') or '-'} ({r.get('vehicle_plate') or '-'}) — {str(r.get('sale_date') or '')} — Venda: {brl(safe_float(r.get('sale_price')))}"
            sales_opts[sid] = label

        sale_id = st.selectbox(
            "Selecione a venda",
            options=list(sales_opts.keys()),
            format_func=lambda x: sales_opts.get(int(x), str(x)),
            key="edit_sale_select",
        )
        row = sales[sales["id"].astype(int) == int(sale_id)].iloc[0].to_dict()

        # snapshot de vínculos
        snap = sale_link_snapshot(int(sale_id))

        # mostrar resumo
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retidas (pend.)", str(snap.get("retentions_pending", 0)))
        c2.metric("Pendências (pend.)", str(snap.get("payables_pending", 0)))
        c3.metric("Lançamentos no caixa", str(snap.get("cash_total", 0)))
        c4.metric("Veículo ID", str(int(row.get("vehicle_id") or 0)))

        tab1, tab2 = st.tabs(["✏️ Editar", "🗑️ Excluir"])

        with tab1:
            # veículos para selecionar (inclui vendidos, para permitir manter o atual)
            vdf = list_vehicles(include_sold=True, include_deleted=False)
            v_opts = {}
            if not vdf.empty:
                for _, vr in vdf.iterrows():
                    vid = int(vr["id"])
                    v_opts[vid] = f"[{vid}] {vr.get('model') or '-'} ({vr.get('plate') or '-'}) — {vr.get('status') or ''}"
            else:
                v_opts[int(row.get("vehicle_id") or 0)] = f"[{int(row.get('vehicle_id') or 0)}] (veículo)"

            # defaults
            cur_sale_date = _parse_date_any(row.get("sale_date"))
            cur_vehicle_id = int(row.get("vehicle_id") or 0)

            # inferir estados padrão de comissao/garantia
            def _infer_mode(kind: str) -> str:
                kind = str(kind).upper()
                if kind == "COMISSAO":
                    if snap.get("payable_comissao_pending_id"):
                        return "PENDENTE"
                    if snap.get("cash_comissao_out_id"):
                        return "PAGO"
                    return "NAO_ALTERAR"
                if kind == "GARANTIA":
                    if snap.get("payable_garantia_pending_id"):
                        return "PENDENTE"
                    if snap.get("cash_garantia_out_id"):
                        return "PAGO"
                    return "NAO_ALTERAR"
                return "NAO_ALTERAR"

            default_comm = _infer_mode("COMISSAO")
            default_warr = _infer_mode("GARANTIA")
            default_ret = "PENDENTE" if snap.get("retention_pending_id") else "NAO_ALTERAR"

            with st.form("form_sale_edit", clear_on_submit=False):
                colA, colB, colC = st.columns(3)
                with colA:
                    vehicle_id_new = st.selectbox(
                        "Veículo*",
                        options=list(v_opts.keys()),
                        index=list(v_opts.keys()).index(cur_vehicle_id) if cur_vehicle_id in v_opts else 0,
                        format_func=lambda x: v_opts.get(int(x), str(x)),
                    )
                    sale_date_new = st.date_input("Data da venda", value=cur_sale_date, format="DD/MM/YYYY")
                    sale_price_new = st.number_input("Preço de venda (R$)", min_value=0.0, value=float(safe_float(row.get("sale_price"))), step=100.0)

                with colB:
                    received_amount_new = st.number_input("Recebido (R$)", min_value=0.0, value=float(safe_float(row.get("received_amount"))), step=100.0)
                    retained_amount_new = st.number_input("Retida (R$)", min_value=0.0, value=float(safe_float(row.get("retained_amount"))), step=50.0)
                    buyer_new = st.text_input("Comprador", value=str(row.get("buyer") or ""))

                with colC:
                    commission_amount_new = st.number_input("Comissão (R$)", min_value=0.0, value=float(safe_float(row.get("commission_amount"))), step=50.0)
                    warranty_cost_new = st.number_input("Garantia (custo) (R$)", min_value=0.0, value=float(safe_float(row.get("warranty_cost"))), step=50.0)
                    notes_new = st.text_area("Observações", value=str(row.get("notes") or ""), height=110)

                st.markdown("#### Sincronização (opcional)")
                cS1, cS2 = st.columns(2)
                with cS1:
                    sync_received_cash = st.checkbox("Sincronizar entrada de **RECEBIDO** no caixa (VENDA)", value=True)
                    received_cash_date = st.date_input("Data do lançamento do recebido", value=_parse_date_any(row.get("sale_date")), format="DD/MM/YYYY")
                    retention_mode_ui = st.radio(
                        "Retida",
                        options=["PENDENTE", "NAO_ALTERAR"],
                        index=0 if default_ret == "PENDENTE" else 1,
                        format_func=lambda x: "Pendente (criar/atualizar automaticamente)" if x == "PENDENTE" else "Não alterar (manual / já liberada)",
                    )
                with cS2:
                    commission_mode_ui = st.radio(
                        "Comissão",
                        options=["PENDENTE", "PAGO", "NAO_ALTERAR"],
                        index=["PENDENTE", "PAGO", "NAO_ALTERAR"].index(default_comm),
                        format_func=lambda x: (
                            "Pendente (criar/atualizar pendência)" if x == "PENDENTE" else
                            "Pago (lançar/atualizar no caixa e marcar como pago)" if x == "PAGO" else
                            "Não alterar (manual)"
                        ),
                    )
                    commission_pay_date = st.date_input("Data base comissão (pend./pag.)", value=_parse_date_any(row.get("sale_date")), format="DD/MM/YYYY", key="comm_pay_date")

                    warranty_mode_ui = st.radio(
                        "Garantia",
                        options=["PENDENTE", "PAGO", "NAO_ALTERAR"],
                        index=["PENDENTE", "PAGO", "NAO_ALTERAR"].index(default_warr),
                        format_func=lambda x: (
                            "Pendente (criar/atualizar pendência)" if x == "PENDENTE" else
                            "Pago (lançar/atualizar no caixa e marcar como pago)" if x == "PAGO" else
                            "Não alterar (manual)"
                        ),
                    )
                    warranty_pay_date = st.date_input("Data base garantia (pend./pag.)", value=_parse_date_any(row.get("sale_date")), format="DD/MM/YYYY", key="warr_pay_date")

                save = st.form_submit_button("Salvar alterações", use_container_width=True)

                if save:
                    if float(sale_price_new or 0) <= 0:
                        st.error("Informe o **preço de venda**.")
                    else:
                        # validações de veículo
                        vid_new = int(vehicle_id_new)
                        vid_old = int(cur_vehicle_id)

                        # não permitir editar para veículo excluído
                        vcheck = sb_select("vehicles", "id,status", filters=[("id", "eq", vid_new)], limit=1)
                        if not vcheck.empty and str(vcheck.loc[0, "status"]) == "EXCLUIDO":
                            st.error("Este veículo está **EXCLUÍDO/ARQUIVADO**. Selecione outro.")
                            st.stop()

                        # se trocar, verifica se já existe outra venda para esse veículo
                        if vid_new != vid_old:
                            other = sb_select("sales", "id", filters=[("vehicle_id", "eq", vid_new), ("id", "neq", int(sale_id))], limit=1)
                            if not other.empty:
                                st.error("Este veículo já possui uma venda registrada. Não é possível vincular esta venda a ele.")
                                st.stop()

                        try:
                            update_sale_record(
                                sale_id=int(sale_id),
                                vehicle_id=vid_new,
                                sale_date=iso(sale_date_new),
                                sale_price=float(sale_price_new),
                                received_amount=float(received_amount_new),
                                retained_amount=float(retained_amount_new),
                                commission_amount=float(commission_amount_new),
                                warranty_cost=float(warranty_cost_new),
                                buyer=buyer_new,
                                notes=notes_new,
                                sync_received_cash=bool(sync_received_cash),
                                received_cash_date=iso(received_cash_date),
                                retention_mode=str(retention_mode_ui),
                                commission_mode=str(commission_mode_ui),
                                commission_pay_date=iso(commission_pay_date),
                                warranty_mode=str(warranty_mode_ui),
                                warranty_pay_date=iso(warranty_pay_date),
                            )
                            st.success("Venda atualizada com sucesso.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao atualizar venda: {_err_text(e)}")

        with tab2:
            st.warning("Excluir uma venda pode afetar KPIs e o saldo do caixa. Use com cuidado.")
            cats = st.multiselect(
                "Ao excluir, também apagar lançamentos de caixa vinculados (por categoria):",
                options=["VENDA", "COMISSAO", "GARANTIA", "RETIDA"],
                default=["VENDA", "COMISSAO", "GARANTIA"],
                help="Por padrão, removemos os lançamentos mais 'derivados' da venda. 'RETIDA' normalmente é histórico de recebimento, apague só se tiver certeza.",
            )
            revert_vehicle = st.checkbox("Voltar veículo para EM_ESTOQUE (se não houver outra venda)", value=True)

            confirm = st.text_input("Digite EXCLUIR para confirmar", value="", key="sale_delete_confirm")
            do_delete = st.button("Excluir venda agora", type="primary", use_container_width=True)

            if do_delete:
                if confirm.strip().upper() != "EXCLUIR":
                    st.error("Confirmação inválida. Digite EXCLUIR para prosseguir.")
                else:
                    try:
                        delete_sale_record(
                            sale_id=int(sale_id),
                            delete_cash_categories=list(cats),
                            revert_vehicle_status=bool(revert_vehicle),
                        )
                        st.success("Venda excluída.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao excluir venda: {_err_text(e)}")


def page_expenses() -> None:
    st.subheader("🧰📣 Despesas (Peças / Anúncios / Mecânica / Elétrica / Pintura / Lanternagem)")

    vehicles = list_vehicles(include_sold=True)
    opts = {None: "— Geral (sem veículo) —"}
    if not vehicles.empty:
        for _, r in vehicles.iterrows():
            opts[int(r["id"])] = f"[{int(r['id'])}] {r['model']} ({r.get('plate') or '-'})"

    with st.expander("➕ Lançar despesa", expanded=True):
        with st.form("form_expense", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                expense_date = st.date_input("Data", value=date.today(), format="DD/MM/YYYY")
                category = st.selectbox("Categoria", list(EXPENSE_CATEGORIES.keys()))
            with col2:
                vehicle_id = st.selectbox("Vincular ao veículo", options=list(opts.keys()), format_func=lambda x: opts.get(x, str(x)))
                amount = st.number_input("Valor (R$)", min_value=0.0, value=0.0, step=50.0)
            with col3:
                move_cash = st.checkbox("Movimentar caixa agora (saída)", value=True)

            description = st.text_input("Descrição", placeholder="Ex.: Troca de pneus / Tráfego pago")
            ok = st.form_submit_button("Salvar despesa", use_container_width=True)

            if ok:
                if float(amount or 0) <= 0:
                    st.error("Informe um valor maior que zero.")
                else:
                    add_expense(None if vehicle_id is None else int(vehicle_id), iso(expense_date), category, description, float(amount), move_cash)
                    st.success("Despesa registrada.")

    st.divider()

    exp = list_expenses()
    if exp.empty:
        st.info("Nenhuma despesa registrada.")
        return

    
    # ----------------------------
    # Editar / Excluir
    # ----------------------------
    with st.expander("✏️ Editar / Excluir despesa", expanded=False):
        # opções
        exp_opts: dict[int, str] = {}
        for _, r in exp.iterrows():
            eid = int(r.get("id") or 0)
            d = str(r.get("expense_date") or "")
            cat = expense_category_label(str(r.get("category") or "")) if "category" in r else str(r.get("category") or "")
            val = brl(safe_float(r.get("amount")))
            veh = (r.get("vehicle_model") or "-")
            plc = (r.get("vehicle_plate") or "-")
            exp_opts[eid] = f"[{eid}] {d} — {cat} — {val} — {veh} ({plc})"

        if not exp_opts:
            st.caption("Nenhuma despesa para editar.")
        else:
            eid_sel = st.selectbox(
                "Selecione a despesa",
                options=list(exp_opts.keys()),
                format_func=lambda x: exp_opts.get(int(x), str(x)),
                key="edit_expense_select",
            )
            row = exp[exp["id"].astype(int) == int(eid_sel)].iloc[0].to_dict()

            tab1, tab2 = st.tabs(["✏️ Editar", "🗑️ Excluir"])

            def _to_date(v: Any) -> date:
                try:
                    if v is None:
                        return date.today()
                    s = str(v).strip()
                    if not s or s.lower() == "nan":
                        return date.today()
                    return datetime.fromisoformat(s).date()
                except Exception:
                    return date.today()

            with tab1:
                # veículos para (re)vincular
                vdf = list_vehicles(include_sold=True, include_deleted=True)
                v_opts2: dict[Optional[int], str] = {None: "— Geral (sem veículo) —"}
                if not vdf.empty:
                    for _, vr in vdf.iterrows():
                        vid = int(vr["id"])
                        v_opts2[vid] = f"[{vid}] {vr.get('model') or '-'} ({vr.get('plate') or '-'})"

                with st.form("form_edit_expense", clear_on_submit=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        exp_date_e = st.date_input("Data", value=_to_date(row.get("expense_date")), format="DD/MM/YYYY")
                        cat_e = st.selectbox("Categoria", list(EXPENSE_CATEGORIES.keys()),
                                             index=list(EXPENSE_CATEGORIES.keys()).index(expense_category_label(str(row.get("category") or "OUTROS")))
                                             if expense_category_label(str(row.get("category") or "OUTROS")) in list(EXPENSE_CATEGORIES.keys()) else 0)
                    with c2:
                        row_vid = row.get("vehicle_id")
                        try:
                            if row_vid is None or (isinstance(row_vid, float) and pd.isna(row_vid)):
                                row_vid_int = None
                            else:
                                row_vid_int = int(row_vid)
                        except Exception:
                            row_vid_int = None

                        veh_e = st.selectbox(
                            "Vincular ao veículo",
                            options=list(v_opts2.keys()),
                            format_func=lambda x: v_opts2.get(x, str(x)),
                            index=list(v_opts2.keys()).index(row_vid_int) if row_vid_int in v_opts2 else 0,
                        )
                        amt_e = st.number_input("Valor (R$)", min_value=0.0, value=float(safe_float(row.get("amount"))), step=50.0)
                    with c3:
                        upd_cash = st.checkbox("Atualizar (se existir) lançamento no Caixa", value=True,
                                               help="Se essa despesa foi lançada no Caixa, atualiza data/valor/descrição/categoria/veículo.")

                    desc_e = st.text_input("Descrição", value=str(row.get("description") or ""))

                    ok_edit = st.form_submit_button("Salvar alterações", use_container_width=True)
                    if ok_edit:
                        if float(amt_e or 0) <= 0:
                            st.error("Informe um valor maior que zero.")
                        else:
                            update_expense(int(eid_sel), None if veh_e is None else int(veh_e), iso(exp_date_e), cat_e, desc_e, float(amt_e), update_cash_movement=upd_cash)
                            st.success("Despesa atualizada.")
                            st.rerun()

            with tab2:
                st.warning("⚠️ Excluir é irreversível. Se houver lançamento no Caixa vinculado, ele também pode ser excluído.")
                del_cash = st.checkbox("Excluir também o lançamento no Caixa (se existir)", value=True)
                confirm = st.checkbox("Eu entendo que esta ação é IRREVERSÍVEL.", value=False)
                typed = st.text_input("Digite EXCLUIR para confirmar", value="", key="type_del_expense")
                if st.button("Excluir despesa", type="primary", use_container_width=True):
                    if not confirm or typed.strip().upper() != "EXCLUIR":
                        st.error("Confirme a ação marcando a caixa e digitando **EXCLUIR**.")
                    else:
                        delete_expense_hard(int(eid_sel), delete_cash=del_cash)
                        st.success("Despesa excluída.")
                        st.rerun()

    df = exp.copy()
    if "category" in df.columns:
        df["category"] = df["category"].map(lambda x: expense_category_label(str(x)))

    df = df.rename(columns={
        "id": "ID", "expense_date": "Data", "category": "Categoria", "description": "Descrição", "amount": "Valor",
        "vehicle_model": "Veículo", "vehicle_plate": "Placa"
    })

    show_table_pro(df, title="📋 Despesas (relatório)", money_cols=["Valor"], date_cols=["Data"], height=420)


def page_cash() -> None:
    st.subheader("💰 Caixa")

    opening = safe_float(get_setting("opening_balance") or 0)
    cash_df = list_cash()
    cash_in = safe_float(cash_df.loc[cash_df["direction"] == "IN", "amount"].sum()) if not cash_df.empty else 0.0
    cash_out = safe_float(cash_df.loc[cash_df["direction"] == "OUT", "amount"].sum()) if not cash_df.empty else 0.0
    saldo = opening + cash_in - cash_out

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saldo atual", brl(saldo))
    c2.metric("Entradas", brl(cash_in))
    c3.metric("Saídas", brl(cash_out))
    c4.metric("Saldo inicial", brl(opening))

    st.divider()

    # ----------------------------
    # Lançamento manual (opcional)
    # ----------------------------
    with st.expander("➕ Lançamento manual no Caixa", expanded=False):
        vehicles = list_vehicles(include_sold=True, include_deleted=True)
        vopts: dict[Optional[int], str] = {None: "— Sem veículo —"}
        if not vehicles.empty:
            for _, r in vehicles.iterrows():
                vid = int(r["id"])
                vopts[vid] = f"[{vid}] {r.get('model') or '-'} ({r.get('plate') or '-'})"

        with st.form("form_cash_manual", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mdate = st.date_input("Data", value=date.today(), format="DD/MM/YYYY")
                direction = st.selectbox("Tipo", options=["IN", "OUT"], format_func=lambda x: "Entrada" if x=="IN" else "Saída")
            with c2:
                category = st.text_input("Categoria", value="OUTROS")
                amount = st.number_input("Valor (R$)", min_value=0.0, value=0.0, step=50.0)
            with c3:
                veh = st.selectbox("Vincular ao veículo", options=list(vopts.keys()), format_func=lambda x: vopts.get(x, str(x)))

            desc = st.text_input("Descrição", value="")
            okm = st.form_submit_button("Salvar lançamento", use_container_width=True)
            if okm:
                if float(amount or 0) <= 0:
                    st.error("Informe um valor maior que zero.")
                else:
                    add_cash_manual(iso(mdate), direction, category, desc, float(amount), None if veh is None else int(veh))
                    st.success("Lançamento inserido.")
                    st.rerun()


    colp1, colp2 = st.columns(2)
    with colp1:
        st.markdown("#### ⏳ Retidas pendentes (para receber)")
        pend = list_retentions_pending()
        if pend.empty:
            st.caption("Nenhuma retida pendente.")
        else:
            show = pend.copy()
            # formatação via column_config (show_table_pro)
            show = show.rename(columns={"id": "ID", "created_date": "Criada em", "sale_date": "Data venda", "amount": "Valor",
                                        "vehicle_model": "Veículo", "vehicle_plate": "Placa"})
            cols = [c for c in ["ID", "Data venda", "Veículo", "Placa", "Valor", "Criada em"] if c in show.columns]
            st.dataframe(show[cols], use_container_width=True, hide_index=True)

            with st.form("form_release_retention", clear_on_submit=True):
                rid = st.selectbox("Selecione a retida para baixar", options=pend["id"].tolist())
                rdate = st.date_input("Data do recebimento", value=date.today(), format="DD/MM/YYYY")
                rdesc = st.text_input("Descrição", value="Liberação de retida")
                ok = st.form_submit_button("Baixar retida (entrada no caixa)", use_container_width=True)
                if ok:
                    release_retention(int(rid), iso(rdate), rdesc.strip() or "Liberação de retida")
                    st.success("Retida baixada.")

    with colp2:
        st.markdown("#### ⏳ Pendências (comissão/garantia) para pagar")
        pay = list_payables_pending()
        if pay.empty:
            st.caption("Nenhuma pendência.")
        else:
            show = pay.copy()
            # formatação via column_config (show_table_pro)
            show = show.rename(columns={"id": "ID", "kind": "Tipo", "created_date": "Criada em", "sale_date": "Data venda",
                                        "amount": "Valor", "vehicle_model": "Veículo", "vehicle_plate": "Placa"})
            cols = [c for c in ["ID", "Tipo", "Data venda", "Veículo", "Placa", "Valor", "Criada em"] if c in show.columns]
            st.dataframe(show[cols], use_container_width=True, hide_index=True)

            with st.form("form_pay_payable", clear_on_submit=True):
                pid = st.selectbox("Selecione a pendência para baixar", options=pay["id"].tolist())
                pdate = st.date_input("Data do pagamento", value=date.today(), format="DD/MM/YYYY")
                pdesc = st.text_input("Descrição", value="Baixa de pendência")
                ok = st.form_submit_button("Baixar pendência (saída no caixa)", use_container_width=True)
                if ok:
                    pay_payable(int(pid), iso(pdate), pdesc.strip() or "Baixa de pendência")
                    st.success("Pendência paga.")

    st.divider()

    
    # ----------------------------
    # Editar / Excluir lançamento
    # ----------------------------
    with st.expander("✏️ Editar / Excluir lançamento do Caixa", expanded=False):
        cash_raw = list_cash()
        if cash_raw.empty:
            st.caption("Nenhum lançamento para editar.")
        else:
            opts: dict[int, str] = {}
            for _, r in cash_raw.iterrows():
                cid = int(r.get("id") or 0)
                d = str(r.get("mov_date") or "")
                typ = "Entrada" if str(r.get("direction") or "").upper() == "IN" else "Saída"
                cat = str(r.get("category") or "")
                val = brl(safe_float(r.get("amount")))
                veh = str(r.get("vehicle_model") or "-")
                plc = str(r.get("vehicle_plate") or "-")
                opts[cid] = f"[{cid}] {d} — {typ} — {cat} — {val} — {veh} ({plc})"

            cid_sel = st.selectbox(
                "Selecione o lançamento",
                options=list(opts.keys()),
                format_func=lambda x: opts.get(int(x), str(x)),
                key="edit_cash_select",
            )
            row = cash_raw[cash_raw["id"].astype(int) == int(cid_sel)].iloc[0].to_dict()

            tab1, tab2 = st.tabs(["✏️ Editar", "🗑️ Excluir"])

            def _to_date(v: Any) -> date:
                try:
                    if v is None:
                        return date.today()
                    s = str(v).strip()
                    if not s or s.lower() == "nan":
                        return date.today()
                    return datetime.fromisoformat(s).date()
                except Exception:
                    return date.today()

            with tab1:
                vehicles = list_vehicles(include_sold=True, include_deleted=True)
                vopts: dict[Optional[int], str] = {None: "— Sem veículo —"}
                if not vehicles.empty:
                    for _, vr in vehicles.iterrows():
                        vid = int(vr["id"])
                        vopts[vid] = f"[{vid}] {vr.get('model') or '-'} ({vr.get('plate') or '-'})"

                cur_vid = row.get("vehicle_id")
                try:
                    cur_vid = int(cur_vid) if cur_vid is not None else None
                except Exception:
                    cur_vid = None

                with st.form("form_edit_cash", clear_on_submit=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        d_e = st.date_input("Data", value=_to_date(row.get("mov_date")), format="DD/MM/YYYY")
                        dir_e = st.selectbox("Tipo", options=["IN","OUT"], index=0 if str(row.get("direction") or "IN").upper()=="IN" else 1,
                                             format_func=lambda x: "Entrada" if x=="IN" else "Saída")
                    with c2:
                        cat_e = st.text_input("Categoria", value=str(row.get("category") or "OUTROS"))
                        amt_e = st.number_input("Valor (R$)", min_value=0.0, value=float(safe_float(row.get("amount"))), step=50.0)
                    with c3:
                        veh_e = st.selectbox("Vincular ao veículo", options=list(vopts.keys()), format_func=lambda x: vopts.get(x, str(x)),
                                             index=list(vopts.keys()).index(cur_vid) if cur_vid in vopts else 0)

                    desc_e = st.text_input("Descrição", value=str(row.get("description") or ""))

                    st.caption("⚠️ Observação: editar lançamentos gerados automaticamente (venda/compra/despesa/retida) pode desalinhar outros relatórios.")
                    ok = st.form_submit_button("Salvar alterações", use_container_width=True)
                    if ok:
                        if float(amt_e or 0) <= 0:
                            st.error("Informe um valor maior que zero.")
                        else:
                            update_cash_movement(int(cid_sel), iso(d_e), dir_e, cat_e, desc_e, float(amt_e), None if veh_e is None else int(veh_e))
                            st.success("Lançamento atualizado.")
                            st.rerun()

            with tab2:
                st.warning("⚠️ Excluir lançamento é irreversível.")
                confirm = st.checkbox("Eu entendo que esta ação é IRREVERSÍVEL.", value=False, key="conf_del_cash")
                typed = st.text_input("Digite EXCLUIR para confirmar", value="", key="type_del_cash")
                if st.button("Excluir lançamento", type="primary", use_container_width=True):
                    if not confirm or typed.strip().upper() != "EXCLUIR":
                        st.error("Confirme a ação marcando a caixa e digitando **EXCLUIR**.")
                    else:
                        delete_cash_movement_hard(int(cid_sel))
                        st.success("Lançamento excluído.")
                        st.rerun()


    st.markdown("#### 📒 Extrato / Lançamentos")
    if cash_df.empty:
        st.info("Nenhum lançamento.")
        return

    df = cash_df.copy()
    # formatação via column_config (show_table_pro)
    df["direction"] = df["direction"].map(lambda x: "Entrada" if x == "IN" else "Saída")
    df = df.rename(columns={"id": "ID", "mov_date": "Data", "direction": "Tipo", "category": "Categoria",
                            "description": "Descrição", "amount": "Valor", "vehicle_model": "Veículo", "vehicle_plate": "Placa"})
    show_table_pro(df, money_cols=['Valor','amount'], date_cols=['Data','mov_date'], height=420)


def page_reports() -> None:
    st.subheader("📤 Relatórios / Exportação (Excel)")

    vehicles = list_vehicles(include_sold=True)
    sales = list_sales()
    exp = list_expenses()
    cash = list_cash()
    ret_p = list_retentions_pending()
    pay_p = list_payables_pending()

    xbytes = to_excel_bytes({
        "veiculos": vehicles,
        "vendas": sales,
        "despesas": exp,
        "caixa": cash,
        "retidas_pendentes": ret_p,
        "pendencias_pagar": pay_p,
    })

    st.download_button(
        "⬇️ Baixar Excel (tudo)",
        data=xbytes,
        file_name=f"relatorio_agencia_carros_{date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def page_settings() -> None:
    st.subheader("⚙️ Configurações")
    current = safe_float(get_setting("opening_balance") or 0)
    new = st.number_input("Saldo inicial (R$)", value=float(current), step=100.0, min_value=0.0)
    if st.button("Salvar saldo inicial", use_container_width=True):
        set_setting("opening_balance", str(float(new)))
        st.success("Saldo inicial atualizado.")


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    ui_inject_global_style()

    url, key = get_supabase_cfg()
    if not url or not key:
        ui_header("Conecte ao Supabase para iniciar.", ["Configuração"])
        page_connect()
        return
    ok, schema_ready, status_msg = supabase_healthcheck()

    pills = [
        "Supabase OK" if ok else "Supabase ERRO",
        "Schema OK" if schema_ready else "Sem tabelas",
    ]
    ui_header(status_msg, pills)

    if not ok:
        st.error("Falha de autenticação/permissão (URL/KEY). Ajuste e recarregue.")
        page_connect()
        return

    with st.sidebar:
        st.markdown("### Navegação")

        default = 7 if not schema_ready else 0

        st.markdown("**📅 Período global**")
        dt_ini = st.date_input("Data inicial", value=None, format="DD/MM/YYYY", key="__g_dt_ini")
        dt_fim = st.date_input("Data final", value=None, format="DD/MM/YYYY", key="__g_dt_fim")
        st.session_state["GLOBAL_DATE_INI"] = iso(dt_ini) if isinstance(dt_ini, date) else None
        st.session_state["GLOBAL_DATE_FIM"] = iso(dt_fim) if isinstance(dt_fim, date) else None
        # st.caption("Aplica em Dashboard, Vendas, Despesas e Caixa.")

        

        pages = {
            "📊 Dashboard": page_dashboard,
            "🚗 Veículos/Compras": page_vehicles,
            "🧾 Vendas": page_sales,
            "🧰📣 Despesas": page_expenses,
            "💰 Caixa": page_cash,
            "📤 Relatórios": page_reports,
            "⚙️ Configurações": page_settings,
            "🧱 Setup (SQL)": page_setup,
        }

        choice = st.radio("Ir para:", list(pages.keys()), index=default, label_visibility="collapsed")

        st.divider()

        if st.button("🔄 Recarregar", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            bump_cache()
            st.rerun()

        if st.button("🔌 Trocar conexão", use_container_width=True):
            st.session_state.pop("SUPABASE_URL", None)
            st.session_state.pop("SUPABASE_KEY", None)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        st.caption(" ")
        st.caption("© Controle de Agência • Streamlit")

    if not schema_ready and choice != "🧱 Setup (SQL)":
        st.warning("As tabelas ainda não foram criadas/explicitas. Vá em **Setup (SQL)** e rode o SQL.")
        page_setup()
        return

    pages[choice]()




if __name__ == "__main__":
    main()
