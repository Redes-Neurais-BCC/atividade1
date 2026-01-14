import pandas as pd
import re
from pathlib import Path

# === Caminhos principais ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === Arquivos de entrada ===
GAMES_CSV = INTERIM_DIR / "dal-games-traduzido.csv"
ROSTER_CSV = INTERIM_DIR / "dal-roster-traduzido.csv"
PLAYERS_TOTAL_CSV = INTERIM_DIR / "dal-players-total-traduzido.csv"
PLAYERS_MEDIA_CSV = INTERIM_DIR / "dal-players-media-traduzido.csv"

print("Iniciando limpeza e integração dos dados do Dallas Mavericks 2024–26\n")

# === 1 - Jogos ===
print("\nProcessando jogos...")
games_df = pd.read_csv(GAMES_CSV)

# Converter "confronto" em mando de quadra (1 = casa, 0 = fora)
games_df["mando-de-jogo"] = games_df["confronto"].apply(lambda x: 1 if "vs." in str(x) else 0)

# Converter resultado em numérico (W=1, L=0)
games_df["resultado"] = games_df["resultado"].map({"W": 1, "L": 0, 1: 1, 0: 0})

# Remover hífens das datas
games_df["data-jogo"] = games_df["data-jogo"].astype(str).str.replace("-", "")

games_df.fillna(0, inplace=True)
games_df.to_csv(PROCESSED_DIR / "dallas_games_2024_25_26.csv", index=False)


# === 2 - Roster ===
print("\nProcessando roster...")
roster_df = pd.read_csv(ROSTER_CSV)

def convert_height_to_cm(height_str):
    match = re.match(r"(\d+)-(\d+)", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return round(feet * 30.48 + inches * 2.54, 1)
    return None

def convert_weight_to_kg(weight_str):
    try:
        return round(float(weight_str) * 0.453592, 1)
    except:
        return None

roster_df["altura-cm"] = roster_df["altura"].apply(convert_height_to_cm)
roster_df["peso-kg"] = roster_df["peso"].apply(convert_weight_to_kg)

position_map = {"G": 1, "F": 2, "F-C": 3, "C-F": 4, "C": 5}
roster_df["posicao-g-f-fc-cf-c"] = roster_df["posicao"].map(position_map).fillna(0)

colunas_roster = ["nome-jogador", "posicao-g-f-fc-cf-c", "altura-cm", "peso-kg", "idade"]
roster_clean = roster_df[colunas_roster].drop_duplicates(subset=["nome-jogador"], keep="first")
roster_clean.to_csv(INTERIM_DIR / "dal-roster-clean.csv", index=False)


# === 3 - Estatísticas Totais (somando 24–25 e 25–26) ===
print("\nProcessando estatísticas totais...")
players_total_df = pd.read_csv(PLAYERS_TOTAL_CSV)

colunas_somar = [
    "jogos-disputados", "minutos", "arremessos-convertidos", "arremessos-tentados",
    "triplos-convertidos", "triplos-tentados",
    "lances-livres-convertidos", "lances-livres-tentados",
    "rebotes-ofensivos", "rebotes-defensivos", "rebotes-totais",
    "assistencias", "erros", "roubos", "tocos", "faltas", "pontos"
]
colunas_somar = [c for c in colunas_somar if c in players_total_df.columns]

players_total_sum = players_total_df.groupby("nome-jogador")[colunas_somar].sum().reset_index()

# Calcular porcentagens totais
players_total_sum["porcentagem-arremessos_total"] = (
    players_total_sum["arremessos-convertidos"] / players_total_sum["arremessos-tentados"]
).fillna(0) * 100

players_total_sum["porcentagem-triplos_total"] = (
    players_total_sum["triplos-convertidos"] / players_total_sum["triplos-tentados"]
).fillna(0) * 100

players_total_sum["porcentagem-lances-livres_total"] = (
    players_total_sum["lances-livres-convertidos"] / players_total_sum["lances-livres-tentados"]
).fillna(0) * 100

# Adicionar sufixo _total
for col in colunas_somar:
    if col != "nome-jogador":
        players_total_sum.rename(columns={col: f"{col}_total"}, inplace=True)

players_total_sum.to_csv(INTERIM_DIR / "dal-players-total-somado.csv", index=False)


# === 4 - Estatísticas Médias (ponderadas) ===
print("\nProcessando estatísticas médias ponderadas...")
players_media_df = pd.read_csv(PLAYERS_MEDIA_CSV)

colunas_media = [
    "nome-jogador", "jogos-disputados", "minutos",
    "arremessos-convertidos", "arremessos-tentados", "triplos-convertidos", "triplos-tentados",
    "lances-livres-convertidos", "lances-livres-tentados",
    "rebotes-ofensivos", "rebotes-defensivos", "rebotes-totais",
    "assistencias", "erros", "roubos", "tocos", "faltas", "pontos"
]
players_media_df = players_media_df[colunas_media]

def weighted_avg(df, value_col, weight_col="jogos-disputados"):
    try:
        return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()
    except ZeroDivisionError:
        return 0

players_media_grouped = (
    players_media_df.groupby("nome-jogador")
    .apply(lambda g: pd.Series({
        col: weighted_avg(g, col) if col != "jogos-disputados" else g["jogos-disputados"].sum()
        for col in colunas_media if col != "nome-jogador"
    }))
    .reset_index()
)

# Calcular porcentagens médias
players_media_grouped["porcentagem-arremessos_media"] = (
    players_media_grouped["arremessos-convertidos"] / players_media_grouped["arremessos-tentados"]
).fillna(0) * 100

players_media_grouped["porcentagem-triplos_media"] = (
    players_media_grouped["triplos-convertidos"] / players_media_grouped["triplos-tentados"]
).fillna(0) * 100

players_media_grouped["porcentagem-lances-livres_media"] = (
    players_media_grouped["lances-livres-convertidos"] / players_media_grouped["lances-livres-tentados"]
).fillna(0) * 100

# Adicionar sufixo _media
for col in colunas_media:
    if col != "nome-jogador":
        players_media_grouped.rename(columns={col: f"{col}_media"}, inplace=True)

players_media_grouped.to_csv(INTERIM_DIR / "dal-players-media-ponderada.csv", index=False)


# === 5 - Merge Final ===
print("\nMesclando roster + médias + totais...")
merged = pd.merge(roster_clean, players_media_grouped, on="nome-jogador", how="left")
merged = pd.merge(merged, players_total_sum, on="nome-jogador", how="left")

merged.fillna(0, inplace=True)

# Eficiência simples
merged["eficiencia-simples"] = (
    merged["pontos_media"] / merged["arremessos-tentados_media"]
).replace([float("inf"), -float("inf")], 0)

# === 6 - Reorganizar colunas ===
colunas_finais = [
    "nome-jogador", "posicao-g-f-fc-cf-c", "altura-cm", "peso-kg", "idade",
    "jogos-disputados_media", "minutos_media", "arremessos-convertidos_media", "arremessos-tentados_media",
    "porcentagem-arremessos_media", "triplos-convertidos_media", "triplos-tentados_media",
    "porcentagem-triplos_media", "lances-livres-convertidos_media", "lances-livres-tentados_media",
    "porcentagem-lances-livres_media", "rebotes-ofensivos_media", "rebotes-defensivos_media",
    "rebotes-totais_media", "assistencias_media", "erros_media", "roubos_media", "tocos_media",
    "faltas_media", "pontos_media",
    "jogos-disputados_total", "minutos_total", "arremessos-convertidos_total", "arremessos-tentados_total",
    "porcentagem-arremessos_total", "triplos-convertidos_total", "triplos-tentados_total",
    "porcentagem-triplos_total", "lances-livres-convertidos_total", "lances-livres-tentados_total",
    "porcentagem-lances-livres_total", "rebotes-ofensivos_total", "rebotes-defensivos_total",
    "rebotes-totais_total", "assistencias_total", "erros_total", "roubos_total", "tocos_total",
    "faltas_total", "pontos_total", "eficiencia-simples"
]
merged = merged[colunas_finais]

# Miles Kelly zerado, removendo para nao distorcer os calculos:
merged = merged[merged["nome-jogador"] != "Miles Kelly"]

# === Salvar ===
merged.to_csv(PROCESSED_DIR / "dallas_players_2024_25_26.csv", index=False)
print(f"\n✅ Arquivo final salvo: dallas_players_2024_25_26.csv ({len(merged)} linhas)")
