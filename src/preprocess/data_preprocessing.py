import pandas as pd
from pathlib import Path
import json

# Caminhos principais
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos dos CSVs originais
games_csv = DATA_DIR / "original/dal_games_2024_25_26.csv"
roster_csv = DATA_DIR / "original/dal_roster_2024_25_26.csv"
player_stats_media_csv = DATA_DIR / "original/dal_players_season_stats_media_2024_25_26.csv"
player_stats_total_csv = DATA_DIR / "original/dal_players_season_stats_acumulado_2024_25_26.csv"

# Caminho do mapeamento de colunas
mapping_json = ROOT_DIR / "data" / "mappings" / "column_mapping.json"

print("Carregando mapeamento de colunas...")
with open(mapping_json, "r", encoding="utf-8") as f:
    column_mapping = json.load(f)

def rename_columns(df, mapping):
    """Renomeia colunas com base no dicionário de tradução"""
    renamed = {col: mapping.get(col, col) for col in df.columns}
    return df.rename(columns=renamed)

# --- Processar cada CSV ---
print("🔹 Processando jogos...")
games_df = pd.read_csv(games_csv)
games_df = rename_columns(games_df, column_mapping)

print("Processando roster...")
roster_df = pd.read_csv(roster_csv)
roster_df = rename_columns(roster_df, column_mapping)

print("Processando estatísticas dos jogadores (médias)...")
players_media_df = pd.read_csv(player_stats_media_csv)
players_media_df = rename_columns(players_media_df, column_mapping)

print("Processando estatísticas dos jogadores (acumulado)...")
players_total_df = pd.read_csv(player_stats_total_csv)
players_total_df = rename_columns(players_total_df, column_mapping)

# --- Salvar resultados intermediários ---
games_df.to_csv(INTERIM_DIR / "dal-games-traduzido.csv", index=False)
roster_df.to_csv(INTERIM_DIR / "dal-roster-traduzido.csv", index=False)
players_media_df.to_csv(INTERIM_DIR / "dal-players-media-traduzido.csv", index=False)
players_total_df.to_csv(INTERIM_DIR / "dal-players-total-traduzido.csv", index=False)

print("\nArquivos traduzidos salvos em data/processed/")
