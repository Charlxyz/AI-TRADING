import json

# Fichier d'entrée
input_file = "BTCUSDT_5m_1m.json"

# Fichier de sortie
output_file = "BTCUSDT_5m_1m_min.json"

# Lecture
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Écriture sans indentation
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

print(f"Fichier enregistré : {output_file}")