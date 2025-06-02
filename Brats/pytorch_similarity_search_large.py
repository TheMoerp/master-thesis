import torch
import time
import numpy as np

# Einstellungen mit größeren Datenmengen
dim = 128            # Dimension der Vektoren
num_vectors = 1000000  # 1 Million Vektoren in der Datenbank
num_queries = 1000     # 1000 Abfragevektoren
k = 10                 # Top-k Nachbarn suchen

print(f"Suche unter {num_vectors} Vektoren mit {num_queries} Abfragen...")

# Vektoren generieren (zufällig)
print("Generiere Vektoren...")
np.random.seed(1234)
vectors_np = np.random.random((num_vectors, dim)).astype('float32')
queries_np = np.random.random((num_queries, dim)).astype('float32')

# Funktion für Brute-Force-Suche mit PyTorch
def pytorch_search(queries, vectors, k, device_str="cpu"):
    device = torch.device(device_str)
    
    # Konvertieren und auf Gerät verschieben
    start_transfer = time.time()
    vectors_t = torch.from_numpy(vectors).to(device)
    queries_t = torch.from_numpy(queries).to(device)
    
    # Normierung der Vektoren für Cosine-Ähnlichkeit
    vectors_t = torch.nn.functional.normalize(vectors_t, dim=1)
    queries_t = torch.nn.functional.normalize(queries_t, dim=1)
    transfer_time = time.time() - start_transfer
    
    print(f"Datentransfer und Vorverarbeitung auf {device_str}: {transfer_time:.4f} Sekunden")
    
    # Eigentliche Berechnung
    start_compute = time.time()
    
    # Batch-Verarbeitung für große Datensätze
    batch_size = 10000  # Anpassen je nach GPU-Speicher
    all_top_similarities = []
    all_top_indices = []
    
    for i in range(0, queries_t.shape[0], batch_size):
        batch_queries = queries_t[i:i+batch_size]
        batch_similarities = torch.matmul(batch_queries, vectors_t.t())
        batch_top_similarities, batch_top_indices = torch.topk(batch_similarities, k=k, dim=1)
        all_top_similarities.append(batch_top_similarities.cpu())
        all_top_indices.append(batch_top_indices.cpu())
    
    # Ergebnisse zusammenführen
    top_similarities = torch.cat(all_top_similarities, dim=0).numpy()
    top_indices = torch.cat(all_top_indices, dim=0).numpy()
    
    compute_time = time.time() - start_compute
    total_time = transfer_time + compute_time
    
    return top_indices, top_similarities, compute_time, total_time

# GPU-Suche durchführen
print("\nFühre Suche auf GPU durch...")
if torch.cuda.is_available():
    gpu_indices, gpu_similarities, gpu_compute_time, gpu_total_time = pytorch_search(
        queries_np, vectors_np, k, "cuda"
    )
    
    print(f"GPU Berechnungszeit: {gpu_compute_time:.4f} Sekunden")
    print(f"GPU Gesamtzeit (inkl. Transfer): {gpu_total_time:.4f} Sekunden")
    print(f"Durchschnittliche Zeit pro Abfrage: {(gpu_compute_time / num_queries) * 1000:.4f} ms")
else:
    print("GPU nicht verfügbar!")

# CPU-Suche durchführen
print("\nFühre Suche auf CPU durch...")
cpu_indices, cpu_similarities, cpu_compute_time, cpu_total_time = pytorch_search(
    queries_np, vectors_np, k, "cpu"
)

print(f"CPU Berechnungszeit: {cpu_compute_time:.4f} Sekunden")
print(f"CPU Gesamtzeit: {cpu_total_time:.4f} Sekunden")
print(f"Durchschnittliche CPU-Zeit pro Abfrage: {(cpu_compute_time / num_queries) * 1000:.4f} ms")

# Beschleunigung berechnen
if torch.cuda.is_available():
    print(f"\nBeschleunigung (nur Berechnung): {cpu_compute_time / gpu_compute_time:.2f}x")
    print(f"Beschleunigung (gesamt): {cpu_total_time / gpu_total_time:.2f}x")

# Überprüfe, ob die Ergebnisse übereinstimmen
if torch.cuda.is_available():
    match_count = np.sum(np.all(gpu_indices[:5] == cpu_indices[:5], axis=1))
    print(f"\nÜbereinstimmung der ersten 5 Ergebnisse: {match_count}/5 ({match_count/5*100:.1f}%)")
    
    # Beispiel-Ergebnisse anzeigen
    print("\nBeispiel-Ergebnisse (GPU) für die ersten 3 Abfragen:")
    for i in range(min(3, num_queries)):
        print(f"Abfrage {i}:")
        for j in range(min(5, k)):
            print(f"  Top {j+1}: Index {gpu_indices[i, j]}, Ähnlichkeit: {gpu_similarities[i, j]:.4f}") 