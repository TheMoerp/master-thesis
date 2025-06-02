import torch
import time
import numpy as np

# Einstellungen
dim = 128         # Dimension der Vektoren
num_vectors = 100000  # Anzahl der Vektoren in der Datenbank
num_queries = 100     # Anzahl der Abfragevektoren
k = 10                # Top-k Nachbarn suchen

# Vektoren generieren (zufällig)
np.random.seed(1234)
vectors_np = np.random.random((num_vectors, dim)).astype('float32')
queries_np = np.random.random((num_queries, dim)).astype('float32')

# Numpy zu PyTorch Tensoren konvertieren und auf GPU verschieben, wenn verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

vectors = torch.from_numpy(vectors_np).to(device)
queries = torch.from_numpy(queries_np).to(device)

# Normierung der Vektoren für Cosine-Ähnlichkeit (optional)
vectors = torch.nn.functional.normalize(vectors, dim=1)
queries = torch.nn.functional.normalize(queries, dim=1)

# Funktion für Brute-Force-Suche mit PyTorch
def pytorch_search(queries, vectors, k):
    start_time = time.time()
    
    # Berechne Ähnlichkeiten (dot product für normalisierte Vektoren = Cosine-Ähnlichkeit)
    # [num_queries, num_vectors]
    similarities = torch.matmul(queries, vectors.t())
    
    # Finde Top-k Ähnlichkeiten
    # torch.topk gibt zurück: (Werte, Indizes)
    top_similarities, top_indices = torch.topk(similarities, k=k, dim=1)
    
    end_time = time.time()
    return top_indices.cpu().numpy(), top_similarities.cpu().numpy(), end_time - start_time

# Suche durchführen
top_indices, top_similarities, elapsed_time = pytorch_search(queries, vectors, k)

# Ergebnisse anzeigen
print(f"Suche abgeschlossen in {elapsed_time:.4f} Sekunden")
print(f"Durchschnittliche Zeit pro Abfrage: {(elapsed_time / num_queries) * 1000:.4f} ms")

# Beispiel für die ersten 3 Abfragen und ihre Top-5 Ergebnisse anzeigen
print("\nBeispiel-Ergebnisse für die ersten 3 Abfragen:")
for i in range(min(3, num_queries)):
    print(f"Abfrage {i}:")
    for j in range(min(5, k)):
        print(f"  Top {j+1}: Index {top_indices[i, j]}, Ähnlichkeit: {top_similarities[i, j]:.4f}")

# Vergleich mit CPU-Verarbeitung
print("\nZum Vergleich: Durchführung derselben Suche auf CPU...")
vectors_cpu = vectors.cpu()
queries_cpu = queries.cpu()
top_indices_cpu, top_similarities_cpu, elapsed_time_cpu = pytorch_search(queries_cpu, vectors_cpu, k)

print(f"CPU-Suche abgeschlossen in {elapsed_time_cpu:.4f} Sekunden")
print(f"Durchschnittliche CPU-Zeit pro Abfrage: {(elapsed_time_cpu / num_queries) * 1000:.4f} ms")
print(f"Beschleunigung durch GPU: {elapsed_time_cpu / elapsed_time:.2f}x") 