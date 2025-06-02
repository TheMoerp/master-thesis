import numpy as np
import faiss

# Überprüfe, ob Faiss mit GPU-Unterstützung kompiliert wurde
print(f"Faiss Version: {faiss.__version__}")
print(f"GPU-Ressourcen verfügbar: {faiss.get_num_gpus()}")

try:
    # Erstelle einen einfachen Index für einen Test
    d = 64                           # Dimension
    nb = 100000                      # Datenbankgröße
    nq = 10                          # Anzahl der Abfragen
    np.random.seed(1234)             # Setzt einen festen Seed für Reproduzierbarkeit
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    # CPU-Index erstellen
    cpu_index = faiss.IndexFlatL2(d)  # L2-Distanz verwenden
    print(f"CPU-Index ist trainiert: {cpu_index.is_trained}")
    cpu_index.add(xb)                 # Vektoren zum Index hinzufügen
    print(f"Anzahl Vektoren im Index: {cpu_index.ntotal}")

    try:
        # Versuchen, einen GPU-Index zu erstellen
        print("\nVersuche GPU-Index zu erstellen...")
        res = faiss.StandardGpuResources()  # GPU-Ressourcen initialisieren
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("GPU-Index erfolgreich erstellt!")
        
        # Führe eine Suche auf dem GPU-Index durch
        k = 5                          # Anzahl der nächsten Nachbarn
        D, I = gpu_index.search(xq, k) # Distanz, Indizes
        print(f"\nErgebnis der Suche (Indizes der Top-{k} Nachbarn):")
        print(I[:5])  # Zeige die ersten 5 Abfrageergebnisse
        print("GPU-Funktionalität ist verfügbar!")
    except Exception as e:
        print(f"Fehler bei GPU-Test: {e}")
        print("GPU-Funktionalität ist NICHT verfügbar in der installierten Version.")
        print("Die installierte Faiss-Version unterstützt keine GPU-Operationen.")
        
except Exception as e:
    print(f"Allgemeiner Fehler: {e}") 