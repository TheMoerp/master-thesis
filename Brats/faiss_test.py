import numpy as np
import faiss

def test_gpu_faiss(d=64, nb=1000, nq=5, k=5):
    # GPU-Ressourcen initialisieren
    res = faiss.StandardGpuResources()
    # CPU-Index erstellen
    cpu_index = faiss.IndexFlatL2(d)
    # Auf GPU 0 verschieben
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # Zufallsdaten
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    # Daten hinzufügen und suchen
    gpu_index.add(xb)
    D, I = gpu_index.search(xq, k)

    print("Indexgröße:", gpu_index.ntotal)
    print(f"Top-{k} Nachbarn für {nq} Queries:")
    print("Indizes:\n", I)
    print("Distanzen:\n", D)

if __name__ == "__main__":
    print("Starte FAISS-GPU Test…")
    test_gpu_faiss()
