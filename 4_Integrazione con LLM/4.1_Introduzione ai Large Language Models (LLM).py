import math

def calculate_training_resources(n_params, n_tokens, days, flops_per_gpu_real):
    """
    Stima le risorse necessarie per l'addestramento di un LLM.
    
    Args:
        n_params (float): Numero di parametri del modello (N).
        n_tokens (float): Numero di token nel dataset (D).
        days (int): Tempo limite per l'addestramento in giorni.
        flops_per_gpu_real (float): Performance reale della GPU in FLOP/s.
        
    Returns:
        dict: Risultati del calcolo (FLOPs totali, Secondi, Numero GPU).
    """
    # 1. Calcolo FLOPs totali (Formula di Chinchilla: C = 6 * N * D)
    total_flops = 6 * n_params * n_tokens
    
    # 2. Conversione tempo in secondi
    total_seconds = days * 24 * 3600
    
    # 3. Calcolo della potenza di calcolo totale necessaria (FLOP/s)
    required_throughput = total_flops / total_seconds
    
    # 4. Calcolo del numero di GPU necessarie
    num_gpus = required_throughput / flops_per_gpu_real
    
    return {
        "total_flops": total_flops,
        "total_seconds": total_seconds,
        "required_throughput": required_throughput,
        "num_gpus": math.ceil(num_gpus)  # Arrotondiamo per eccesso
    }

# --- INPUT RICHIESTI DAL PUNTO 3 ---
N = 70e9            # 70B parametri
D = 2e12            # 2T token
DAYS = 30           # 30 giorni
H100_REAL_FLOPS = 1e15  # 1 PetaFLOP/s reale (10^15)

# --- ESECUZIONE ---
results = calculate_training_resources(N, D, DAYS, H100_REAL_FLOPS)

print(f"--- LLM Training Budget Report ---")
print(f"Compute Totale (C):    {results['total_flops']:.2e} FLOPs")
print(f"Secondi Totali:        {results['total_seconds']:,} s")
print(f"Throughput Richiesto:  {results['required_throughput']:.2e} FLOP/s")
print(f"GPU H100 Necessarie:   {results['num_gpus']} unità")