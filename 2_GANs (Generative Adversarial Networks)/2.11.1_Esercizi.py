# Esercizio
# Implementa degli ulteriori step di crescita nel codice proposto, passando da una risoluzione 16x16 ad una 64x64
# implementa all'interno del grow il dimezzamento dei channel quando la risoluzione raggiunge 64x64

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

class SimpleGenerator(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()
        self.z_dim = z_dim
        
        # Inizio: 4x4 con 512 canali
        self.input_layer = nn.Linear(z_dim, 512 * 4 * 4)
        
        self.blocks = nn.ModuleList([
            SimpleBlock(512, 512)
        ])
        
        self.to_rgb_layers = nn.ModuleList([
            nn.Conv2d(512, 3, kernel_size=1)
        ])
        
        self.current_res_step = 0 
        self.alpha = 1.0

    def grow(self):
        self.current_res_step += 1
        new_res = 4 * (2 ** self.current_res_step)
        
        # Recuperiamo il numero di canali in uscita dall'ultimo blocco esistente
        # Accediamo all'ultimo blocco della lista per sapere quanti canali "ereditiamo"
        prev_channels = self.blocks[-1].conv2.out_channels
        new_channels = prev_channels
        
        # LOGICA RICHIESTA: Dimezzamento canali a 64x64
        # Se la nuova risoluzione è 64 o superiore, dimezziamo i canali
        if new_res >= 64:
            new_channels = prev_channels // 2
            
        print(f"\n[GROWTH] Passaggio a {new_res}x{new_res}. Canali: {prev_channels} -> {new_channels}")
        
        # 1. Aggiungiamo il nuovo blocco (gestisce il cambio canali se necessario)
        self.blocks.append(SimpleBlock(prev_channels, new_channels))
        
        # 2. Aggiungiamo il nuovo strato ToRGB per la nuova risoluzione
        self.to_rgb_layers.append(nn.Conv2d(new_channels, 3, kernel_size=1))
        
        # Reset alpha per iniziare il fading del nuovo livello
        self.alpha = 0.0

    def forward(self, z):
        # 1. Proiezione iniziale 4x4
        out = self.input_layer(z)
        out = out.view(-1, 512, 4, 4)
        
        # 2. Passaggio attraverso i blocchi stabilizzati
        # Si ferma prima dell'ultimo blocco se siamo in fase di fading
        for i in range(self.current_res_step):
            out = self.blocks[i](out)
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            
        # 3. Gestione Fading
        if self.current_res_step == 0:
            out = self.blocks[0](out)
            return self.to_rgb_layers[0](out)
        
        # Ramo Vecchio (Upsample della risoluzione precedente)
        # Usiamo il ToRGB del livello precedente
        old_rgb = self.to_rgb_layers[self.current_res_step - 1](out)
        
        # Ramo Nuovo (Dettagli della nuova risoluzione)
        new_features = self.blocks[self.current_res_step](out)
        new_rgb = self.to_rgb_layers[self.current_res_step](new_features)
        
        # Mix Alpha
        return (1 - self.alpha) * old_rgb + self.alpha * new_rgb

# --- SIMULAZIONE DELLA CRESCITA FINO A 64x64 ---

def run_growth_simulation():
    torch.manual_seed(42)
    gen = SimpleGenerator(512)
    z = torch.randn(1, 512)

    # Step 0: 4x4 (Iniziale)
    print(f"Step 0 (4x4) - Shape: {gen(z).shape}")

    # Step 1: Crescita a 8x8
    gen.grow()
    gen.alpha = 1.0
    print(f"Step 1 (8x8) - Shape: {gen(z).shape}")

    # Step 2: Crescita a 16x16
    gen.grow()
    gen.alpha = 1.0
    print(f"Step 2 (16x16) - Shape: {gen(z).shape}")

    # --- NUOVI STEP RICHIESTI ---

    # Step 3: Crescita a 32x32 (Canali ancora a 512)
    gen.grow()
    gen.alpha = 1.0
    print(f"Step 3 (32x32) - Shape: {gen(z).shape}")

    # Step 4: Crescita a 64x64 (DIMEZZAMENTO CANALI ATTIVATO)
    gen.grow()
    gen.alpha = 1.0
    print(f"Step 4 (64x64) - Shape: {gen(z).shape}")

if __name__ == "__main__":
    run_growth_simulation()