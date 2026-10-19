import torch
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    Calcola la loss del VAE combinando MSE e Kullback-Leibler Divergence (KLD).
    """
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    logvar = torch.clamp(logvar, min=-10, max=10)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = MSE + beta * KLD
    return total_loss, MSE, KLD

def evaluate_model(model, data_loader, device='cpu', beta: float = 1.0):
    """
    Esegue la valutazione del modello restituendo le medie delle loss.
    Utile per il monitoraggio durante il training o nei test endpoint del backend.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_kld = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            recon_x, mu, logvar = model(x)
            loss, mse, kld = loss_function(recon_x, x, mu, logvar, beta)
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            
    num_batches = len(data_loader)
    return {
        'total_loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'kld': total_kld / num_batches
    }