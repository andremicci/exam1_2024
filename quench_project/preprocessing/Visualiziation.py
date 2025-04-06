import random
import numpy as np
import matplotlib.pyplot as plt


def visualize_sequence(data,quenched=False,debug=False):
    """
    Visualizza una sequenza di mappa di calore a tempo per tempo.
    sequence: numpy array
        La sequenza di mappe di calore (formato: (24, 15, 15)).
    title: str
        Titolo della visualizzazione.
    """
    if quenched:
         filtered_data = data[data['label'] == 1]
    else:
        filtered_data = data[data['label'] == 0]
  
    sampled_data = data.iloc[random.choice(filtered_data.index)]
    sequence = sampled_data['sequence']
    quench = sampled_data['quench']
    
    if debug:
        print("Quench: ", quench)
    

        # Controlla che la forma sia corretta
    if sequence.shape != (24, 15, 15):
        print("La sequenza non ha la forma corretta. Deve essere (24, 15, 15).")
        return

    # Crea un grafico con 24 subplot (uno per ogni passo temporale)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.ravel()  # Per avere un array 1D dei subplot
    num_quenches = 0
    # Visualizza ogni mappa di calore
    for t in range(24):
        ax = axes[t]
        ax.imshow(sequence[t], cmap='hot', interpolation='nearest')
        ax.set_title(f'Time step {t+1}')
        ax.axis('off') 
        if quenched:
            
            for element in quench:
                if element['step'] == t+1:
                    pixel = element['pixel']  # Estrai le coordinate del quench
                    x, y = pixel[0], pixel[1]
                    ax.scatter(y, x, color='green', s=800, edgecolor='black', marker='X')
                    ax.set_facecolor('red') 
                    num_quenches +=1

    # Aggiungi un titolo generale
    fig.suptitle(f"Quenched Sequence,number of quenches {num_quenches} " if quenched else "Unquenched Sequence", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Per non sovrapporre il titolo
    plt.show()

def plot_temperature_histograms(data):
    """
    Calcola le temperature medie per ogni timestep e crea due istogrammi
    (uno per label=0 e uno per label=1), unendo tutti i valori delle sequenze.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con colonne 'sequence' (np.array di shape (24, 15, 15)) e 'label'
    """

    def extract_timestep_means(sequences):
        means = []
        for seq in sequences:
           
            timestep_means = np.mean(seq, axis=(1,2))  # media su ogni timestep
            means.extend(timestep_means)  # aggiungi i 24 valori
            
        return np.array(means)

    # Separa i dati
    quenched_data = data[data['label'] == 1]
    unquenched_data = data[data['label'] == 0]

    # Estrai le medie
    means_quenched = extract_timestep_means(quenched_data['sequence'])
    means_unquenched = extract_timestep_means(unquenched_data['sequence'])

    # Plot
    plt.figure(figsize=(8, 4))
    bins = np.linspace(
        min(min(means_quenched), min(means_unquenched)),
        max(max(means_quenched), max(means_unquenched)),
        40
    )

    plt.hist(means_quenched, bins=bins, alpha=0.6, label='Quenched', color='red', density=True)
    plt.hist(means_unquenched, bins=bins, alpha=0.6, label='Unquenched', color='blue', density=True)
    
    plt.xlabel("Temperature at Timesteps (Averaged over 15x15 grid)")
    plt.ylabel("Normalized Frequency")
    plt.title("Histogram of Timestep-wise Temperatures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_label_distribution_pie(data):
    """
    Crea un grafico a torta per visualizzare la distribuzione delle sequenze per ogni label (0 e 1),
    con il numero di sequenze mostrato all'interno delle fette.
    
    Parameters:
    data (DataFrame): Il dataset che contiene una colonna 'label' con valori 0 e 1.
    """
    # Conta il numero di sequenze per ogni label
    label_counts = data['label'].value_counts()

    # Crea il grafico a torta
    plt.figure(figsize=(8,6))
    wedges, texts, autotexts = plt.pie(label_counts, 
                                       labels=label_counts.index, 
                                       autopct='%1.1f%%', 
                                       startangle=90, 
                                       colors=['#66b3ff', '#ff9966'], 
                                       explode=(0.1, 0),  # Aggiungi effetto di esplosione per la prima fetta
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})  # Aggiungi bordi alle fette

    # Migliora l'aspetto dei testi
    for autotext in autotexts:
        autotext.set(fontsize=14, fontweight='bold', color='white')  # Cambia stile del testo delle percentuali

    # Aggiungi numeri al centro delle fette
    for i, count in enumerate(label_counts):
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2
        x = 0.5 * wedges[i].r * np.cos(np.deg2rad(angle))  # Calcola la posizione x del testo
        y = 0.5 * wedges[i].r * np.sin(np.deg2rad(angle))  # Calcola la posizione y del testo
        plt.text(x, y, f'{count}', ha='center', va='center', fontsize=12, color='black', fontweight='bold')

    # Aggiungi il titolo
    plt.title("Distribuzione delle sequenze per ogni label (0 = senza quench, 1 = con quench)", fontsize=16, fontweight='bold')

    # Aggiungi la legenda
    plt.legend(wedges, 
               labels=[f'Label 0: {label_counts[0]} sequenze', f'Label 1: {label_counts[1]} sequenze'], 
               title="Labels", 
               loc="best", 
               fontsize=12)

    # Assicura che il grafico sia rotondo
    plt.axis('equal')
    plt.show()
