import random
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

