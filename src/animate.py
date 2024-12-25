import numpy as np 
import matplotlib.pyplot as plt
from celluloid import Camera


def create_visualization(Plista,t):

    mapa = np.zeros((18,18)) # silent is zero 
    
    k=0
    j=0
    for i in Plista[t]:
        
        if(k%18 ==0 and k!=0):
            k = 0
            j += 1
        if i == 0:
            mapa[k][j] = 0.5    # zero is 0.5
        else:
            mapa[k][j] = i    # 1 is 1, -1 is -1
        k += 1
    
    return mapa



def create_party_visualization(Plista,t):

    mapa = np.zeros((5,5)) # silent is zero 
    
    k=0
    j=0
    for i in Plista[t]:
        
        if(k%5 ==0 and k!=0):
            k = 0
            j += 1
        if i == 0:
            mapa[k][j] = 0.5    # zero is 0.5
        else:
            mapa[k][j] = i    # 1 is 1, -1 is -1
        k += 1
    
    return mapa




def animate_unordered_(Plista):

    for i in range(5,120,5):
        #plt.imshow(np.sort( create_visualization(Plista,i)))
        plt.imshow( create_visualization(Plista,i), cmap = 'magma')
        plt.savefig('partisanloc' + str(i)+'.png')





from matplotlib.patches import Patch
from celluloid import Camera
import numpy as np

def create_legend():
    # Define colors based on the 'magma' colormap for each label.
    cmap = plt.get_cmap('magma')
    silent_color = cmap((0 - (-1)) / 2)  # Maps to 0.5 in the colormap
    neutral_color = cmap((0.5 - (-1)) / 2)  # Maps to 0.75 in the colormap
    infavour_color = cmap((1 - (-1)) / 2)  # Maps to 1 in the colormap
    against_color = cmap(((-1) - (-1)) / 2)  # Maps to 0 in the colormap

    # Create list of Legend elements
    legend_elements = [
        Patch(facecolor=silent_color, edgecolor='k', label='Silent'),
        Patch(facecolor=neutral_color, edgecolor='k', label='Neutral'),
        Patch(facecolor=infavour_color, edgecolor='k', label='In favour'),
        Patch(facecolor=against_color, edgecolor='k', label='Against')
    ]

    return legend_elements

def animate_ordered(Plista, times):
    fig = plt.figure()
    camera = Camera(fig)

    for i, _time in enumerate(times):
        plt.imshow(np.sort(create_visualization(Plista, _time)), cmap='magma', vmin=-1, vmax=1)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        camera.snap()

    # Add legend to the plot
    legend_elements = create_legend()
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    animation = camera.animate()
    animation.save('sortedmodeldynamic.gif', writer='pillow')



def animate_unordered(Plista, times):
    fig = plt.figure()
    camera = Camera(fig)

    for i, _time in enumerate(times):
        plt.imshow(create_visualization(Plista, _time), cmap='magma', vmin=-1, vmax=1)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        camera.snap()

    # Add legend to the plot
    legend_elements = create_legend()
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    animation = camera.animate()
    animation.save('modeldynamic.gif', writer='pillow')



def animate_party(Plista, times, party):
    fig = plt.figure()
    camera = Camera(fig)

    for i, _time in enumerate(times):
        plt.imshow(np.sort(create_party_visualization(Plista, _time)), cmap='magma', vmin=-1, vmax=1)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        plt.title(f"{party} opinion evolution")
        camera.snap()

    # Add legend to the plot
    legend_elements = create_legend()
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    animation = camera.animate()
    animation.save(f'{party}_modeldynamic.gif', writer='pillow')