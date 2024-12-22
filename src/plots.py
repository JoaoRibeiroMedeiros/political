
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

def plot_fluxes(Experimento, save = False):

    fig, axs = plt.subplots(3, figsize = (10,5))

    # AXES: A -> K ; K -> O ; A -> O;
    PhiAK = np.transpose(Experimento.fluxes)[0]
    PhiKO = np.transpose(Experimento.fluxes)[1]
    PhiAO = np.transpose(Experimento.fluxes)[2]

    fluxmax = np.max(np.concatenate([PhiAK,PhiKO,PhiAO]))
    fluxmin =np.min(np.concatenate([PhiAK,PhiKO,PhiAO]))

    plt.setp(axs, yticks=[-fluxmin, 0, fluxmax])

    axs[0].set_title(r'$\phi_{AK}$') 
    axs[0].plot(PhiAK)
    axs[0].set_ylim(-4,4)

    axs[1].set_title(r'$\phi_{KO}$') 
    axs[1].plot(PhiKO)
    axs[1].set_ylim(-4,4)

    axs[2].set_title(r'$\phi_{AO}$') 
    axs[2].plot(PhiAO)
    axs[2].set_ylim(-4,4)

    fig.tight_layout()

    if save:
        plt.savefig('Fluxes_λ_'+str(Experimento.l)+'_δ_'+str(Experimento.delta)+'.png')


def plot_vote_sets_evolution(Experimento, save = False):

    plt.figure(figsize=(10,7))

    plt.suptitle('Opinions Sets Evolution', fontsize = 20)

    plt.title('λ = %s, δ = %s'%(Experimento.l,Experimento.delta), fontsize = 16)
   
    plt.plot(Experimento.times, Experimento.A, label = 'number of congressmen in Λ', color = 'green')

    plt.plot(Experimento.times, Experimento.O, label = 'number of congressmen in Ω', color = 'red')

    plt.plot(Experimento.times,Experimento.K, label = 'number of congressmen in K', color = 'gray')

    plt.xlabel('Time', fontsize = 14)

    plt.ylabel('Number of Congressmen', fontsize = 14)

    plt.xticks(rotation=40)

    plt.legend()
    

    if save:
        plt.savefig('Opinions_Sets_Evolution_λ_'+str(Experimento.l)+'_δ_'+str(Experimento.delta)+'.png')



def create_visualization(Plista,t):

    mapa = np.zeros((18,18))

    k=0
    j=0

    for i in Plista[t]:
        
        if(k%18 ==0 and k!=0):
            k=0
            j+=1
        if i==0:
            mapa[k][j]=0.5
        else:
            mapa[k][j]=i
        k+=1
    
    return mapa


def create_animation(Plista, ordered = True):
    
    if ordered: 
        for i in range(5,120,5):
            plt.imshow(np.sort( create_visualization(Plista,i)), cmap = 'magma')
            #plt.imshow( create_visualization(Plista,i))
            plt.savefig('partisan' + str(i)+'.png')

    if ordered:
        for i in range(5,120,5):
            #plt.imshow(np.sort( create_visualization(Plista,i)))
            plt.imshow( create_visualization(Plista,i), cmap = 'magma')
            plt.savefig('partisanloc' + str(i)+'.png')


def plot_party_evolution(Experimento, party = 'PT', save = False):

    serie_A, serie_K, serie_O = Experimento.serie_temporal_partido(party)

    df_party = pd.concat([pd.DataFrame(x) for x in [Experimento.times, serie_A, serie_K, serie_O]],axis=1)

    df_party.columns = ['time','Λ','K','Ω']
    df_party = df_party.set_index('time',drop=True)

    df_party.plot(figsize=(10,7),color=['green', 'gray', 'red'])#    \n Lag=%s'%lag)

    plt.title('%s Opinions Sets Evolution'%party, fontsize = 20) 

    plt.xlabel('Time', fontsize = 14)

    plt.ylabel('Number of Congressmen', fontsize = 14)
    
    plt.tight_layout()
    if save:
        plt.savefig(party+'_Opinions_Sets_Evolution_λ_'+str(Experimento.l)+'_δ_'+str(Experimento.delta)+'.png')


def plot_volatilities(prob_analysis):
    plt.figure(figsize=(19, 6))

    threshold1 = 0.5
    threshold2 = 0.7

    colors = ['gray' if v < threshold1 else 'orange' if v < threshold2 else 'red' for v in prob_analysis.statement_volatilities]

    plt.scatter(range(len(prob_analysis.statement_volatilities)), prob_analysis.statement_volatilities, c=colors, alpha=0.5)

    plt.suptitle('Statement Volatilities Scatter Plot ', fontsize = 20 )
    plt.title('λ = %s, δ = %s'%(prob_analysis.l,prob_analysis.delta), fontsize = 16)
    plt.xlabel('Politician Id')
    plt.ylabel('Statement Volatility')
    plt.grid(True)

    # Create simple legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Low', markersize=10, markerfacecolor='gray', alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Medium', markersize=10, markerfacecolor='orange', alpha=0.5),
                       Line2D([0], [0], marker='o', color='w', label='High', markersize=10, markerfacecolor='red', alpha=0.5)]
    plt.legend(handles=legend_elements, title='Volatility Levels')

    plt.show()



def plot_prob_evolution(from_time_cut_to_probability, date_of_reckoning, name):

    # date_of_reckoning
    # Extract dates and probabilities for plotting
    dates = list(from_time_cut_to_probability.keys())
    A_probabilities = [from_time_cut_to_probability[date]['A'] for date in dates]
    O_probabilities = [from_time_cut_to_probability[date]['O'] for date in dates]
    # Plot the time series
    plt.figure(figsize=(12, 6))

    # Plot A probabilities with a darker green color
    plt.plot(dates, A_probabilities, label='A Probability', marker='o', color='#228B22')  # ForestGreen

    # Plot O probabilities with a darker red color
    plt.plot(dates, O_probabilities, label='O Probability', marker='o', color='#8B0000')  # DarkRed

    # Add the vertical line for the date of reckoning
    plt.axvline(x=date_of_reckoning, color='r', linestyle='--', label='Date of Reckoning')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.title(name + ' - Time Series of A and O Probabilities')

    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels if necessary
    plt.xticks(rotation=45)

    # Adjust layout to fit elements
    plt.tight_layout()

    # Show the plot
    plt.show()
