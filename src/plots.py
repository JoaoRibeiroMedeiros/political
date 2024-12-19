
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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



def plot_volatilities(Experimento):

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(Experimento.volatilities)), Experimento.volatilities, alpha=0.5)
    plt.title('Volatilities Scatter Plot')
    plt.xlabel('Index')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()